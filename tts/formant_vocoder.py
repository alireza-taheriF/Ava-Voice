"""Rule-based formant vocoder — Ava's first path from speech plan to sound.

This closes the "bridge to synthesis" gap: it renders an
:class:`~ava_voice.acoustics.acoustic_schema.AcousticProfile` (phoneme
timeline + F0 curve + energy curve) into 16-bit PCM using a classic
source–filter design (Klatt-style, heavily simplified):

* **Voicing source** — a Rosenberg glottal pulse driven by the plan's hertz
  F0 curve, differentiated to model lip radiation.
* **Vocal tract** — three cascaded second-order formant resonators whose
  targets come from a per-phoneme table and glide smoothly between phonemes.
* **Frication / bursts** — white noise through a band-pass resonator, gated
  per phoneme class (fricatives, stop bursts, affricates, /h/ aspiration).
* **Energy** — the plan's energy curve scales the output frame by frame;
  pause segments are silent.

Honest scope
------------
This is a *baseline*, not a voice. It proves the plan is renderable end to
end, keeps timing and intonation from the plan exactly, and gives
downstream systems (e.g. telephony) real PCM to carry. It is robotic, its
intelligibility has **not** been measured, and it is meant to be replaced by
a trained acoustic model + neural vocoder behind the same function
signature. The rule-based G2P's ``V`` placeholder (unresolved short vowel) is
rendered as a neutral central vowel; with the neural G2P backend no
placeholders occur.

Pure Python standard library — no NumPy, no model weights, deterministic for
a given ``seed``.
"""

from __future__ import annotations

import io
import math
import random
import wave
from array import array
from dataclasses import dataclass
from typing import Final, Iterable

from ava_voice.acoustics.acoustic_schema import AcousticProfile

__all__ = [
    "PhonemeSpec",
    "VocoderConfig",
    "PHONEME_TABLE",
    "synthesize_profile",
    "synthesize_plan",
    "to_pcm16",
    "to_wav_bytes",
]


@dataclass(frozen=True)
class PhonemeSpec:
    """Synthesis targets for one phoneme symbol.

    ``voice`` and ``noise`` are linear amplitudes in ``[0, 1]``; ``kind``
    selects the temporal shape (steady, stop closure + burst, affricate).
    """

    f1: float
    f2: float
    f3: float
    voice: float
    noise: float = 0.0
    noise_hz: float = 0.0
    noise_bw: float = 1500.0
    kind: str = "steady"  # steady | stop | affricate | silence


# Approximate targets for an adult speaker. Vowel values follow published
# Persian vowel-space measurements in rounded form; consonants use locus-style
# targets so transitions into neighbouring vowels sound plausible.
PHONEME_TABLE: Final[dict[str, PhonemeSpec]] = {
    # vowels
    "i": PhonemeSpec(300, 2250, 3000, 1.0),
    "e": PhonemeSpec(460, 1850, 2600, 1.0),
    "æ": PhonemeSpec(660, 1650, 2500, 1.0),
    "a": PhonemeSpec(720, 1150, 2500, 1.0),
    "o": PhonemeSpec(480, 900, 2450, 1.0),
    "u": PhonemeSpec(320, 800, 2350, 1.0),
    "V": PhonemeSpec(520, 1450, 2500, 0.9),  # unresolved short vowel → schwa
    # sonorants
    "m": PhonemeSpec(280, 1000, 2200, 0.45),
    "n": PhonemeSpec(280, 1500, 2500, 0.45),
    "l": PhonemeSpec(380, 1150, 2650, 0.6),
    "r": PhonemeSpec(460, 1300, 1800, 0.6),
    "y": PhonemeSpec(300, 2200, 3000, 0.7),
    "v": PhonemeSpec(320, 1150, 2300, 0.45, 0.12, 3500, 3000),
    # voiceless fricatives
    "f": PhonemeSpec(350, 1200, 2400, 0.0, 0.25, 4500, 4000),
    "s": PhonemeSpec(400, 1700, 2600, 0.0, 0.45, 5500, 2000),
    "sh": PhonemeSpec(400, 1900, 2500, 0.0, 0.5, 2900, 1500),
    "x": PhonemeSpec(450, 1300, 2400, 0.0, 0.35, 1600, 1200),
    "h": PhonemeSpec(500, 1500, 2500, 0.0, 0.18, 1500, 3000),
    # voiced fricatives
    "z": PhonemeSpec(300, 1700, 2600, 0.35, 0.3, 5200, 2000),
    "zh": PhonemeSpec(300, 1900, 2500, 0.35, 0.3, 2800, 1500),
    "gh": PhonemeSpec(400, 1200, 2400, 0.4, 0.15, 1400, 1200),
    # stops (closure + burst)
    "p": PhonemeSpec(300, 900, 2300, 0.0, 0.35, 1000, 2000, "stop"),
    "b": PhonemeSpec(300, 900, 2300, 0.25, 0.25, 1000, 2000, "stop"),
    "t": PhonemeSpec(350, 1700, 2600, 0.0, 0.4, 4000, 3000, "stop"),
    "d": PhonemeSpec(350, 1700, 2600, 0.25, 0.3, 4000, 3000, "stop"),
    "k": PhonemeSpec(350, 2000, 2600, 0.0, 0.4, 2200, 1500, "stop"),
    "g": PhonemeSpec(350, 2000, 2600, 0.25, 0.3, 2200, 1500, "stop"),
    "q": PhonemeSpec(400, 1200, 2400, 0.0, 0.4, 1400, 1200, "stop"),
    # affricates
    "ch": PhonemeSpec(350, 1900, 2500, 0.0, 0.45, 2900, 1500, "affricate"),
    "j": PhonemeSpec(350, 1900, 2500, 0.25, 0.35, 2900, 1500, "affricate"),
    # glottal stop
    "'": PhonemeSpec(500, 1500, 2500, 0.0, 0.0, kind="silence"),
}
_FALLBACK: Final[PhonemeSpec] = PhonemeSpec(500, 1500, 2500, 0.3)
_BANDWIDTHS: Final[tuple[float, float, float]] = (80.0, 110.0, 170.0)


@dataclass(frozen=True)
class VocoderConfig:
    """Rendering options.

    ``smoothing_ms`` sets how quickly formant targets glide between phonemes;
    ``seed`` makes the noise source deterministic.
    """

    sample_rate: int = 16_000
    gain: float = 0.6
    smoothing_ms: float = 18.0
    open_quotient: float = 0.6
    seed: int = 0


class _Resonator:
    """Second-order digital resonator (Klatt 1980) with per-sample retuning."""

    __slots__ = ("sr", "y1", "y2")

    def __init__(self, sr: int) -> None:
        self.sr = sr
        self.y1 = 0.0
        self.y2 = 0.0

    def step(self, x: float, freq: float, bw: float) -> float:
        sr = self.sr
        c = -math.exp(-2.0 * math.pi * bw / sr)
        b = 2.0 * math.exp(-math.pi * bw / sr) * math.cos(2.0 * math.pi * freq / sr)
        a = 1.0 - b - c
        y = a * x + b * self.y1 + c * self.y2
        self.y2 = self.y1
        self.y1 = y
        return y


def _sample_curve(values: tuple[float, ...] | list[float], hop_ms: float, t_ms: float) -> float:
    if not values:
        return 0.0
    pos = t_ms / hop_ms
    i = int(pos)
    if i >= len(values) - 1:
        return float(values[-1])
    frac = pos - i
    return float(values[i]) * (1.0 - frac) + float(values[i + 1]) * frac


def _control_tracks(profile: AcousticProfile, n_ctrl: int) -> list[tuple[PhonemeSpec, float, float]]:
    """Per-millisecond (spec, voice_gate, noise_gate) assignments."""
    silence = PhonemeSpec(500, 1500, 2500, 0.0, kind="silence")
    track: list[tuple[PhonemeSpec, float, float]] = [(silence, 0.0, 0.0)] * n_ctrl
    for ph in profile.phoneme_durations:
        spec = PHONEME_TABLE.get(ph.symbol, _FALLBACK)
        start = max(0, int(ph.start_ms))
        end = min(n_ctrl, max(start + 1, int(ph.end_ms)))
        length = end - start
        for k in range(start, end):
            rel = (k - start) / max(1, length)
            voice, noise = spec.voice, spec.noise
            if spec.kind == "stop":
                # closure (voice bar only) for 65 %, then a burst that decays.
                if rel < 0.65:
                    voice, noise = spec.voice * 0.5, 0.0
                else:
                    voice = spec.voice
                    noise = spec.noise * math.exp(-(rel - 0.65) * 8.0)
            elif spec.kind == "affricate":
                if rel < 0.4:
                    voice, noise = spec.voice * 0.4, 0.0
            elif spec.kind == "silence":
                voice, noise = 0.0, 0.0
            track[k] = (spec, voice, noise)
    return track


def _smooth(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out: list[float] = []
    acc = 0.0
    half = window // 2
    n = len(values)
    # centred moving average via prefix sums
    prefix = [0.0]
    for v in values:
        acc += v
        prefix.append(acc)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append((prefix[hi] - prefix[lo]) / (hi - lo))
    return out


def synthesize_profile(
    profile: AcousticProfile, config: VocoderConfig | None = None
) -> array:
    """Render one sentence's acoustic profile to float samples in ``[-1, 1]``.

    Returns an ``array('f')`` of length ``round(total_duration_ms * sr / 1000)``.
    """
    cfg = config or VocoderConfig()
    sr = cfg.sample_rate
    total_ms = float(profile.total_duration_ms)
    n_ctrl = max(1, int(math.ceil(total_ms)))
    n_samples = int(round(total_ms * sr / 1000.0))
    spf = sr / 1000.0  # samples per control millisecond

    track = _control_tracks(profile, n_ctrl)
    win = max(1, int(cfg.smoothing_ms))
    f1 = _smooth([t[0].f1 for t in track], win)
    f2 = _smooth([t[0].f2 for t in track], win)
    f3 = _smooth([t[0].f3 for t in track], win)
    vg = _smooth([t[1] for t in track], max(1, win // 3))
    ng = _smooth([t[2] for t in track], max(1, win // 4))
    nhz = [t[0].noise_hz or 1500.0 for t in track]
    nbw = [t[0].noise_bw for t in track]

    pitch = profile.pitch_curve_hz
    energy = profile.energy_curve
    p_vals, p_hop = list(pitch.values), float(pitch.hop_ms)
    e_vals, e_hop = list(energy.values), float(energy.hop_ms)
    default_f0 = float(profile.speaker_f0_hz) or 150.0

    paused = [False] * n_ctrl
    for seg in profile.pause_segments:
        for k in range(max(0, int(seg.start_ms)), min(n_ctrl, int(seg.end_ms))):
            paused[k] = True

    rng = random.Random(cfg.seed)
    r1, r2, r3 = _Resonator(sr), _Resonator(sr), _Resonator(sr)
    rn = _Resonator(sr)
    b1, b2, b3 = _BANDWIDTHS
    oq = cfg.open_quotient
    tp = oq * 0.66  # opening phase
    tn = oq - tp  # closing phase

    out = array("f", bytes(4 * n_samples))
    phase = 0.0
    prev_flow = 0.0
    for n in range(n_samples):
        t_ms = n / spf
        k = min(n_ctrl - 1, int(t_ms))
        if paused[k]:
            r1.step(0.0, f1[k], b1)
            r2.step(0.0, f2[k], b2)
            r3.step(0.0, f3[k], b3)
            rn.step(0.0, nhz[k], nbw[k])
            out[n] = 0.0
            prev_flow = 0.0
            continue
        f0 = _sample_curve(p_vals, p_hop, t_ms) or default_f0
        phase += f0 / sr
        if phase >= 1.0:
            phase -= 1.0
        # Rosenberg glottal flow, then differentiate (lip radiation).
        if phase < tp:
            flow = 0.5 * (1.0 - math.cos(math.pi * phase / tp))
        elif phase < oq:
            flow = math.cos(0.5 * math.pi * (phase - tp) / tn)
        else:
            flow = 0.0
        src = (flow - prev_flow) * (sr / max(f0, 50.0)) * 0.08
        prev_flow = flow

        voiced = src * vg[k]
        white = rng.uniform(-1.0, 1.0)
        aspir = white * 0.02 * vg[k]
        y = r1.step(voiced + aspir, f1[k], b1)
        y = r2.step(y, f2[k], b2)
        y = r3.step(y, f3[k], b3)
        fric = rn.step(white, nhz[k], nbw[k]) * ng[k] * 3.0
        amp = _sample_curve(e_vals, e_hop, t_ms)
        out[n] = (y + fric) * amp

    peak = max((abs(v) for v in out), default=0.0)
    if peak > 0.0:
        scale = cfg.gain / peak
        for i in range(n_samples):
            out[i] *= scale
    return out


def synthesize_plan(
    plan: object,
    config: VocoderConfig | None = None,
    sentence_gap_ms: float = 120.0,
) -> array:
    """Render every sentence of a :class:`~ava_voice.core.speech_plan.SpeechPlan`.

    Sentences are concatenated with ``sentence_gap_ms`` of silence.
    """
    cfg = config or VocoderConfig()
    gap = array("f", bytes(4 * int(cfg.sample_rate * sentence_gap_ms / 1000.0)))
    out = array("f")
    sentences: Iterable = getattr(plan, "sentences")
    for i, sentence in enumerate(sentences):
        if i:
            out.extend(gap)
        out.extend(synthesize_profile(sentence.acoustic, cfg))
    return out


def to_pcm16(samples: array | list[float]) -> bytes:
    """Clip float samples to ``[-1, 1]`` and pack as little-endian int16."""
    pcm = array("h", (int(max(-1.0, min(1.0, s)) * 32767) for s in samples))
    if pcm.itemsize != 2:  # pragma: no cover - exotic platforms
        raise RuntimeError("int16 array required")
    import sys

    if sys.byteorder != "little":  # pragma: no cover
        pcm.byteswap()
    return pcm.tobytes()


def to_wav_bytes(samples: array | list[float], sample_rate: int = 16_000) -> bytes:
    """Encode float samples as a mono 16-bit PCM WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(to_pcm16(samples))
    return buf.getvalue()


def _main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m ava_voice.tts.formant_vocoder "متن" out.wav``."""
    import argparse

    from ava_voice.core.speech_plan import plan_speech

    ap = argparse.ArgumentParser(description="Render Persian text to WAV (baseline formant vocoder).")
    ap.add_argument("text")
    ap.add_argument("out", help="output .wav path")
    ap.add_argument("--sample-rate", type=int, default=16_000)
    args = ap.parse_args(argv)
    cfg = VocoderConfig(sample_rate=args.sample_rate)
    samples = synthesize_plan(plan_speech(args.text), cfg)
    with open(args.out, "wb") as fh:
        fh.write(to_wav_bytes(samples, cfg.sample_rate))
    print(f"wrote {args.out}: {len(samples) / cfg.sample_rate:.2f} s @ {cfg.sample_rate} Hz")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
