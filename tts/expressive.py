"""Expressive, human-like rendering on top of the neural voice.

:class:`ExpressiveVoice` turns text into speech the way Ava's design intends:
**emotion → prosody → voice**. For each sentence:

1. Ava's emotion and prosody engines (:func:`~ava_voice.core.speech_plan.plan_speech`)
   produce a :class:`~ava_voice.prosody.prosody_schema.ProsodyProfile`.
2. :func:`style_from_profile` maps that profile onto controls the voice
   actually has:

   ==================  ===========================================  =====================
   profile field       effect                                       mechanism
   ==================  ===========================================  =====================
   ``speaking_rate``   faster / slower speech                       VITS ``length_scale``
   ``pitch_variance``  livelier vs. flatter delivery                VITS ``noise_scale``,
                                                                    PSOLA range expansion
   ``pitch_base``      higher (joy) / lower (sadness) voice         PSOLA mean shift
   ``contour_shape``   rising, falling, oscillating, broken melody  PSOLA time-varying ratio
   ``energy_level``    louder / softer                              gain
   ``pause_duration``  longer / shorter breaks between sentences    inserted silence
   ``pause_density``   how often the speaker breathes               breath placement
   ==================  ===========================================  =====================

3. Breaths are inserted the way people take them: an inhale before a longer
   sentence or after a long pause, and a soft exhale (sigh) before sad
   sentences, placed relative to the speech level.

:class:`ExpressiveVoice` also renders **backchannels** ("بله"، "آهان"، "اوهوم"،
"خب") and bare breaths, cached, for conversational systems that need to react
the moment the other person stops talking.
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass
from typing import Final

import numpy as np

from ava_voice.tts import dsp

__all__ = ["VoiceStyle", "style_from_profile", "ExpressiveVoice", "BACKCHANNELS"]


@dataclass(frozen=True)
class VoiceStyle:
    length_scale: float = 1.0
    noise_scale: float = 0.35
    noise_w: float = 0.55
    pitch_semitones: float = 0.0
    range_gain: float = 0.0  # >0 widens the F0 excursion around its mean
    contour: str = "falling"
    contour_semitones: float = 0.0  # size of the end-of-sentence bend
    gain_db: float = 0.0
    pause_scale: float = 1.0
    breathiness: float = 0.3  # probability-like weight for breath insertion


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def style_from_profile(profile: object, intensity: float = 1.0) -> VoiceStyle:
    """Map an Ava prosody profile onto voice controls.

    ``intensity`` scales how strongly emotion is expressed (0 = neutral
    reading, 1 = default, up to ~1.5 = theatrical).
    """
    g = lambda name, default: float(getattr(profile, name, default))  # noqa: E731
    rate = g("speaking_rate", 1.0)
    var = g("pitch_variance", 0.2)
    base = g("pitch_base", 0.0)
    energy = g("energy_level", 0.4)
    contour = str(getattr(getattr(profile, "contour_shape", "falling"), "value", getattr(profile, "contour_shape", "falling")))
    k = _clip(intensity, 0.0, 2.0)
    return VoiceStyle(
        length_scale=_clip(1.0 / (1.0 + k * (rate - 1.0)), 0.8, 1.25),
        noise_scale=_clip(0.35 + k * 0.6 * (var - 0.2), 0.25, 0.6),
        noise_w=_clip(0.55 + k * 0.4 * (var - 0.2), 0.4, 0.8),
        pitch_semitones=_clip(k * 0.9 * base, -3.0, 3.0),
        range_gain=_clip(k * 1.4 * (var - 0.2), -0.35, 0.6),
        contour=contour,
        contour_semitones=_clip(k * (1.0 + 4.0 * var), 0.0, 3.0),
        gain_db=_clip(k * 9.0 * (energy - 0.4), -4.0, 3.0),
        pause_scale=_clip(g("pause_duration", 1.0), 0.6, 1.8),
        breathiness=_clip(0.25 + 0.6 * g("pause_density", 0.2), 0.0, 0.85),
    )


#: Conversational backchannels: text for logging, IPA fed straight to the voice
#: (interjections are outside G2P training data, so they are spelled here).
BACKCHANNELS: Final[dict[str, tuple[str, str]]] = {
    "bale": ("بله،", "baˈle,"),
    "ahan": ("آهان،", "ʔɑːˈhɑːn,"),
    "ohom": ("اوهوم،", "ʔuˈhum,"),
    "khob": ("خب،", "ˈxob,"),
    "bashe": ("باشه،", "bɑːˈʃe,"),
}


class ExpressiveVoice:
    """Emotion-driven, breathing renderer around :class:`~ava_voice.tts.vits.PersianVITS`."""

    def __init__(self, voice: object | None = None, intensity: float = 1.0, breaths: bool = True,
                 seed: int | None = None, speed: float = 1.0, pitch_semitones: float = 0.0) -> None:
        if voice is None:
            from ava_voice.tts.vits import PersianVITS

            voice = PersianVITS()
        self.voice = voice
        self.sample_rate: int = voice.sample_rate
        self.intensity = intensity
        self.breaths = breaths
        #: Global delivery: ``speed`` > 1 talks faster, ``pitch_semitones`` > 0
        #: raises the voice (formant-preserving) — e.g. a younger, livelier
        #: sound for a deep, slow voice. Applied on top of per-sentence emotion.
        self.speed = speed
        self.pitch_semitones = pitch_semitones
        self._rng = random.Random(seed)
        self._cache: dict[tuple[str, int], np.ndarray] = {}
        self._lock = threading.Lock()

    # -------------------------------------------------------------- pieces
    def _contour_ratio(self, n: int, f0: np.ndarray, voiced: np.ndarray, hop: int, style: VoiceStyle) -> np.ndarray:
        """Per-sample pitch ratio: mean shift × range expansion × contour bend."""
        st = np.full(n, style.pitch_semitones, np.float64)
        t = np.linspace(0.0, 1.0, n)
        c, a = style.contour, style.contour_semitones
        if c == "rising":
            st += a * np.clip((t - 0.6) / 0.4, 0, 1) ** 1.5
        elif c == "falling":
            st -= 0.5 * a * np.clip((t - 0.7) / 0.3, 0, 1)
        elif c == "oscillating":
            st += 0.4 * a * np.sin(2 * np.pi * 2.5 * t)
        elif c == "broken":
            steps = np.repeat(np.array([0.0, 0.6, -0.4, 0.8, -0.2]) * a, int(np.ceil(n / 5)))[:n]
            st += steps
        ratio = 2.0 ** (st / 12.0)
        if style.range_gain and voiced.any():
            mean = float(np.exp(np.mean(np.log(f0[voiced]))))
            dev = np.ones(len(f0))
            dev[voiced] = (f0[voiced] / mean) ** style.range_gain  # widen/narrow around mean
            # hold the last voiced value through unvoiced gaps, then upsample
            last = 1.0
            for i in range(len(dev)):
                if voiced[i]:
                    last = dev[i]
                else:
                    dev[i] = last
            idx = np.minimum(np.arange(n) // hop, len(dev) - 1)
            ratio *= dev[idx]
        return ratio

    def _sentence(self, text: str, style: VoiceStyle) -> np.ndarray:
        from dataclasses import replace

        style = replace(style, length_scale=style.length_scale / max(0.5, self.speed),
                        pitch_semitones=style.pitch_semitones + self.pitch_semitones)
        v = self.voice
        ipa = v.phonemize(text)
        wav = v.synthesize_ipa(ipa, length_scale=style.length_scale, noise_scale=style.noise_scale,
                               noise_w=style.noise_w)
        if not len(wav):
            return wav
        needs_pitch = abs(style.pitch_semitones) > 0.05 or style.contour_semitones > 0.05 or abs(style.range_gain) > 0.02
        if needs_pitch:
            hop = int(self.sample_rate * 0.005)
            f0, voiced = dsp.track_f0(wav, self.sample_rate, 5.0)
            ratio = self._contour_ratio(len(wav), f0, voiced, hop, style)
            wav = dsp.psola(wav, self.sample_rate, ratio, 5.0, f0, voiced)
        return dsp.apply_gain_db(wav, style.gain_db)

    def breath(self, kind: str = "inhale", ms: float | None = None, relative_db: float = -24.0,
               speech_db: float = -20.0) -> np.ndarray:
        """A breath ``relative_db`` below a typical speech level."""
        ms = ms if ms is not None else self._rng.uniform(260, 420) if kind == "inhale" else self._rng.uniform(380, 600)
        return dsp.breath(self.sample_rate, ms, speech_db + relative_db, kind, seed=self._rng.randrange(1 << 30))

    def backchannel(self, key: str) -> np.ndarray:
        """Cached short reaction ("بله،", "آهان،" …), spoken softly and a bit slower."""
        with self._lock:
            hit = self._cache.get((key, 0))
        if hit is not None:
            return hit
        _, ipa = BACKCHANNELS[key]
        wav = self.voice.synthesize_ipa(ipa, length_scale=1.12, noise_scale=0.45, noise_w=0.6)
        wav = dsp.apply_gain_db(wav, -3.0)
        with self._lock:
            self._cache[(key, 0)] = wav
        return wav

    # -------------------------------------------------------------- render
    def render(self, text: str, leading_breath: bool | None = None) -> np.ndarray:
        """Text → expressive float32 speech at :attr:`sample_rate`.

        ``leading_breath``: force (True) or suppress (False) an inhale before
        the first sentence; ``None`` lets the breathing policy decide.
        """
        from ava_voice.core.speech_plan import plan_speech

        sr = self.sample_rate
        plan = plan_speech(text)
        chunks: list[np.ndarray] = []
        for i, s in enumerate(plan.sentences):
            style = style_from_profile(s.prosody.profile, self.intensity)
            words = len(s.text.split())
            if i:
                chunks.append(dsp.silence(sr, 180 * style.pause_scale / max(0.5, self.speed)))
            take = False
            if self.breaths:
                if i == 0 and leading_breath is not None:
                    take = leading_breath
                else:
                    take = words >= 7 and self._rng.random() < style.breathiness + (0.3 if i else 0.0)
            if take:
                sad = style.pitch_semitones < -0.3 and style.length_scale > 1.02
                chunks.append(self.breath("exhale" if sad else "inhale"))
                chunks.append(dsp.silence(sr, self._rng.uniform(40, 90)))
            chunks.append(self._sentence(s.text, style))
        return dsp.concat(chunks, sr)


def _main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m ava_voice.tts.expressive "متن" out.wav [--intensity 1.2]``."""
    import argparse
    import wave

    ap = argparse.ArgumentParser(description="Persian text → expressive, breathing speech (WAV)")
    ap.add_argument("text")
    ap.add_argument("out")
    ap.add_argument("--voice-dir")
    ap.add_argument("--g2p-dir")
    ap.add_argument("--intensity", type=float, default=1.0)
    ap.add_argument("--no-breath", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    from ava_voice.tts.vits import PersianVITS

    ev = ExpressiveVoice(PersianVITS(args.voice_dir, args.g2p_dir), args.intensity, not args.no_breath, args.seed)
    wav = ev.render(args.text, leading_breath=not args.no_breath)
    with wave.open(args.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(ev.sample_rate)
        w.writeframes((np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes())
    print(f"wrote {args.out}: {len(wav) / ev.sample_rate:.2f} s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
