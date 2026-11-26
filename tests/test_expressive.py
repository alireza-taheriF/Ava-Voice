"""Tests for expressive rendering: pitch tracking, PSOLA, breath, emotion → style."""

from __future__ import annotations

import numpy as np
import pytest

from ava_voice.core.speech_plan import plan_speech
from ava_voice.tts import dsp
from ava_voice.tts.expressive import BACKCHANNELS, ExpressiveVoice, VoiceStyle, style_from_profile

SR = 22050


def harmonic(f0: float, seconds: float = 1.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.1 * sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 15))).astype(np.float32)


def test_track_f0_finds_pitch_and_silence() -> None:
    x = np.concatenate([harmonic(150, 0.5), np.zeros(SR // 2, np.float32)])
    f0, voiced = dsp.track_f0(x, SR)
    assert abs(np.median(f0[voiced]) - 150) < 1.0
    assert voiced[: len(voiced) // 3].all() and not voiced[-len(voiced) // 4 :].any()


@pytest.mark.parametrize("semitones", [-3.0, 2.0, 4.0])
def test_psola_shifts_pitch_and_keeps_duration(semitones: float) -> None:
    x = harmonic(140)
    y = dsp.psola(x, SR, 2 ** (semitones / 12))
    assert len(y) == len(x)
    f1, v1 = dsp.track_f0(y, SR)
    assert abs(np.median(f1[v1]) / 140 - 2 ** (semitones / 12)) < 0.01


def test_psola_keeps_formants() -> None:
    # A strong resonance at 1 kHz must stay at 1 kHz when pitch moves up 4 st.
    t = np.arange(SR) / SR
    x = sum(np.sin(2 * np.pi * 125 * k * t) * np.exp(-((125 * k - 1000) / 300) ** 2) for k in range(1, 40))
    x = (0.2 * x / np.max(np.abs(x))).astype(np.float32)
    y = dsp.psola(x, SR, 2 ** (4 / 12))
    def centroid(sig: np.ndarray) -> float:
        s = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        f = np.fft.rfftfreq(len(sig), 1 / SR)
        return float((s * f).sum() / s.sum())
    assert abs(centroid(y) - centroid(x)) / centroid(x) < 0.1  # resampling would move it ~26 %


def test_psola_time_varying_ratio_rises() -> None:
    x = harmonic(120, 1.2)
    ratio = np.linspace(1.0, 2 ** (3 / 12), len(x))
    f1, v1 = dsp.track_f0(dsp.psola(x, SR, ratio), SR)
    q = len(f1) // 5
    assert np.median(f1[:q][v1[:q]]) < 125 < 135 < np.median(f1[-q:][v1[-q:]])


@pytest.mark.parametrize("kind", ["inhale", "exhale"])
def test_breath_level_and_shape(kind: str) -> None:
    b = dsp.breath(SR, 400, -34.0, kind, seed=3)
    assert len(b) == int(SR * 0.4)
    assert abs(dsp.rms_db(b) - (-34.0)) < 0.5
    third = len(b) // 3
    first, last = dsp.rms_db(b[:third]), dsp.rms_db(b[-third:])
    assert (first < last) if kind == "inhale" else (first > last)  # inhale swells, exhale fades
    spec = np.abs(np.fft.rfft(b))
    f = np.fft.rfftfreq(len(b), 1 / SR)
    assert spec[(f > 800) & (f < 3500)].mean() > 3 * spec[f > 7000].mean()  # vocal-tract band


def test_concat_fades_edges() -> None:
    out = dsp.concat([np.ones(1000, np.float32), np.ones(1000, np.float32)], SR, fade_ms=5)
    assert out[0] == 0 and out[999] < 0.01 and out[500] == 1


def test_emotion_moves_style_in_the_right_direction() -> None:
    joy = style_from_profile(plan_speech("عالیه! خیلی خوشحالم که تماس گرفتید!").sentences[0].prosody.profile)
    sad = style_from_profile(plan_speech("متأسفانه دکتر امروز حضور ندارند.").sentences[0].prosody.profile)
    neutral = style_from_profile(plan_speech("نوبت شما ثبت شد.").sentences[0].prosody.profile)
    assert joy.pitch_semitones > neutral.pitch_semitones > sad.pitch_semitones
    assert joy.length_scale < neutral.length_scale < sad.length_scale  # joy faster, sadness slower
    assert joy.gain_db > neutral.gain_db
    assert joy.noise_scale > neutral.noise_scale  # livelier delivery
    flat = style_from_profile(plan_speech("نوبت شما ثبت شد.").sentences[0].prosody.profile, intensity=0.0)
    assert flat.pitch_semitones == 0 and flat.length_scale == 1.0


class _ToneVoice:
    """Stand-in for PersianVITS: a 130 Hz harmonic tone, 60 ms per IPA char."""

    sample_rate = SR

    def __init__(self) -> None:
        self.calls: list[tuple[str, float | None]] = []

    def phonemize(self, text: str) -> str:
        return text

    def synthesize_ipa(self, ipa, length_scale=None, noise_scale=None, noise_w=None):
        self.calls.append((ipa, length_scale))
        return harmonic(130, 0.06 * len(ipa.replace(" ", "")) * (length_scale or 1.0))


def test_expressive_render_with_breaths_and_backchannels() -> None:
    voice = _ToneVoice()
    ev = ExpressiveVoice(voice, seed=0)
    text = "سلام! خیلی خوشحالم که امروز با کلینیک ما تماس گرفتید. متأسفانه دکتر حضور ندارند."
    with_breath = ev.render(text, leading_breath=True)
    voice2 = _ToneVoice()
    no_breath = ExpressiveVoice(voice2, breaths=False, seed=0).render(text)
    assert len(with_breath) > len(no_breath)
    assert len(voice.calls) == 3  # one synthesis per sentence
    # joyful second sentence faster than the sad third
    assert voice.calls[1][1] < voice.calls[2][1]
    # the forced leading breath is quiet and comes first
    head = with_breath[: int(SR * 0.2)]
    assert dsp.rms_db(head) < -20
    a = ev.backchannel("ahan")
    assert ev.backchannel("ahan") is a  # cached
    assert set(BACKCHANNELS) >= {"bale", "ahan", "ohom", "khob"}
    assert isinstance(VoiceStyle(), VoiceStyle)
