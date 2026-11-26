"""Tests for the baseline formant vocoder (plan → PCM)."""

from __future__ import annotations

import io
import wave

from ava_voice.core.speech_plan import plan_speech
from ava_voice.tts.formant_vocoder import (
    PHONEME_TABLE,
    VocoderConfig,
    synthesize_plan,
    synthesize_profile,
    to_wav_bytes,
)
from ava_voice.linguistics.phonemizer import VOWEL_SYMBOLS


def _rms(xs) -> float:
    xs = list(xs)
    return (sum(x * x for x in xs) / max(1, len(xs))) ** 0.5


def _autocorr_f0(xs, sr: int) -> float:
    xs = list(xs)
    best_lag, best = 0, float("-inf")
    for lag in range(sr // 400, sr // 60):
        c = sum(xs[i] * xs[i + lag] for i in range(len(xs) - lag))
        if c > best:
            best, best_lag = c, lag
    return sr / best_lag


def test_every_vowel_symbol_has_a_spec() -> None:
    assert VOWEL_SYMBOLS <= set(PHONEME_TABLE)


def test_length_matches_plan_duration() -> None:
    profile = plan_speech("سلام").sentences[0].acoustic
    out = synthesize_profile(profile)
    expected = round(profile.total_duration_ms * 16_000 / 1000)
    assert len(out) == expected


def test_output_is_bounded_and_deterministic() -> None:
    profile = plan_speech("امروز هوا خوب است.").sentences[0].acoustic
    a = synthesize_profile(profile, VocoderConfig(seed=7))
    b = synthesize_profile(profile, VocoderConfig(seed=7))
    assert list(a) == list(b)
    assert max(abs(x) for x in a) <= 0.6 + 1e-6
    assert _rms(a) > 0.01


def test_pitch_follows_the_plan_f0() -> None:
    profile = plan_speech("آب").sentences[0].acoustic
    sr = 16_000
    out = synthesize_profile(profile)
    first = profile.phoneme_durations[0]
    assert first.symbol == "a"
    start = int((first.start_ms + 30) * sr / 1000)
    window = out[start : start + 800]
    planned = profile.pitch_curve_hz.values[int((first.start_ms + 55) / 10)]
    est = _autocorr_f0(window, sr)
    assert abs(est - planned) / planned < 0.05


def test_pause_segments_are_silent() -> None:
    plan = plan_speech("امروز هوا خیلی خوب است، فردا باران می‌بارد.")
    profile = plan.sentences[0].acoustic
    assert profile.pause_segments, "test sentence must contain a pause"
    seg = profile.pause_segments[0]
    out = synthesize_profile(profile)
    lo = int((seg.start_ms + 5) * 16)
    hi = int((seg.end_ms - 5) * 16)
    assert hi > lo
    assert max(abs(x) for x in out[lo:hi]) == 0.0


def test_plan_and_wav_roundtrip() -> None:
    plan = plan_speech("سلام! خوب هستی؟")
    out = synthesize_plan(plan)
    data = to_wav_bytes(out, 16_000)
    with wave.open(io.BytesIO(data)) as w:
        assert w.getframerate() == 16_000
        assert w.getsampwidth() == 2
        assert w.getnchannels() == 1
        assert w.getnframes() == len(out)
