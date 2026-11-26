"""Whisper-on-onnxruntime frontend tests; the model test needs AVA_WHISPER_DIR."""

import os

import numpy as np
import pytest

from ava_voice.asr.whisper_onnx import _mel_filters, log_mel_spectrogram


def test_mel_filterbank_shape_and_order() -> None:
    fb = _mel_filters(128)
    assert fb.shape == (128, 201)
    assert (fb >= 0).all()
    peaks = fb.argmax(axis=1)
    assert np.all(np.diff(peaks) >= 0)  # centre frequencies increase
    assert (fb.sum(axis=1) > 0).all()  # no empty filter at 16 kHz / 400-pt FFT


def test_log_mel_frames_and_range() -> None:
    sr = 16000
    t = np.arange(sr) / sr
    mel = log_mel_spectrogram(0.5 * np.sin(2 * np.pi * 1000 * t), 80)
    assert mel.shape == (80, 100)  # 10 ms hop, last frame dropped
    assert mel.max() <= 1.5 and mel.min() >= mel.max() - 2.0 - 1e-6  # 8 dB-decade clamp, /4 scaling
    # energy concentrates in the band containing 1 kHz
    band = mel.mean(axis=1).argmax()
    assert 15 < band < 40


@pytest.mark.skipif(not os.environ.get("AVA_WHISPER_DIR"), reason="set AVA_WHISPER_DIR to run Whisper")
def test_whisper_transcribes_ava_neural_voice() -> None:
    from ava_voice.asr.whisper_onnx import WhisperOnnxASR
    from ava_voice.tts.vits import PersianVITS

    tts = PersianVITS()
    asr = WhisperOnnxASR(os.environ["AVA_WHISPER_DIR"])
    text = asr.transcribe(tts.synthesize("خواهش می‌کنم. روز خوبی داشته باشید."), tts.sample_rate)
    assert "خوب" in text and "روز" in text
