"""Audio helpers for voice cloning (NumPy only): WAV I/O, resampling, QC."""

from __future__ import annotations

import io
import wave
from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["read_wav", "write_wav", "resample", "QC", "analyze", "trim_and_normalize"]

TARGET_RATE = 22_050


def read_wav(path_or_bytes) -> tuple[np.ndarray, int]:
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else str(path_or_bytes)
    with wave.open(src, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("16-bit PCM WAV required")
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def write_wav(path, x: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Band-limited resampling via FFT (fine for offline dataset preparation)."""
    if src == dst or len(x) == 0:
        return np.asarray(x, np.float32)
    n = int(round(len(x) * dst / src))
    X = np.fft.rfft(x)
    keep = min(len(X), n // 2 + 1)
    Y = np.zeros(n // 2 + 1, complex)
    Y[:keep] = X[:keep]
    return (np.fft.irfft(Y, n) * (n / len(x))).astype(np.float32)


def _frame_db(x: np.ndarray, sr: int, ms: float = 20.0) -> np.ndarray:
    hop = max(1, int(sr * ms / 1000))
    n = len(x) // hop
    if n == 0:
        return np.array([-120.0])
    f = x[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(f * f, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-6))


@dataclass
class QC:
    duration_s: float
    peak_db: float
    clipped_pct: float
    speech_db: float
    noise_db: float
    snr_db: float
    ok: bool
    problems: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def analyze(x: np.ndarray, sr: int) -> QC:
    """Recording quality: level, clipping and a frame-percentile SNR estimate."""
    dur = len(x) / sr
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    clipped = float(np.mean(np.abs(x) > 0.985) * 100) if len(x) else 0.0
    db = _frame_db(x, sr)
    speech = float(np.percentile(db, 90))
    noise = float(np.percentile(db, 10))
    snr = speech - noise
    problems = []
    if dur < 0.6:
        problems.append("خیلی کوتاه است")
    if dur > 25:
        problems.append("خیلی طولانی است")
    if clipped > 0.1:
        problems.append("صدا زیادی بلند است و دیستورت شده؛ کمی از میکروفون فاصله بگیر")
    if speech < -38:
        problems.append("صدا خیلی آرام است؛ نزدیک‌تر یا بلندتر صحبت کن")
    if snr < 25:
        problems.append("نویز پس‌زمینه زیاد است؛ جای ساکت‌تری ضبط کن")
    return QC(round(dur, 2), round(20 * np.log10(max(peak, 1e-6)), 1), round(clipped, 3),
              round(speech, 1), round(noise, 1), round(snr, 1), not problems, problems)


def trim_and_normalize(x: np.ndarray, sr: int, pad_ms: float = 150.0, target_db: float = -20.0) -> np.ndarray:
    """Cut leading/trailing silence (keeping ``pad_ms``) and set the speech RMS to ``target_db``."""
    db = _frame_db(x, sr, 10.0)
    hop = int(sr * 0.01)
    thr = max(float(np.percentile(db, 10)) + 12.0, -55.0)
    idx = np.nonzero(db > thr)[0]
    if len(idx):
        pad = int(sr * pad_ms / 1000)
        x = x[max(0, idx[0] * hop - pad) : min(len(x), (idx[-1] + 1) * hop + pad)]
    active = db[db > thr]
    if len(active):
        cur = 20 * np.log10(np.mean(10 ** (active / 20)))
        x = x * 10 ** ((target_db - cur) / 20)
    peak = float(np.max(np.abs(x))) if len(x) else 0
    if peak > 0.95:
        x = x * (0.95 / peak)
    return x.astype(np.float32)
