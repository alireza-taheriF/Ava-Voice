"""Signal processing for expressive speech: pitch tracking, TD-PSOLA, breath.

Everything here is NumPy only and works on mono float32 audio in ``[-1, 1]``.

* :func:`track_f0` — frame-wise F0 with a normalized-autocorrelation detector
  (YIN-style cumulative-mean normalization), voiced/unvoiced decision included.
* :func:`psola` — time-domain pitch-synchronous overlap-add. It changes pitch
  by a **time-varying ratio** while keeping duration and, unlike resampling,
  **keeping the formants**: the voice gets higher or lower without turning into
  a chipmunk. That is what lets Ava's emotion engine move mean pitch, widen or
  narrow the intonation range, and bend the sentence contour (rising, falling,
  oscillating) on top of a VITS voice that has no pitch control of its own.
* :func:`breath` — a synthetic human breath (inhale or soft exhale): shaped
  noise with a vocal-tract-like spectral envelope and an asymmetric envelope,
  placed relative to the speech level.
"""

from __future__ import annotations

import numpy as np

__all__ = ["track_f0", "psola", "breath", "rms_db", "apply_gain_db", "silence", "concat"]


def rms_db(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0
    return 20.0 * np.log10(r) if r > 0 else -120.0


def apply_gain_db(x: np.ndarray, db: float) -> np.ndarray:
    return np.clip(x * (10.0 ** (db / 20.0)), -1.0, 1.0).astype(np.float32)


def silence(sr: int, ms: float) -> np.ndarray:
    return np.zeros(int(sr * ms / 1000.0), np.float32)


def concat(chunks: list[np.ndarray], sr: int, fade_ms: float = 8.0) -> np.ndarray:
    """Concatenate with short raised-cosine fades at every edge (no clicks)."""
    n = int(sr * fade_ms / 1000.0)
    out = []
    for c in chunks:
        c = np.asarray(c, np.float32).copy()
        if n and len(c) > 2 * n and np.any(c):
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n, dtype=np.float32))
            c[:n] *= ramp
            c[-n:] *= ramp[::-1]
        out.append(c)
    return np.concatenate(out) if out else np.zeros(0, np.float32)


# --------------------------------------------------------------------------- #
# Pitch tracking
# --------------------------------------------------------------------------- #
def track_f0(x: np.ndarray, sr: int, hop_ms: float = 5.0, fmin: float = 65.0, fmax: float = 400.0,
             threshold: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(f0_hz, voiced)`` per hop; ``f0`` is 0 where unvoiced.

    YIN: the difference function ``d(τ) = e(0) + e(τ) − 2·r(τ)`` is computed for
    all frames at once (batched FFT cross-correlation + cumulative energies),
    then cumulative-mean normalized; the first dip below ``threshold`` is
    refined parabolically. A frame is voiced when such a dip exists and the
    frame carries energy.
    """
    x = np.asarray(x, np.float64)
    hop = max(1, int(sr * hop_ms / 1000.0))
    tmax = int(sr / fmin)
    tmin = max(2, int(sr / fmax))
    win = 2 * tmax
    span = win + tmax
    n_frames = max(0, (len(x) - span) // hop + 1)
    f0 = np.zeros(n_frames)
    voiced = np.zeros(n_frames, bool)
    if n_frames == 0:
        return f0, voiced
    idx = np.arange(span)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx]  # (F, span)
    a = frames[:, :win]
    gate = 10 ** (-45 / 20.0) * max(1e-9, float(np.max(np.abs(x))))
    loud = np.sqrt(np.mean(a * a, axis=1)) >= gate
    nfft = 1 << int(np.ceil(np.log2(span + win)))
    r = np.fft.irfft(np.fft.rfft(frames, nfft) * np.conj(np.fft.rfft(a, nfft)), nfft)[:, : tmax + 1]
    sq = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(frames * frames, axis=1)], axis=1)
    taus = np.arange(tmax + 1)
    e_tau = sq[:, taus + win] - sq[:, taus]  # energy of the window shifted by τ
    d = e_tau[:, :1] + e_tau - 2.0 * r
    d[:, 0] = 0.0
    cs = np.cumsum(d[:, 1:], axis=1)
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(cs, 1e-12)
    below = cmnd[:, tmin:] < threshold
    has = below.any(axis=1) & loud
    first = tmin + np.argmax(below, axis=1)
    for i in np.nonzero(has)[0]:
        tau = int(first[i])
        c = cmnd[i]
        while tau + 1 <= tmax and c[tau + 1] < c[tau]:
            tau += 1
        if 1 <= tau < tmax:
            y0, y1, y2 = c[tau - 1], c[tau], c[tau + 1]
            den = y0 - 2 * y1 + y2
            shift = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        else:
            shift = 0.0
        f0[i] = sr / (tau + shift)
        voiced[i] = True
    # remove isolated voiced/unvoiced flips
    if n_frames >= 3:
        v = voiced.copy()
        for i in range(1, n_frames - 1):
            if voiced[i - 1] == voiced[i + 1] != voiced[i]:
                v[i] = voiced[i - 1]
                if v[i] and f0[i] == 0:
                    f0[i] = 0.5 * (f0[i - 1] + f0[i + 1])
        voiced = v
        f0[~voiced] = 0.0
    return f0, voiced


# --------------------------------------------------------------------------- #
# TD-PSOLA
# --------------------------------------------------------------------------- #
def _pitch_marks(x: np.ndarray, sr: int, f0: np.ndarray, voiced: np.ndarray, hop: int,
                 unvoiced_period: float) -> tuple[np.ndarray, np.ndarray]:
    """Analysis marks (sample positions) and the local period at each mark."""
    marks, periods = [], []
    t = 0.0
    n = len(x)
    while t < n:
        k = min(len(f0) - 1, int(t / hop)) if len(f0) else 0
        if len(f0) and voiced[k] and f0[k] > 0:
            period = sr / f0[k]
            # snap to the local waveform peak so grains are pitch-synchronous
            lo, hi = int(t - 0.3 * period), int(t + 0.3 * period)
            lo, hi = max(0, lo), min(n, hi)
            if hi > lo:
                t = lo + float(np.argmax(np.abs(x[lo:hi])))
        else:
            period = unvoiced_period
        marks.append(t)
        periods.append(period)
        t += period
    return np.asarray(marks), np.asarray(periods)


def psola(x: np.ndarray, sr: int, ratio: np.ndarray | float, hop_ms: float = 5.0,
          f0: np.ndarray | None = None, voiced: np.ndarray | None = None) -> np.ndarray:
    """Pitch-shift ``x`` by ``ratio`` (scalar or per-sample array) with TD-PSOLA.

    ``ratio > 1`` raises pitch. Duration is preserved; formants are preserved
    because each grain is a two-period, Hann-windowed copy of the original.
    Unvoiced stretches are copied with a fixed 5 ms grain grid (no shift).
    """
    x = np.asarray(x, np.float32)
    n = len(x)
    if n == 0:
        return x.copy()
    r = np.full(n, float(ratio)) if np.isscalar(ratio) else np.asarray(ratio, np.float64)
    if np.allclose(r, 1.0):
        return x.copy()
    hop = max(1, int(sr * hop_ms / 1000.0))
    if f0 is None or voiced is None:
        f0, voiced = track_f0(x, sr, hop_ms)
    unv = sr * 0.005
    marks, periods = _pitch_marks(x, sr, f0, voiced, hop, unv)
    out = np.zeros(n + 2 * int(sr / 50), np.float64)
    norm = np.zeros_like(out)
    t_syn = marks[0] if len(marks) else 0.0
    while t_syn < n:
        # nearest analysis mark to the synthesis instant (duration preserved)
        j = int(np.clip(np.searchsorted(marks, t_syn), 0, len(marks) - 1))
        if j > 0 and abs(marks[j - 1] - t_syn) < abs(marks[j] - t_syn):
            j -= 1
        period = periods[j]
        is_voiced = period != unv
        half = int(round(period))
        c = int(round(marks[j]))
        lo, hi = c - half, c + half
        seg = np.zeros(2 * half, np.float64)
        s_lo, s_hi = max(0, lo), min(n, hi)
        if s_hi > s_lo:
            seg[s_lo - lo : s_hi - lo] = x[s_lo:s_hi]
        w = np.hanning(2 * half + 2)[1:-1] if half > 1 else np.ones(2 * half)
        pos = int(round(t_syn)) - half + int(sr / 50)
        if pos >= 0 and pos + 2 * half <= len(out):
            out[pos : pos + 2 * half] += seg * w
            norm[pos : pos + 2 * half] += w
        k = min(n - 1, int(t_syn))
        step = period / r[k] if is_voiced else period
        t_syn += max(1.0, step)
    off = int(sr / 50)
    y = out[off : off + n] / np.maximum(norm[off : off + n], 1e-3)
    # where coverage was thin (edges), fall back to the original signal
    thin = norm[off : off + n] < 0.3
    y[thin] = x[thin]
    return np.clip(y, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Breath
# --------------------------------------------------------------------------- #
def breath(sr: int, ms: float = 320.0, level_db: float = -34.0, kind: str = "inhale",
           seed: int | None = 0) -> np.ndarray:
    """A synthetic breath at ``level_db`` dBFS RMS.

    The noise is shaped by a smooth vocal-tract-like envelope (broad resonances
    near 600 Hz, 1.5 kHz and 2.8 kHz, gentle roll-off above 4 kHz) because a
    breath is turbulence filtered by an open, neutral vocal tract. An inhale
    swells slowly and stops quickly; an exhale (sigh) starts strongly and fades.
    """
    rng = np.random.default_rng(seed)
    n = max(1, int(sr * ms / 1000.0))
    noise = rng.standard_normal(n)
    spec = np.fft.rfft(noise)
    f = np.fft.rfftfreq(n, 1.0 / sr)

    def peak(fc: float, bw: float, g: float) -> np.ndarray:
        return g * np.exp(-0.5 * ((f - fc) / bw) ** 2)

    env = 0.15 + peak(600, 250, 0.6) + peak(1500, 450, 1.0) + peak(2800, 600, 0.7)
    env *= 1.0 / (1.0 + (f / 4500.0) ** 4)  # roll-off
    env *= np.clip(f / 250.0, 0, 1)  # no rumble
    y = np.fft.irfft(spec * env, n)
    t = np.linspace(0.0, 1.0, n)
    if kind == "exhale":
        shape = np.sin(np.pi * np.clip(t, 0, 1) ** 0.6) * (1.0 - 0.4 * t)
    else:  # inhale: slow swell, quick stop
        shape = np.sin(0.5 * np.pi * np.clip(t / 0.75, 0, 1)) ** 1.5 * np.clip((1.0 - t) / 0.25, 0, 1) ** 0.8
    y = y * shape
    cur = rms_db(y)
    y = y * 10 ** ((level_db - cur) / 20.0) if cur > -120 else y
    return np.clip(y, -1.0, 1.0).astype(np.float32)
