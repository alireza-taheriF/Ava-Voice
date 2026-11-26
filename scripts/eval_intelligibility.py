"""Measure Persian TTS intelligibility: synthesize → Whisper → character error rate.

    python scripts/eval_intelligibility.py --whisper-dir models/sherpa-onnx-whisper-turbo \
        [--voice-dir DIR] [--expressive] [--verbose]

CER is computed on Persian letters only (punctuation, spaces, ZWNJ and
diacritics removed; digits verbalized on both sides), so it reflects what a
listener — here Whisper — actually recovered. It includes Whisper's own
errors, so treat it as an upper bound on the TTS error.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np


DATA = Path(__file__).parent / "data" / "tts_eval_fa.txt"


def _norm(t: str) -> str:
    from ava_voice.linguistics.number_verbalizer import verbalize_numbers

    t = verbalize_numbers(unicodedata.normalize("NFC", t))
    for a, b in (("ي", "ی"), ("ك", "ک"), ("ۀ", "ه"), ("أ", "ا"), ("إ", "ا"), ("ئ", "ی"), ("ؤ", "و"), ("‌", "")):
        t = t.replace(a, b)
    t = re.sub(r"[ً-ٰٟ]", "", t)
    return re.sub(r"[^ء-ۿ]", "", t)


def cer(ref: str, hyp: str) -> float:
    r, h = _norm(ref), _norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


def main(argv: list[str] | None = None) -> int:
    import wave

    from ava_voice.asr.whisper_onnx import WhisperOnnxASR
    from ava_voice.tts.expressive import ExpressiveVoice
    from ava_voice.tts.vits import PersianVITS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--whisper-dir", required=True)
    ap.add_argument("--voice-dir")
    ap.add_argument("--g2p-dir")
    ap.add_argument("--sentences", default=str(DATA))
    ap.add_argument("--save", help="directory for the synthesized WAVs")
    ap.add_argument("--expressive", action="store_true", help="render with emotion, PSOLA and breaths")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--pitch", type=float, default=0.0, help="global pitch offset, semitones")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    tts = PersianVITS(args.voice_dir, args.g2p_dir)
    render = (ExpressiveVoice(tts, seed=0, speed=args.speed, pitch_semitones=args.pitch).render
              if args.expressive else (lambda t: tts.synthesize(t, speed=args.speed)))
    asr = WhisperOnnxASR(args.whisper_dir)
    lines = [l.strip() for l in open(args.sentences, encoding="utf-8") if l.strip()]
    scores, rtf = [], []
    for k, text in enumerate(lines):
        t0 = time.monotonic()
        wav = render(text)
        rtf.append((time.monotonic() - t0) / (len(wav) / tts.sample_rate))
        if args.save:
            Path(args.save).mkdir(parents=True, exist_ok=True)
            with wave.open(f"{args.save}/{k:02d}.wav", "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(tts.sample_rate)
                w.writeframes((np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes())
        hyp = asr.transcribe(wav, tts.sample_rate)
        scores.append(cer(text, hyp))
        if args.verbose:
            print(f"{scores[-1]:.3f}  {text}\n       {hyp}\n       {tts.phonemize(text)}", flush=True)
    print(f"voice={tts.name} sentences={len(lines)} mean_CER={np.mean(scores):.4f} "
          f"median_CER={np.median(scores):.4f} TTS_RTF={np.mean(rtf):.3f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
