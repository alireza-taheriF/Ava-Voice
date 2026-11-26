"""Turn your recordings into a Piper training set phonemized exactly like Ava speaks.

    python -m ava_voice.clone.dataset --voice ali --whisper-dir ~/models/sherpa-onnx-whisper-turbo

For every take: quality check (level, clipping, noise), silence trim, loudness
normalization, and optionally a Whisper check that you read the right
sentence (character error under ``--max-cer``). Accepted takes become
``train/wavs/*.wav`` plus:

* ``train/metadata.csv`` — ``file.wav|IPA``, Ava's own phonemes, for Piper's
  ``--data.phoneme_type text``. Training on the same phonemes Ava feeds the
  voice at inference keeps pronunciation consistent, including the G2P fixes
  and the learned lexicon.
* ``train/metadata_text.csv`` — ``file.wav|text``, if you prefer espeak-ng.
* ``train/report.json`` — what was kept, what was rejected and why.

The builder refuses to run without the recorded consent (``consent.json``).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ava_voice.clone.audio import TARGET_RATE, analyze, read_wav, resample, trim_and_normalize, write_wav
from ava_voice.clone.record import recordings_root

__all__ = ["build_dataset"]

MIN_GOOD_MINUTES = 10.0


def _cer(ref: str, hyp: str) -> float:
    from ava_voice.tts.self_check import _cer as cer
    from ava_voice.tts.self_check import normalize_for_compare as n

    return cer(n(ref), n(hyp))


def build_dataset(rec_dir: Path, phonemize, asr=None, max_cer: float = 0.35, out: Path | None = None) -> dict:
    rec_dir = Path(rec_dir)
    consent = rec_dir / "consent.json"
    if not consent.exists() or not (rec_dir / "consent.wav").exists():
        raise PermissionError(f"no recorded consent in {rec_dir} — record it with ava_voice.clone.record first")
    script = json.loads((rec_dir / "script.json").read_text(encoding="utf-8"))
    out = Path(out or rec_dir / "train")
    if out.exists():
        shutil.rmtree(out)
    (out / "wavs").mkdir(parents=True)
    kept, rejected, total_s = [], [], 0.0
    ipa_rows, text_rows = [], []
    for wav in sorted((rec_dir / "wavs").glob("utt_*.wav")):
        idx = int(wav.stem.split("_")[1])
        text = script[idx] if idx < len(script) else ""
        x, sr = read_wav(wav)
        x = resample(x, sr, TARGET_RATE)
        qc = analyze(x, TARGET_RATE)
        if not qc.ok:
            rejected.append({"file": wav.name, "text": text, "why": qc.problems})
            continue
        y = trim_and_normalize(x, TARGET_RATE)
        if asr is not None:
            heard = asr.transcribe(y, TARGET_RATE)
            err = _cer(text, heard)
            if err > max_cer:
                rejected.append({"file": wav.name, "text": text, "why": [f"متن خوانده‌شده با جمله نمی‌خواند (CER {err:.2f}): {heard}"]})
                continue
        name = f"{wav.stem}.wav"
        write_wav(out / "wavs" / name, y, TARGET_RATE)
        ipa = phonemize(text)
        ipa_rows.append(f"{name}|{ipa}")
        text_rows.append(f"{name}|{text}")
        kept.append(name)
        total_s += len(y) / TARGET_RATE
    (out / "metadata.csv").write_text("\n".join(ipa_rows) + "\n", encoding="utf-8")
    (out / "metadata_text.csv").write_text("\n".join(text_rows) + "\n", encoding="utf-8")
    shutil.copy(consent, out / "consent.json")
    report = {
        "kept": len(kept), "rejected": rejected, "minutes": round(total_s / 60, 2),
        "enough": total_s / 60 >= MIN_GOOD_MINUTES, "sample_rate": TARGET_RATE, "dir": str(out),
        "consent": json.loads(consent.read_text(encoding="utf-8")),
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a Piper training set from Ava voice recordings")
    ap.add_argument("--voice", required=True)
    ap.add_argument("--recordings", help=f"default: {recordings_root()}/<voice>")
    ap.add_argument("--whisper-dir", help="check each take was read correctly (recommended)")
    ap.add_argument("--max-cer", type=float, default=0.35)
    ap.add_argument("--g2p-dir")
    ap.add_argument("--voice-dir", help="any Ava voice (only its phonemizer is used)")
    args = ap.parse_args(argv)
    from ava_voice.tts.vits import PersianVITS

    voice = PersianVITS(args.voice_dir, args.g2p_dir)
    asr = None
    if args.whisper_dir:
        from ava_voice.asr.whisper_onnx import WhisperOnnxASR

        asr = WhisperOnnxASR(args.whisper_dir)
    rec = Path(args.recordings) if args.recordings else recordings_root() / args.voice
    rep = build_dataset(rec, voice.phonemize, asr, args.max_cer)
    print(f"kept {rep['kept']} takes, {rep['minutes']} minutes → {rep['dir']}")
    for r in rep["rejected"]:
        print(f"  rejected {r['file']}: {'; '.join(r['why'])}")
    if not rep["enough"]:
        print(f"note: {rep['minutes']} min is below the ~{MIN_GOOD_MINUTES:.0f} min that gives a convincing clone; record more")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
