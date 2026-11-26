"""Install a trained voice into Ava and smoke-test it.

    python -m ava_voice.clone.install --onnx ali.onnx --config config.json --name ali

Copies the model and its config into ``~/.cache/ava-voice/voice-<name>/``,
checks that the phoneme map matches Ava's phoneme inventory, and speaks a test
sentence to ``voice-<name>/test.wav``. Point Ava/PVAP at the folder
(``AVA_TTS_VOICE_DIR`` or ``[voice] voice_dir``).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

__all__ = ["install_voice"]

TEST_SENTENCE = "سلام! این صدای جدید منه. امروز هوا خیلی خوبه، نه؟"


def install_voice(onnx: Path, config: Path, name: str, home: Path | None = None) -> Path:
    from ava_voice.tts import models
    from ava_voice.tts.vits import GE2PE_TO_IPA

    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    idmap = cfg.get("phoneme_id_map", {})
    needed = {ch for ipa in GE2PE_TO_IPA.values() for ch in ipa} | {"ˈ", " ", ",", ".", "?", "!", "_", "^", "$"}
    missing = sorted(ch for ch in needed if ch not in idmap)
    if missing:
        raise ValueError(f"voice phoneme map lacks {missing} — train with Piper's default (espeak IPA) map")
    dest = (home or models.model_home()) / f"voice-{name}"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx, dest / f"{name}.onnx")
    (dest / f"{name}.onnx.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Install a trained Piper voice into Ava")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--config", required=True, help="the config.json written by piper.train (--data.config_path)")
    ap.add_argument("--name", required=True)
    args = ap.parse_args(argv)
    dest = install_voice(Path(args.onnx), Path(args.config), args.name)
    from ava_voice.clone.audio import write_wav
    from ava_voice.tts.vits import PersianVITS

    v = PersianVITS(dest)
    write_wav(dest / "test.wav", v.synthesize(TEST_SENTENCE), v.sample_rate)
    print(f"installed → {dest}\nlisten: {dest / 'test.wav'}\nuse it: AVA_TTS_VOICE_DIR={dest}  (PVAP: [voice] voice_dir)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
