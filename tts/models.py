"""Model registry and downloader for Ava's neural Persian TTS.

Two artefacts are needed:

* **G2P** — Homo-GE2PE (MIT), run in pure NumPy by
  :mod:`ava_voice.linguistics.ge2pe_numpy`.
* **Voice** — a Persian VITS voice exported to ONNX (Piper format), run with
  ``onnxruntime``. The voices below are mirrored as tarballs on the
  sherpa-onnx GitHub release page.

``python -m ava_voice.tts.models fetch [--voice gyro]`` downloads both into
``~/.cache/ava-voice`` (or ``$AVA_MODEL_HOME``). Paths can also be set
explicitly with ``AVA_G2P_MODEL_PATH`` and ``AVA_TTS_VOICE_DIR``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

__all__ = ["VOICES", "DEFAULT_VOICE", "model_home", "g2p_dir", "voice_dir", "fetch_g2p", "fetch_voice"]

_SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
_GE2PE_REPO = "https://github.com/MahtaFetrat/Homo-GE2PE"


@dataclass(frozen=True)
class Voice:
    key: str
    archive: str  # sherpa-onnx tarball stem
    onnx: str  # model file inside the tarball
    gender: str
    note: str

    @property
    def url(self) -> str:
        return f"{_SHERPA}/{self.archive}.tar.bz2"


VOICES: dict[str, Voice] = {
    v.key: v
    for v in (
        Voice("gyro", "vits-piper-fa_IR-gyro-medium", "fa_IR-gyro-medium.onnx", "male", "CER 12%"),
        Voice("amir", "vits-piper-fa_IR-amir-medium", "fa_IR-amir-medium.onnx", "male", "CER 24%"),
        Voice("reza_ibrahim", "vits-piper-fa_IR-reza_ibrahim-medium", "fa_IR-reza_ibrahim-medium.onnx", "male", "CER 20%"),
        Voice("ganji", "vits-piper-fa_IR-ganji-medium", "fa_IR-ganji-medium.onnx", "male", "default — CER 5-9%"),
        Voice("ganji_adabi", "vits-piper-fa_IR-ganji_adabi-medium", "fa_IR-ganji_adabi-medium.onnx", "male", "CER 11%, literary style"),
        Voice("haaniye", "vits-mimic3-fa-haaniye_low", "fa-haaniye_low.onnx", "female", "CER 11%, low-quality model"),
    )
}
DEFAULT_VOICE = "ganji"
# CER = Whisper-turbo round-trip character error on 20 phone-agent sentences
# with Ava's frontend at Piper default noise (see PVAP tools/eval_tts.py).


def model_home() -> Path:
    return Path(os.environ.get("AVA_MODEL_HOME", Path.home() / ".cache" / "ava-voice")).expanduser()


def g2p_dir() -> Path:
    env = os.environ.get("AVA_G2P_MODEL_PATH")
    return Path(env).expanduser() if env else model_home() / "homo-ge2pe"


def voice_dir(key: str | None = None) -> Path:
    env = os.environ.get("AVA_TTS_VOICE_DIR")
    if env and key is None:
        return Path(env).expanduser()
    return model_home() / VOICES[key or DEFAULT_VOICE].archive


def _download(url: str, dest: Path) -> None:
    print(f"downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as fh:
        shutil.copyfileobj(r, fh)


def fetch_g2p(dest: Path | None = None) -> Path:
    dest = dest or g2p_dir()
    if (dest / "model.safetensors").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "homo-ge2pe.zip"
        if shutil.which("git"):
            subprocess.run(["git", "clone", "-q", "--depth", "1", _GE2PE_REPO, f"{tmp}/repo"], check=True)
            shutil.copy(Path(tmp) / "repo" / "model-weights" / "homo-ge2pe.zip", zpath)
        else:
            _download(f"{_GE2PE_REPO}/raw/main/model-weights/homo-ge2pe.zip", zpath)
        with zipfile.ZipFile(zpath) as z:
            for name in ("config.json", "model.safetensors", "generation_config.json"):
                (dest / name).write_bytes(z.read(name))
    return dest


def fetch_voice(key: str = DEFAULT_VOICE, home: Path | None = None) -> Path:
    v = VOICES[key]
    home = home or model_home()
    out = home / v.archive
    if (out / v.onnx).exists():
        return out
    home.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tpath = Path(tmp) / "voice.tar.bz2"
        _download(v.url, tpath)
        with tarfile.open(tpath) as t:
            members = [m for m in t.getmembers() if not m.name.startswith("/") and ".." not in m.name]
            t.extractall(home, members=members)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download Ava Voice neural TTS models")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--voice", default=DEFAULT_VOICE, choices=sorted(VOICES))
    sub.add_parser("list")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        for v in VOICES.values():
            print(f"{v.key:14s} {v.gender:6s} {v.note}")
        return 0
    print(fetch_g2p())
    print(fetch_voice(args.voice))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
