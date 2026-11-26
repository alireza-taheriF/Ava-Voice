"""Record your voice for cloning, in the browser, on your own computer.

    python -m ava_voice.clone.record --voice ali          # → http://localhost:8765

The page walks you through:

1. **Consent.** You type your name and read a consent sentence aloud. It is
   saved with the recordings (``consent.json`` + ``consent.wav``), and the
   dataset builder refuses to build a voice without it. Only clone a voice
   with its owner's recorded consent, and only for that person's own use.
2. **Recording.** About 140 sentences are shown one by one (≈ 12–15 minutes).
   Every take is checked at once for clipping, level and background noise; you
   can listen back and re-record.

Recordings go to ``~/ava-voice-recordings/<voice>/`` (22.05 kHz mono WAV).
Then run :mod:`ava_voice.clone.dataset`. The server listens on ``localhost``
only, because browsers allow the microphone there without HTTPS.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from ava_voice.clone.audio import TARGET_RATE, analyze, read_wav, resample, write_wav

__all__ = ["load_script", "Recorder", "make_server", "CONSENT_TEXT", "recordings_root"]

SCRIPT = Path(__file__).parent / "data" / "script_fa.txt"
PAGE = Path(__file__).parent / "data" / "record.html"
CONSENT_TEXT = ("من، {name}، با رضایت کامل صدای خودم را ضبط می‌کنم تا یک صدای مصنوعی شخصی "
                "برای استفاده‌ی خودم ساخته شود.")


def recordings_root() -> Path:
    return Path.home() / "ava-voice-recordings"


def load_script(path: Path = SCRIPT) -> list[str]:
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


class Recorder:
    """Storage + QC behind the recording page (usable without HTTP in tests)."""

    def __init__(self, voice: str, root: Path | None = None, script: list[str] | None = None) -> None:
        if not re.fullmatch(r"[\w-]{1,40}", voice):
            raise ValueError("voice name: letters, digits, - or _ only")
        self.dir = (root or recordings_root()) / voice
        self.wavs = self.dir / "wavs"
        self.wavs.mkdir(parents=True, exist_ok=True)
        self.script = script or load_script()
        self._lock = threading.Lock()
        self.dir.joinpath("script.json").write_text(json.dumps(self.script, ensure_ascii=False, indent=1), encoding="utf-8")

    def state(self) -> dict:
        done = sorted(int(p.stem.split("_")[1]) for p in self.wavs.glob("utt_*.wav"))
        consent = self.dir / "consent.json"
        return {"sentences": self.script, "done": done,
                "consent": json.loads(consent.read_text(encoding="utf-8")) if consent.exists() else None,
                "consent_template": CONSENT_TEXT}

    def _decode(self, body: bytes, rate: int) -> np.ndarray:
        x = np.frombuffer(body, "<i2").astype(np.float32) / 32768.0
        return resample(x, rate, TARGET_RATE)

    def save_consent(self, name: str, body: bytes, rate: int) -> dict:
        name = name.strip()
        if not name or len(name) > 80:
            raise ValueError("name required")
        x = self._decode(body, rate)
        qc = analyze(x, TARGET_RATE)
        if qc.duration_s < 3:
            raise ValueError("consent recording too short — read the whole sentence")
        with self._lock:
            write_wav(self.dir / "consent.wav", x, TARGET_RATE)
            info = {"name": name, "text": CONSENT_TEXT.format(name=name),
                    "recorded_at": _dt.datetime.now().isoformat(timespec="seconds")}
            (self.dir / "consent.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        return info

    def save_take(self, idx: int, body: bytes, rate: int) -> dict:
        if not 0 <= idx < len(self.script):
            raise ValueError("bad sentence index")
        if not (self.dir / "consent.json").exists():
            raise PermissionError("record the consent sentence first")
        x = self._decode(body, rate)
        qc = analyze(x, TARGET_RATE)
        with self._lock:
            write_wav(self.wavs / f"utt_{idx:04d}.wav", x, TARGET_RATE)
        return qc.to_dict()

    def take_path(self, idx: int) -> Path:
        return self.wavs / f"utt_{idx:04d}.wav"


def make_server(rec: Recorder, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode())

        def do_GET(self):  # noqa: N802
            u = urlsplit(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/state":
                self._json(200, rec.state())
            elif u.path == "/api/audio":
                p = rec.dir / "consent.wav" if q.get("id") == "consent" else rec.take_path(int(q.get("id", -1)))
                if p.exists():
                    self._send(200, p.read_bytes(), "audio/wav")
                else:
                    self._json(404, {"error": "not recorded"})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            u = urlsplit(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            n = int(self.headers.get("Content-Length", 0))
            if n > 50 * 1024 * 1024:
                return self._json(413, {"error": "too large"})
            body = self.rfile.read(n)
            try:
                rate = int(q.get("rate", 48000))
                if u.path == "/api/consent":
                    self._json(200, rec.save_consent(q.get("name", ""), body, rate))
                elif u.path == "/api/take":
                    self._json(200, rec.save_take(int(q.get("id", -1)), body, rate))
                else:
                    self._json(404, {"error": "not found"})
            except PermissionError as e:
                self._json(403, {"error": str(e)})
            except ValueError as e:
                self._json(400, {"error": str(e)})

    return ThreadingHTTPServer((host, port), H)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Record your voice for an Ava voice clone")
    ap.add_argument("--voice", required=True, help="name for this voice, e.g. ali")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    rec = Recorder(args.voice)
    srv = make_server(rec, port=args.port)
    print(f"recording into {rec.dir}\nopen http://localhost:{args.port} in Chrome or Safari (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
