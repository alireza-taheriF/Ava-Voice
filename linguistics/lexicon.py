"""Pronunciation lexicon: word → Homo-GE2PE-notation pronunciation.

Three layers; a human-curated entry always wins over a learned one
(user > builtin > learned):

1. **builtin** — ``linguistics/data/colloquial_lexicon.tsv``: colloquial
   (Tehrani) forms and common homographs the G2P model gets wrong
   (``بهت`` → *behet*, ``میشه`` → *mi$e*, ``بریم`` → *berim* …);
2. **user** — ``$AVA_LEXICON`` or ``~/.config/ava-voice/lexicon.tsv``: your
   own corrections (names, slang, brand words);
3. **learned** — ``~/.config/ava-voice/learned_lexicon.tsv``: written by
   :mod:`ava_voice.tts.self_check`, which listens to Ava's own speech and keeps
   the pronunciation variant a recognizer understands best.

Format: ``word<TAB>pronunciation`` per line, ``#`` comments. A pronunciation
may contain spaces (one written word spoken as two).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

__all__ = ["Lexicon", "default_lexicon", "user_lexicon_path", "learned_lexicon_path"]

_BUILTIN = Path(__file__).parent / "data" / "colloquial_lexicon.tsv"


def _config_dir() -> Path:
    return Path(os.environ.get("AVA_CONFIG_HOME", Path.home() / ".config" / "ava-voice")).expanduser()


def user_lexicon_path() -> Path:
    env = os.environ.get("AVA_LEXICON")
    return Path(env).expanduser() if env else _config_dir() / "lexicon.tsv"


def learned_lexicon_path() -> Path:
    return _config_dir() / "learned_lexicon.tsv"


def _key(word: str) -> str:
    # unify Arabic/Persian letters and drop diacritics so lookups are robust
    w = word.strip().replace("ي", "ی").replace("ك", "ک").replace("ـ", "")
    return "".join(ch for ch in w if not ("ً" <= ch <= "ٟ" or ch == "ٰ"))


def _read(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "\t" not in line:
            continue
        word, pron = line.split("\t", 1)
        if word.strip() and pron.strip():
            out[_key(word)] = pron.strip()
    return out


class Lexicon:
    def __init__(self, builtin: Path | None = _BUILTIN, user: Path | None = None,
                 learned: Path | None = None) -> None:
        self.user_path = user if user is not None else user_lexicon_path()
        self.learned_path = learned if learned is not None else learned_lexicon_path()
        self._lock = threading.Lock()
        self._builtin = _read(builtin) if builtin else {}
        self.reload()

    def reload(self) -> None:
        with self._lock:
            self._user = _read(self.user_path)
            self._learned = _read(self.learned_path)

    def lookup(self, word: str) -> str | None:
        k = _key(word)
        with self._lock:
            return self._user.get(k) or self._builtin.get(k) or self._learned.get(k)

    def is_curated(self, word: str) -> bool:
        """True if a human wrote this entry (builtin or user) — never re-learned."""
        k = _key(word)
        with self._lock:
            return k in self._user or k in self._builtin

    def __contains__(self, word: str) -> bool:
        return self.lookup(word) is not None

    def __len__(self) -> int:
        return len(set(self._builtin) | set(self._user) | set(self._learned))

    def learn(self, word: str, pron: str) -> None:
        """Persist a learned pronunciation (used only where no curated entry exists)."""
        k = _key(word)
        with self._lock:
            self._learned[k] = pron
            self.learned_path.parent.mkdir(parents=True, exist_ok=True)
            lines = ["# learned by ava_voice.tts.self_check — edit or delete freely"]
            lines += [f"{w}\t{p}" for w, p in sorted(self._learned.items())]
            self.learned_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def forget(self, word: str) -> None:
        k = _key(word)
        with self._lock:
            if self._learned.pop(k, None) is not None:
                lines = ["# learned by ava_voice.tts.self_check — edit or delete freely"]
                lines += [f"{w}\t{p}" for w, p in sorted(self._learned.items())]
                self.learned_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_DEFAULT: Lexicon | None = None


def default_lexicon() -> Lexicon:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Lexicon()
    return _DEFAULT
