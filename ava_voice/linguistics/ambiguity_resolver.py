"""Ambiguity resolution → canonical spoken forms (Phase 1A).

Persian writing is systematically ambiguous in two ways that matter for speech:

1. **Colloquial vs. formal orthography** — informal spellings (``میرم``,
   ``نمیدونم``) must be mapped to their canonical *spoken* forms (``می‌روم``,
   ``نمی‌دانم``) before phonemization.
2. **Homographs** — identical spellings with distinct pronunciations
   (``کرم`` → *kerm* / *karam* / *kerem*). These cannot be resolved from
   spelling alone, so this stage records a **default reading plus explicit
   alternatives**, leaving final selection to a context model (out of scope for
   Phase 1A) while remaining fully deterministic.

This module is the *only* place colloquial→formal conversion happens, keeping
:mod:`persian_normalizer` strictly orthographic (single responsibility).

Example
-------
>>> resolver = AmbiguityResolver()
>>> resolver.resolve_token("میرم").canonical
'می‌روم'
>>> r = resolver.resolve_token("کرم")
>>> r.is_ambiguous, r.alternatives
(True, ('کرم', 'کرم'))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from ava_voice.core.logger import get_logger

_logger = get_logger("linguistics.ambiguity")

ZWNJ: Final[str] = "\u200c"

# Colloquial (spoken/informal) → canonical formal spoken form. Keys are matched
# against the ZWNJ-insensitive surface (so "می‌رم" and "میرم" both match).
_COLLOQUIAL_TO_CANONICAL: Final[dict[str, str]] = {
    "میرم": "می‌روم",
    "میری": "می‌روی",
    "میره": "می‌رود",
    "میگم": "می‌گویم",
    "میگی": "می‌گویی",
    "میگه": "می‌گوید",
    "نمیدونم": "نمی‌دانم",
    "نمیدونی": "نمی‌دانی",
    "میخوام": "می‌خواهم",
    "میخوای": "می‌خواهی",
    "میتونم": "می‌توانم",
    "میشه": "می‌شود",
    "بریم": "برویم",
    "بیا": "بیا",
    "چیه": "چیست",
    "کجایی": "کجا هستی",
}

# Homographs: canonical *default* reading is the surface itself (spelling is
# unchanged); the value documents the legal pronunciations for a context model.
# Alternatives are stored as human-readable romanizations for clarity.
_HOMOGRAPHS: Final[dict[str, tuple[str, ...]]] = {
    "کرم": ("kerm", "karam", "kerem"),   # worm / generosity / cream
    "مرد": ("mard", "mord"),             # man / (he) died
    "شکر": ("shekar", "shokr"),          # sugar / thanks
    "سیر": ("sir", "seyr"),              # garlic·full / stroll
    "گل": ("gol", "gel"),                # flower / mud
    "کشت": ("kesht", "kosht"),           # cultivation / (he) killed
}


def _strip_zwnj(token: str) -> str:
    """Return ``token`` with all ZWNJ removed (for lookup keys)."""
    return token.replace(ZWNJ, "")


@dataclass(frozen=True)
class Resolution:
    """Result of resolving a single token to its canonical spoken form.

    Attributes
    ----------
    surface:
        The original input token.
    canonical:
        The chosen canonical spoken form (default reading for homographs).
    is_ambiguous:
        ``True`` when multiple legal readings exist and could not be resolved
        from spelling alone.
    alternatives:
        All legal readings (romanized for homographs, orthographic otherwise).
    resolution_kind:
        One of ``"identity"``, ``"colloquial"`` or ``"homograph"`` — documents
        which rule fired (useful for auditing / patent claims).
    """

    surface: str
    canonical: str
    is_ambiguous: bool = False
    alternatives: tuple[str, ...] = field(default_factory=tuple)
    resolution_kind: str = "identity"


class AmbiguityResolver:
    """Deterministic resolver from written forms to canonical spoken forms."""

    def __init__(
        self,
        colloquial_map: dict[str, str] | None = None,
        homographs: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self._colloquial: dict[str, str] = dict(_COLLOQUIAL_TO_CANONICAL)
        if colloquial_map:
            self._colloquial.update(colloquial_map)
        self._homographs: dict[str, tuple[str, ...]] = dict(_HOMOGRAPHS)
        if homographs:
            self._homographs.update(homographs)

    # -- Public API --------------------------------------------------------- #
    def resolve_token(self, token: str) -> Resolution:
        """Resolve a single ``token`` to a :class:`Resolution`.

        Precedence: colloquial mapping first (it rewrites the surface), then
        homograph detection, else identity.
        """
        clean = token.strip()
        if not clean:
            return Resolution(surface=token, canonical=token)

        key = _strip_zwnj(clean)

        canonical = self._colloquial.get(key)
        if canonical is not None:
            return Resolution(
                surface=token,
                canonical=canonical,
                resolution_kind="colloquial",
            )

        readings = self._homographs.get(key)
        if readings is not None:
            return Resolution(
                surface=token,
                canonical=clean,  # spelling unchanged; reading deferred
                is_ambiguous=len(readings) > 1,
                alternatives=readings,
                resolution_kind="homograph",
            )

        return Resolution(surface=token, canonical=clean)

    def resolve_text(self, text: str) -> str:
        """Return ``text`` with every token replaced by its canonical form.

        Whitespace tokenization is used; punctuation attached to tokens is left
        untouched by resolution (the token graph handles punctuation
        separately). Deterministic default readings are applied for homographs.
        """
        resolved = [self.resolve_token(tok).canonical for tok in text.split()]
        return " ".join(resolved)


_default_resolver: Final[AmbiguityResolver] = AmbiguityResolver()


def resolve_token(token: str) -> Resolution:
    """Resolve a token using the module-level default resolver."""
    return _default_resolver.resolve_token(token)


def resolve_text(text: str) -> str:
    """Resolve text using the module-level default resolver."""
    return _default_resolver.resolve_text(text)
