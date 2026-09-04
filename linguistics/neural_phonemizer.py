"""Neural Persian grapheme-to-phoneme conversion via Homo-GE2PE.

The rule-based backend in :mod:`~ava_voice.linguistics.phonemizer` cannot recover
Persian's unwritten short vowels: it emits :data:`~ava_voice.linguistics.phonemizer.VOWEL_PLACEHOLDER`
instead, and roughly three in four common words carry at least one. This backend
replaces that guesswork with `Homo-GE2PE <https://huggingface.co/MahtaFetrat/Homo-GE2PE-Persian>`_,
a byte-level T5 fine-tuned on the HomoRich dataset (MIT licensed, reported 3.98%
PER and 76.9% homograph accuracy on SentenceBench).

It resolves three of the project's MVP blockers at once: short vowels, ezafe
(اضافه), and number verbalization.

Design notes, all established by probing the real model
-------------------------------------------------------
* **Sentence-level, not word-level.** Homographs are the model's headline
  feature and it needs context to use it: ``گل`` alone returns ``ole``
  (nonsense), while ``گل زیبا است`` returns ``gol`` and ``کفشم پر از گل شد``
  returns ``gel``. :meth:`NeuralPhonemizer.analyze_sentence` therefore issues one
  call per sentence; :meth:`analyze_word` exists only for interface parity and is
  explicitly the degraded path.

* **``use_rules=False``.** The upstream ``GE2PE.rules()`` helper strips the ``1``
  marker (``phoneme.replace('1', '')``) — and that marker *is* the ezafe signal.
  With rules applied, ``کتاب علی`` → ``ketabe @/li``, whose trailing ``e`` is
  indistinguishable from the lexical ``e`` in ``خانه`` → ``xane``. We keep the
  raw output and parse the marker ourselves. (``rules()`` also mis-indexes
  phonemes against graphemes when the input carries harakat, garbling
  ``کتابِ علی`` into ``ketaebe``.)

* **Persian-script gating.** On empty, punctuation-only or Latin input the model
  degenerates and emits up to its 512-token limit of gibberish: ``""`` →
  ``"mkk nik mikon fo mkon mk mikon fo mkon …"``, ``"Django"`` →
  ``"ngoude ngourde ngo/ngole …"``. Non-Persian tokens never reach it.

* **Alignment is verified, never assumed.** The model may merge or split words —
  ``می روم`` (2 words) returns one, ``۱۴۰۳`` (1 word) returns three
  (``hezar @o cehars/d @o se``). When the returned word count does not match the
  input, this backend falls back rather than mis-attributing phonemes.

Phoneme mapping
---------------
The model emits its own ASCII scheme; :data:`_MODEL_SYMBOL_MAP` translates it to
the project alphabet. Only six symbols actually differ, and one of them —
``/`` → :data:`~ava_voice.linguistics.phonemizer.SHORT_A` — is the whole point:
it is the short /æ/ the rule-based backend could not distinguish from long /ɑː/.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, Sequence

from ava_voice.core.logger import get_logger
from ava_voice.linguistics.phonemizer import (
    PersianPhonemizer,
    PhonemeToken,
    WordPhonemes,
)

_logger = get_logger("linguistics.neural_phonemizer")

#: Environment variable holding the path to the unpacked model directory.
MODEL_PATH_ENV: Final[str] = "AVA_G2P_MODEL_PATH"

#: Marker the model appends to a word carrying the ezafe linker.
EZAFE_MARKER: Final[str] = "1"

#: KASRA (U+0650). When a writer types it word-finally they are marking ezafe
#: explicitly — but ``GE2PE.generate`` strips harakat before inference
#: (``GE2PE.py:84``), so the model never sees it and may not infer it from
#: context. Probed: ``کتابِ علی`` in isolation yields ``ketabe1`` (detected), but
#: inside ``این کتابِ علی است`` it yields plain ``ketab`` (missed). We therefore
#: honour an explicit kasra ourselves rather than relying on the model.
KASRA: Final[str] = "ِ"

#: The vowel an explicit ezafe surfaces as.
_EZAFE_VOWEL: Final[str] = "e"

#: Model symbol → project symbol. Unlisted characters map to themselves.
_MODEL_SYMBOL_MAP: Final[dict[str, str]] = {
    "/": "æ",    # short /æ/ (fatḥe) — the vowel the rule backend cannot recover
    "@": "'",    # glottal stop (alef onset / ع / ء)
    "$": "sh",   # ش
    "c": "ch",   # چ
    ";": "zh",   # ژ
    "q": "gh",   # ق / غ
    "w": "v",    # rare variant spelling of و
}

#: Characters the model may emit that are neither phonemes nor the ezafe marker.
_IGNORED_MODEL_CHARS: Final[frozenset[str]] = frozenset({"-", "_", "‌"})

# Persian/Arabic letter range. A token must contain at least one of these to be
# worth sending to the model; anything else degenerates (see module docstring).
_PERSIAN_LETTER_RE: Final[re.Pattern[str]] = re.compile(r"[ؠ-يٮ-ۓ]")

# Persian and ASCII digits. Digit-only tokens are sent to the model on purpose:
# it verbalizes them ("۱۴۰۳" -> "hezar @o cehars/d @o se").
_DIGIT_RE: Final[re.Pattern[str]] = re.compile(r"[0-9۰-۹٠-٩]")


class G2PModelUnavailableError(RuntimeError):
    """Raised when the neural backend cannot be initialised.

    Carries actionable instructions rather than a bare ImportError or a
    filesystem traceback, because the two failure modes (missing extra, missing
    weights) have different fixes and neither is guessable from a stack trace.
    """


class _Generator(Protocol):
    """The single method this module needs from a loaded model."""

    def __call__(self, sentences: Sequence[str]) -> list[str]:
        """Return one raw phoneme string per input sentence."""


_INSTALL_HINT: Final[str] = (
    "Install the optional neural G2P extra and download the model:\n"
    '  pip install -e ".[g2p-neural]"\n'
    "  git clone https://huggingface.co/MahtaFetrat/Homo-GE2PE-Persian\n"
    "  unzip -q Homo-GE2PE-Persian/model-weights/homo-ge2pe.zip -d homo-ge2pe\n"
    f"  export {MODEL_PATH_ENV}=$PWD/homo-ge2pe\n"
    "Or set AVA_G2P_BACKEND=legacy to use the rule-based phonemizer."
)


def resolve_model_path(explicit: str | Path | None = None) -> Path:
    """Return the model directory, or raise with instructions.

    Resolution order: ``explicit`` argument, then :data:`MODEL_PATH_ENV`, then
    the settings' configured path.
    """
    candidate = explicit or os.environ.get(MODEL_PATH_ENV)
    if candidate is None:
        from ava_voice.core.config import get_settings

        candidate = get_settings().g2p_model_path
    if not candidate:
        raise G2PModelUnavailableError(
            f"No G2P model path configured. Set {MODEL_PATH_ENV} or "
            f"AVA_G2P_MODEL_PATH in the environment.\n\n{_INSTALL_HINT}"
        )

    path = Path(candidate).expanduser()
    if not path.is_dir():
        raise G2PModelUnavailableError(
            f"G2P model directory not found: {path}\n\n{_INSTALL_HINT}"
        )
    if not (path / "config.json").exists():
        raise G2PModelUnavailableError(
            f"{path} exists but holds no config.json, so it is not an unpacked "
            f"transformers checkpoint.\n\n{_INSTALL_HINT}"
        )
    return path


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _load_official(model_path: Path) -> _Generator | None:
    """Load through the upstream ``GE2PE`` wrapper, or return ``None``.

    Preferred when available: the wrapper runs Parsivar's normalizer first, which
    verbalizes numerals (``۱۴۰۳`` → ``هزار و چهارصد و سه``) before the model sees
    them. Driving the T5 directly returns ``heg`` for that input.

    Three things must hold, and none is guaranteed: ``GE2PE.py`` on ``sys.path``,
    the unpacked ``Parsivar`` package importable, and — because Parsivar loads its
    resources from the *relative* path ``./Parsivar/resource/...`` — the process
    working directory must contain ``Parsivar/``. Any of these failing is normal,
    not exceptional, so we log at info level and let the caller fall back to
    direct loading.
    """
    try:
        from GE2PE import GE2PE  # type: ignore[import-not-found]

        engine = GE2PE(model_path=str(model_path))
    except Exception as exc:
        # Import errors, Parsivar's own import chain, or its relative-path
        # resource loading — all mean "wrapper unavailable here".
        _logger.info(
            "Upstream GE2PE wrapper unavailable; loading the checkpoint directly "
            "(numeral verbalization will be unavailable)",
            extra={"reason": f"{type(exc).__name__}: {exc}"},
        )
        return None

    def generate(sentences: Sequence[str]) -> list[str]:
        # use_rules=False is deliberate: rules() strips the ezafe marker.
        return engine.generate(list(sentences), use_rules=False)

    return generate


def _load_direct(model_path: Path) -> _Generator:
    """Load the T5 checkpoint with transformers only.

    Mirrors ``GE2PE.generate``'s inference settings (no special tokens, 5-beam
    search, ``max_length=512``) but skips Parsivar, so numerals are not
    verbalized. Callers must keep non-Persian tokens away from the model.
    """
    try:
        from transformers import (  # type: ignore[import-not-found]
            AutoTokenizer,
            T5ForConditionalGeneration,
        )
    except ImportError as exc:
        raise G2PModelUnavailableError(
            f"transformers is not installed.\n\n{_INSTALL_HINT}"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = T5ForConditionalGeneration.from_pretrained(str(model_path))
    model.eval()

    def generate(sentences: Sequence[str]) -> list[str]:
        encoded = tokenizer(
            list(sentences),
            padding=True,
            add_special_tokens=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        produced = model.generate(
            encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            num_beams=5,
            min_length=1,
            max_length=512,
            early_stopping=True,
        )
        return [
            text.strip()
            for text in tokenizer.batch_decode(produced, skip_special_tokens=True)
        ]

    return generate


class _LazyModel:
    """Loads the ~230 MB checkpoint on first use, then caches it.

    Importing this module must stay free, so nothing touches the filesystem or
    torch until a caller actually phonemizes something. The lock makes the first
    concurrent call load exactly once rather than N times.
    """

    __slots__ = ("_explicit_path", "_generator", "_lock")

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._explicit_path = model_path
        self._generator: _Generator | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` if the checkpoint is already in memory."""
        return self._generator is not None

    def generator(self) -> _Generator:
        """Return the cached generator, loading it if necessary."""
        if self._generator is not None:
            return self._generator
        with self._lock:
            if self._generator is not None:  # another thread won the race
                return self._generator
            path = resolve_model_path(self._explicit_path)
            _logger.info("Loading neural G2P model", extra={"path": str(path)})
            generator = _load_official(path) or _load_direct(path)
            self._generator = generator
            _logger.info("Neural G2P model ready")
            return generator


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParsedWord:
    """One word parsed out of the model's raw output.

    Attributes
    ----------
    symbols:
        Project-alphabet phoneme symbols.
    ezafe:
        ``True`` when the trailing :data:`EZAFE_MARKER` was present.
    """

    symbols: tuple[str, ...]
    ezafe: bool


def is_model_ready_token(token: str) -> bool:
    """Return ``True`` if ``token`` is safe to send to the model.

    Persian letters or digits only. Everything else — punctuation, Latin, empty
    strings — makes the model degenerate into repeated gibberish, so it is routed
    to the rule-based backend instead.

    Examples
    --------
    >>> is_model_ready_token("سلام")
    True
    >>> is_model_ready_token("۱۴۰۳")
    True
    >>> is_model_ready_token("Django")
    False
    >>> is_model_ready_token("!!!")
    False
    """
    if not token or not token.strip():
        return False
    return bool(_PERSIAN_LETTER_RE.search(token) or _DIGIT_RE.search(token))


def parse_model_word(raw: str) -> ParsedWord:
    """Translate one whitespace-free model word into project symbols.

    Splits off the trailing ezafe marker, then maps each character through
    :data:`_MODEL_SYMBOL_MAP`. Multi-character results (``$`` → ``sh``) become one
    symbol, not two.

    Examples
    --------
    >>> parse_model_word("ketabe1")
    ParsedWord(symbols=('k', 'e', 't', 'a', 'b', 'e'), ezafe=True)
    >>> parse_model_word("ke$v/r").symbols
    ('k', 'e', 'sh', 'v', 'æ', 'r')
    >>> parse_model_word("@/st").symbols
    ("'", 'æ', 's', 't')
    """
    word = raw.strip()
    ezafe = word.endswith(EZAFE_MARKER)
    if ezafe:
        word = word[: -len(EZAFE_MARKER)]

    symbols: list[str] = []
    for char in word:
        if char in _IGNORED_MODEL_CHARS or char.isspace():
            continue
        if char.isdigit():
            # A stray digit anywhere but the end is not a marker we understand;
            # dropping it is safer than emitting it as a phoneme.
            _logger.debug("Dropping unexpected digit in G2P output", extra={"raw": raw})
            continue
        symbols.append(_MODEL_SYMBOL_MAP.get(char, char))
    return ParsedWord(symbols=tuple(symbols), ezafe=ezafe)


def parse_model_output(raw: str) -> list[ParsedWord]:
    """Split a raw model sentence into per-word :class:`ParsedWord` objects."""
    return [parse_model_word(word) for word in raw.split() if word]


def _to_word_phonemes(parsed: ParsedWord) -> WordPhonemes:
    """Wrap a :class:`ParsedWord` in the project's phoneme container.

    The ezafe vowel stays in the token stream because it is spoken; the final
    token is additionally tagged ``is_ezafe`` so acoustics can find it without
    re-deriving which phoneme the linker was.
    """
    count = len(parsed.symbols)
    tokens = tuple(
        PhonemeToken(
            symbol=symbol,
            is_ezafe=parsed.ezafe and index == count - 1,
        )
        for index, symbol in enumerate(parsed.symbols)
    )
    return WordPhonemes(tokens=tokens, ezafe=parsed.ezafe)


def has_explicit_ezafe(word: str) -> bool:
    """Return ``True`` if ``word`` carries a word-final kasra.

    A writer who types ``کتابِ`` is marking ezafe explicitly. That is more
    reliable than inference, and it is information the model cannot use because
    the upstream wrapper strips harakat before inference.

    Examples
    --------
    >>> has_explicit_ezafe("کتابِ")
    True
    >>> has_explicit_ezafe("کتاب")
    False
    """
    return word.strip().endswith(KASRA)


def _apply_explicit_ezafe(parsed: ParsedWord) -> ParsedWord:
    """Force the ezafe flag on, appending the linking vowel when absent."""
    if parsed.ezafe:
        return parsed
    symbols = parsed.symbols
    if not symbols or symbols[-1] != _EZAFE_VOWEL:
        symbols = (*symbols, _EZAFE_VOWEL)
    return ParsedWord(symbols=symbols, ezafe=True)


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
class NeuralPhonemizer:
    """Persian G2P backed by Homo-GE2PE, with a rule-based safety net.

    Interface-compatible with :class:`~ava_voice.linguistics.phonemizer.PersianPhonemizer`,
    so :class:`~ava_voice.linguistics.token_graph.TokenGraphBuilder` accepts either.

    Every failure mode degrades to the rule-based backend rather than raising:
    a token the model should not see, a word-count mismatch, an empty parse, or
    an inference error. The only hard error is a missing/unloadable model, and
    that surfaces on first use with installation instructions.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        fallback: PersianPhonemizer | None = None,
        model: Any | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model_path:
            Directory holding the unpacked checkpoint. Defaults to
            :data:`MODEL_PATH_ENV` then the settings value.
        fallback:
            Rule-based phonemizer used for non-Persian tokens and on any failure.
        model:
            Pre-built lazy model, for tests. Must expose ``generator()``.
        """
        self._fallback = fallback or PersianPhonemizer()
        self._model = model if model is not None else _LazyModel(model_path)

    # -- Public API --------------------------------------------------------- #
    @property
    def fallback(self) -> PersianPhonemizer:
        """Return the rule-based phonemizer backing this instance."""
        return self._fallback

    def analyze_sentence(self, words: Sequence[str]) -> list[WordPhonemes]:
        """Phonemize a whole sentence in one model call.

        Sentence-level by necessity: the model resolves homographs from context,
        and word-by-word calls lose it (``گل`` alone yields ``ole``; in a sentence
        it yields ``gol`` or ``gel`` correctly).

        Non-Persian tokens are held out, phonemized by the fallback, and spliced
        back in position, so the model only ever sees text it handles.
        """
        if not words:
            return []

        eligible = [
            index for index, word in enumerate(words) if is_model_ready_token(word)
        ]
        results: list[WordPhonemes | None] = [None] * len(words)

        if eligible:
            sentence = " ".join(words[index] for index in eligible)
            parsed = self._infer(sentence, expected=len(eligible))
            if parsed is not None:
                for index, word in zip(eligible, parsed):
                    if has_explicit_ezafe(words[index]):
                        word = _apply_explicit_ezafe(word)
                    results[index] = _to_word_phonemes(word)

        for index, word in enumerate(words):
            if results[index] is None:
                results[index] = WordPhonemes(
                    tokens=tuple(self._fallback.analyze_word(word)), ezafe=False
                )
        return [result for result in results if result is not None]

    def analyze_word(self, word: str) -> list[PhonemeToken]:
        """Phonemize a single word.

        Provided for interface parity. Prefer :meth:`analyze_sentence`: in
        isolation the model has no context for homograph disambiguation and no
        following word against which to detect ezafe.
        """
        return list(self.analyze_sentence([word])[0].tokens) if word.strip() else []

    def phonemize_word(self, word: str) -> list[str]:
        """Return the flat phoneme-symbol sequence for a single ``word``."""
        return [token.symbol for token in self.analyze_word(word)]

    def phonemize(self, text: str) -> list[str]:
        """Phonemize whitespace-separated ``text`` into a flat symbol list."""
        symbols: list[str] = []
        for result in self.analyze_sentence(text.split()):
            symbols.extend(result.symbols)
        return symbols

    def warmup(self) -> None:
        """Load the model now instead of on the first phonemization.

        Useful at application startup so the ~1 s load does not land inside a
        request.
        """
        self._model.generator()

    # -- Internals ---------------------------------------------------------- #
    def _infer(self, sentence: str, expected: int) -> list[ParsedWord] | None:
        """Run inference and validate alignment, or return ``None``.

        ``None`` means "use the fallback": the word counts disagree (the model
        may merge ``می روم`` into one word or expand ``۱۴۰۳`` into three), a word
        parsed to no phonemes, or inference itself failed.
        """
        try:
            raw = self._model.generator()([sentence])
        except G2PModelUnavailableError:
            raise
        except Exception:
            _logger.exception(
                "Neural G2P inference failed; using rule-based fallback",
                extra={"sentence": sentence},
            )
            return None

        if not raw or not raw[0].strip():
            _logger.warning(
                "Neural G2P returned nothing; using rule-based fallback",
                extra={"sentence": sentence},
            )
            return None

        parsed = parse_model_output(raw[0])
        if len(parsed) != expected:
            _logger.warning(
                "Neural G2P word count mismatch; using rule-based fallback",
                extra={"expected": expected, "produced": len(parsed)},
            )
            return None
        if any(not word.symbols for word in parsed):
            _logger.warning(
                "Neural G2P produced an empty word; using rule-based fallback",
                extra={"sentence": sentence},
            )
            return None
        return parsed


#: Process-wide default. Construction is free; the model loads on first use.
_default_phonemizer: Final[NeuralPhonemizer] = NeuralPhonemizer()


def neural_phonemize_word(word: str) -> list[str]:
    """Phonemize a single word using the module-level default backend."""
    return _default_phonemizer.phonemize_word(word)


def neural_phonemize(text: str) -> list[str]:
    """Phonemize text using the module-level default backend."""
    return _default_phonemizer.phonemize(text)
