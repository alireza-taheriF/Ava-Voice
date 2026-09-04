"""Phonemizer backend selection.

One factory, :func:`build_phonemizer`, decides between the dependency-free rule
engine and the neural Homo-GE2PE backend based on ``AVA_G2P_BACKEND``.

This lives in its own module so that :mod:`~ava_voice.linguistics.token_graph`
never imports :mod:`~ava_voice.linguistics.neural_phonemizer` unless the neural
backend is actually selected — that module pulls in ``transformers`` lazily, but
its own import still costs a few hundred milliseconds we should not pay by
default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from ava_voice.core.logger import get_logger
from ava_voice.linguistics.phonemizer import PersianPhonemizer, WordPhonemes

_logger = get_logger("linguistics.backend")


@runtime_checkable
class SentencePhonemizer(Protocol):
    """The contract :class:`~ava_voice.linguistics.token_graph.TokenGraphBuilder` needs.

    Sentence-level rather than word-level: the neural backend resolves homographs
    from surrounding context and detects ezafe from the following word, so it must
    see whole sentences. The rule-based backend implements the same shape by
    mapping over words.
    """

    def analyze_sentence(self, words: Sequence[str]) -> list[WordPhonemes]:
        """Return one :class:`WordPhonemes` per input word, in order."""
        ...


def build_phonemizer(
    backend: str | None = None,
    model_path: str | Path | None = None,
) -> SentencePhonemizer:
    """Return the configured phonemizer backend.

    Parameters
    ----------
    backend:
        ``"legacy"`` or ``"neural"``. Defaults to the ``g2p_backend`` setting.
    model_path:
        Checkpoint directory for the neural backend. Defaults to the
        ``AVA_G2P_MODEL_PATH`` environment variable, then the settings value.

    Notes
    -----
    Selecting ``"neural"`` does not load the model — :class:`~ava_voice.linguistics.neural_phonemizer.NeuralPhonemizer`
    loads it on first use. An unknown backend name logs a warning and falls back
    to ``"legacy"`` rather than raising, so a typo in an environment variable
    degrades instead of taking the process down.
    """
    if backend is None:
        from ava_voice.core.config import get_settings

        settings = get_settings()
        backend = settings.g2p_backend
        model_path = model_path or settings.g2p_model_path

    if backend == "legacy":
        return PersianPhonemizer()
    if backend == "neural":
        from ava_voice.linguistics.neural_phonemizer import NeuralPhonemizer

        return NeuralPhonemizer(model_path=model_path)

    _logger.warning(
        "Unknown G2P backend; using the rule-based phonemizer",
        extra={"backend": backend},
    )
    return PersianPhonemizer()
