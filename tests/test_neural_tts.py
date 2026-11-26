"""Tests for the NumPy Homo-GE2PE port and the neural VITS voice path."""

from __future__ import annotations

import json
import math
import struct

import numpy as np
import pytest

from ava_voice.linguistics.ge2pe_numpy import (
    _bucket,
    byt5_decode,
    byt5_encode,
    load_safetensors,
)
from ava_voice.tts import models
from ava_voice.tts.vits import ge2pe_word_to_ipa


# --------------------------------------------------------------------------- #
# Pure functions (always run)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("word", "ipa"),
    [
        ("s/lam", "saˈlɑːm"),
        ("ketab", "keˈtɑːb"),
        ("nob/te1", "noˈbate"),  # ezafe -e stays unstressed
        ("xaneye1", "xɑːˈneje"),  # ezafe -ye after a vowel
        ("be", "be"),  # clitic: no stress
        ("@/st", "ʔast"),
        ("$/b", "ˈʃab"),
        ("cetor", "tʃeˈtor"),
        ("vaj", "ˈvɑːdʒ"),
        ("mo;e", "moˈʒe"),
    ],
)
def test_ge2pe_word_to_ipa(word: str, ipa: str) -> None:
    assert ge2pe_word_to_ipa(word) == ipa


def test_byt5_roundtrip() -> None:
    text = "سلام دنیا"
    assert byt5_decode(byt5_encode(text)) == text
    assert min(byt5_encode(text)) >= 3


def _ref_bucket(rel: int, bidirectional: bool, nb: int = 32, md: int = 128) -> int:
    """Scalar transcription of transformers' ``_relative_position_bucket``
    (``rel = memory_position - query_position``)."""
    ret = 0
    if bidirectional:
        nb //= 2
        ret += nb if rel > 0 else 0
        n = abs(rel)
    else:
        n = -min(rel, 0)
    max_exact = nb // 2
    if n < max_exact:
        return ret + n
    val = max_exact + int(math.log(n / max_exact) / math.log(md / max_exact) * (nb - max_exact))
    return ret + min(val, nb - 1)


@pytest.mark.parametrize("bidirectional", [True, False])
def test_relative_position_buckets_match_reference(bidirectional: bool) -> None:
    rel = np.arange(-300, 301)
    # our function takes memory - context (key - query), HF's takes the same.
    ours = _bucket(rel.copy(), bidirectional, 32, 128)
    ref = np.array([_ref_bucket(int(r), bidirectional) for r in rel])
    assert np.array_equal(ours, ref)


def test_load_safetensors_roundtrip(tmp_path) -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.array([1.5, -2.0], dtype=np.float16)
    blobs = a.tobytes() + b.tobytes()
    header = {
        "a": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, a.nbytes]},
        "b": {"dtype": "F16", "shape": [2], "data_offsets": [a.nbytes, a.nbytes + b.nbytes]},
        "__metadata__": {"format": "pt"},
    }
    h = json.dumps(header).encode()
    path = tmp_path / "t.safetensors"
    path.write_bytes(struct.pack("<Q", len(h)) + h + blobs)
    out = load_safetensors(path)
    assert np.array_equal(out["a"], a)
    assert out["b"].tolist() == [1.5, -2.0]


def test_voice_registry() -> None:
    assert models.DEFAULT_VOICE in models.VOICES
    for v in models.VOICES.values():
        assert v.url.startswith("https://github.com/k2-fsa/sherpa-onnx/releases/download/")


# --------------------------------------------------------------------------- #
# With real weights (skipped unless downloaded)
# --------------------------------------------------------------------------- #
def _have_g2p() -> bool:
    return (models.g2p_dir() / "model.safetensors").exists()


def _have_voice() -> bool:
    return any(models.voice_dir().glob("*.onnx")) if models.voice_dir().exists() else False


@pytest.mark.skipif(not _have_g2p(), reason="Homo-GE2PE weights not downloaded")
def test_numpy_ge2pe_reproduces_published_homograph_examples() -> None:
    from ava_voice.linguistics.ge2pe_numpy import NumpyT5

    m = NumpyT5(models.g2p_dir())
    run = lambda s: byt5_decode(m.generate(byt5_encode(s)))  # noqa: E731
    assert run("گل زیبا است").split()[0] == "gol"
    assert "gel" in run("کفشم پر از گل شد").split()
    assert run("او دیروز مرد").split()[-1] == "mord"
    assert run("مرد به خانه رفت").split()[0] == "m/rd"


@pytest.mark.skipif(not (_have_g2p() and _have_voice()), reason="neural TTS models not downloaded")
def test_persian_vits_end_to_end() -> None:
    from ava_voice.core.speech_plan import plan_speech
    from ava_voice.tts.vits import PersianVITS

    tts = PersianVITS()
    ipa = tts.phonemize("قیمت ۴۵۰,۰۰۰ تومان است.")
    assert "panˈdʒɑːh heˈzɑːr" in ipa and ipa.endswith(".")
    assert not any(ch.isdigit() for ch in ipa)
    wav = tts.synthesize("سلام، خوش آمدید.")
    assert wav.dtype == np.float32 and 0.6 < len(wav) / tts.sample_rate < 4.0
    assert float(np.sqrt(np.mean(wav**2))) > 0.02
    plan = plan_speech("سلام! امروز هوا خیلی خوب است.")
    assert len(tts.synthesize_plan(plan)) > len(wav)
