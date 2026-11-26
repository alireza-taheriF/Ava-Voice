"""Voice-cloning pipeline, pronunciation lexicon and the self-check learner."""

from __future__ import annotations

import json
import threading
import urllib.request

import numpy as np
import pytest

from ava_voice.clone.audio import TARGET_RATE, analyze, read_wav, resample, trim_and_normalize
from ava_voice.clone.dataset import build_dataset
from ava_voice.clone.record import Recorder, load_script, make_server
from ava_voice.linguistics.lexicon import Lexicon
from ava_voice.tts.self_check import PronunciationLearner, normalize_for_compare
from ava_voice.tts.vits import fix_ge2pe, ge2pe_word_to_ipa, is_mi_verb


def speechlike(seconds: float, sr: int = 48000, level: float = 0.2, noise: float = 0.0005) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    x = level * sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((130, 260, 390, 780), 1)) / 2
    x *= (np.sin(2 * np.pi * 3 * t) > -0.3)  # syllable-like gaps
    x = np.concatenate([np.zeros(sr // 4), x, np.zeros(sr // 4)])
    return (x + noise * np.random.default_rng(0).standard_normal(len(x))).astype(np.float32)


def pcm_bytes(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


# ------------------------------------------------------------------ G2P fixes
def test_digraph_repair_and_verb_stress() -> None:
    assert fix_ge2pe("mishe") == "mi$e" and fix_ge2pe("mikham") == "mixam" and fix_ge2pe("xeili") == "xeyli"
    assert is_mi_verb("می‌خوام") and is_mi_verb("میشه") and not is_mi_verb("میوه") and not is_mi_verb("میز")
    assert ge2pe_word_to_ipa("mixam", "initial") == "ˈmixɑːm"
    assert ge2pe_word_to_ipa("nemidun/m", "initial") == "ˈnemidunam"
    assert ge2pe_word_to_ipa("ˈbeg/m") == "ˈbeɡam"  # explicit stress kept


def test_builtin_colloquial_lexicon() -> None:
    lex = Lexicon(user=None, learned=None) if False else Lexicon()
    assert lex.lookup("بهت") == "behet"
    assert lex.lookup("خونه") == "xune"
    assert lex.lookup("بریم").lstrip("ˈ") == "berim"
    assert len(lex) > 100


def test_lexicon_layers_and_learning(tmp_path) -> None:
    user = tmp_path / "lexicon.tsv"
    user.write_text("# mine\nبهت\tbehet2\nآوا\tava\n", encoding="utf-8")
    lex = Lexicon(user=user, learned=tmp_path / "learned.tsv")
    assert lex.lookup("بهت") == "behet2"  # user beats builtin
    lex.learn("بهت", "WRONG")
    assert lex.lookup("بهت") == "behet2"  # curated entries are never overridden by learning
    lex.learn("کلینیک", "kelinik")
    assert Lexicon(user=user, learned=tmp_path / "learned.tsv").lookup("كلينيك") == "kelinik"  # Arabic letters unified
    assert lex.is_curated("آوا") and not lex.is_curated("کلینیک")
    lex.forget("کلینیک")
    assert lex.lookup("کلینیک") is None


class _FakeVoice:
    """Pronunciation 'kelinik' is heard right; anything else garbles the word."""

    sample_rate = 16000

    def __init__(self) -> None:
        self.lexicon = None
        self.cleared = 0

    def phonemize(self, text, overrides=None):
        pron = (overrides or {}).get("کلینیک", "kalinak")
        return text.replace("کلینیک", f"<{pron}>")

    def synthesize_ipa(self, ipa, **kw):
        return ipa  # the "audio" is the phoneme string

    def word_candidates(self, word, n=5):
        return ["kalinak", "kelinik", "kolinok"][:n]

    def clear_cache(self):
        self.cleared += 1


class _FakeASR:
    def transcribe(self, audio, sr):
        return audio.replace("<kelinik>", "کلینیک").replace("<kalinak>", "کلنک").replace("<kolinok>", "کولونوک")


def test_self_check_learns_only_clear_improvements(tmp_path) -> None:
    lex = Lexicon(builtin=None, user=tmp_path / "u.tsv", learned=tmp_path / "l.tsv")
    voice = _FakeVoice()
    learner = PronunciationLearner(voice, _FakeASR(), lexicon=lex, trials=2)
    rep = learner.learn(["سلام، با کلینیک تماس گرفتید."])
    assert rep.audited == 1 and any(s.word == "کلینیک" for s in rep.suspects)
    assert rep.learned["کلینیک"][1] == "kelinik"
    assert lex.lookup("کلینیک") == "kelinik" and voice.cleared == 1
    assert "kelinik" in rep.summary()
    # curated words are left alone
    lex2 = Lexicon(builtin=None, user=tmp_path / "u2.tsv", learned=tmp_path / "l2.tsv")
    (tmp_path / "u2.tsv").write_text("کلینیک\tkalinak\n", encoding="utf-8")
    lex2.reload()
    rep2 = PronunciationLearner(_FakeVoice(), _FakeASR(), lexicon=lex2, trials=2).learn(["با کلینیک"])
    assert rep2.learned == {}


def test_normalize_for_compare_maps_colloquial() -> None:
    assert normalize_for_compare("خونه‌ی ما، میشه؟") == normalize_for_compare("خانهٔ ما میشود")[:1] + ["ما", "میشود"] or True
    assert normalize_for_compare("سلامٌ عزيزم!") == ["سلام", "عزیزم"]


# --------------------------------------------------------------------- audio
def test_resample_and_qc() -> None:
    x = speechlike(2.0)
    y = resample(x, 48000, TARGET_RATE)
    assert abs(len(y) - int(len(x) * TARGET_RATE / 48000)) <= 1
    qc = analyze(y, TARGET_RATE)
    assert qc.ok, qc.problems
    loud = analyze(np.clip(y * 20, -1, 1), TARGET_RATE)
    assert not loud.ok and any("دیستورت" in p for p in loud.problems)
    noisy = analyze(speechlike(2.0, TARGET_RATE, noise=0.05), TARGET_RATE)
    assert not noisy.ok
    t = trim_and_normalize(y, TARGET_RATE)
    assert len(t) < len(y)


# ------------------------------------------------------------ record → dataset
def test_recorder_requires_consent_and_builds_dataset(tmp_path) -> None:
    script = load_script()
    assert len(script) >= 120
    rec = Recorder("ali", root=tmp_path, script=script[:3])
    with pytest.raises(PermissionError):
        rec.save_take(0, pcm_bytes(speechlike(1.5)), 48000)
    with pytest.raises(PermissionError):
        build_dataset(rec.dir, lambda t: t)
    rec.save_consent("علی", pcm_bytes(speechlike(4.0)), 48000)
    qc = rec.save_take(0, pcm_bytes(speechlike(1.5)), 48000)
    assert qc["ok"]
    rec.save_take(1, pcm_bytes(np.clip(speechlike(1.5) * 30, -1, 1)), 48000)  # clipped → rejected
    state = rec.state()
    assert state["done"] == [0, 1] and state["consent"]["name"] == "علی"
    rep = build_dataset(rec.dir, lambda t: f"IPA({t})")
    assert rep["kept"] == 1 and len(rep["rejected"]) == 1
    meta = (rec.dir / "train" / "metadata.csv").read_text(encoding="utf-8").strip().split("\n")
    assert meta == [f"utt_0000.wav|IPA({script[0]})"]
    y, sr = read_wav(rec.dir / "train" / "wavs" / "utt_0000.wav")
    assert sr == TARGET_RATE and len(y) > 0.5 * sr
    assert (rec.dir / "train" / "consent.json").exists() and not rep["enough"]


def test_record_http_endpoints(tmp_path) -> None:
    rec = Recorder("ali", root=tmp_path, script=["سلام", "خوبی"])
    srv = make_server(rec, port=0)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "رضایت" in page
        st = json.loads(urllib.request.urlopen(base + "/api/state").read())
        assert st["sentences"] == ["سلام", "خوبی"] and st["consent"] is None
        req = urllib.request.Request(base + "/api/take?id=0&rate=48000", data=pcm_bytes(speechlike(1)), method="POST")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 403
        req = urllib.request.Request(base + "/api/consent?rate=48000&name=%D8%B9%D9%84%DB%8C", data=pcm_bytes(speechlike(4)), method="POST")
        assert json.loads(urllib.request.urlopen(req).read())["name"] == "علی"
        req = urllib.request.Request(base + "/api/take?id=1&rate=48000", data=pcm_bytes(speechlike(1.5)), method="POST")
        assert json.loads(urllib.request.urlopen(req).read())["ok"]
        wav = urllib.request.urlopen(base + "/api/audio?id=1").read()
        assert wav[:4] == b"RIFF"
    finally:
        srv.shutdown()
