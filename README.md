# Ava Voice

Emotion-aware Persian (Farsi) text-to-speech infrastructure.

Ava Voice turns Persian text into a **symbolic speech plan**: a normalized token
graph, a detected emotional vector, a prosody profile, and a frame-level acoustic
timeline. Every stage is deterministic, rule-based, immutable and serializable —
no model weights, no randomness, no network calls.

> **Status: natural Persian speech.** Text runs through Ava's frontend
> (normalization, number/date verbalization) and **Homo-GE2PE** G2P (short
> vowels, ezafe, homographs), re-implemented in pure NumPy. The phonemes then
> go to a **neural Persian VITS voice** on onnxruntime. Intelligibility is
> measured, not assumed: Whisper-turbo recovers the text with **≈5–6 %
> character error** on 20 phone-agent sentences. None of this needs PyTorch.
> The older rule-based formant vocoder is kept as a dependency-free fallback.

Speech is **expressive**. Ava's emotion and prosody engines drive tempo,
pitch (formant-preserving PSOLA), intonation contour, loudness, pauses and
breathing for every sentence. The voice takes breaths like a person, and it
can produce backchannels ("بله"، "آهان"، "اوهوم"، "خب") for conversational
agents.

```bash
pip install -e ".[neural]"                 # numpy + onnxruntime
python -m ava_voice.tts.models fetch       # G2P (≈33 MB) + default voice (≈60 MB)
python -m ava_voice.tts.vits --show-ipa "سلام! قیمت ۴۵۰,۰۰۰ تومان است." out.wav
python -m ava_voice.tts.expressive "عالیه! خیلی خوشحالم. متأسفانه دکتر امروز نیستند." out.wav
```

---

## Pipeline

```
text
  │
  ├─ linguistics/  normalize → tokenize → resolve → phonemize → stress   (Phase 1A)
  │                                                    ↓
  │                                              TokenGraph
  ├─ emotion/      lexicon → valence/arousal → intensity → fusion → memory (Phase 1B)
  │                                                    ↓
  │                                            EmotionalVector
  ├─ prosody/      emotion → pitch / tempo / energy / pause / emphasis    (Phase 1C)
  │                                                    ↓
  │                                             ProsodyProfile
  └─ acoustics/    align phonemes → durations → F0 + energy curves        (Phase 2A)
                                                       ↓
                                               AcousticProfile
  └─ tts/          formant_vocoder: glottal source → formant filters → PCM (fallback)

Neural voice path (tts/vits.py):
text → normalize + verbalize → Homo-GE2PE (NumPy T5) → espeak-style IPA + stress
     → VITS voice (onnxruntime) → 22.05 kHz PCM;  prosody speaking_rate → length_scale
```

`core/speech_plan.py` now walks that stack **per sentence** (with emotional
memory across sentence boundaries) and is the public compiler:

```python
from ava_voice.core.speech_plan import plan_speech

plan = plan_speech("سلام! کتابِ علی را خواندم.")
plan.ezafe_count          # linker events, not lexical final /e/
plan.sentences[1].acoustic.ezafe_segments
plan.sentences[1].acoustic.pitch_curve_hz  # absolute F0
```

Each layer consumes the typed output of the previous one and emits a frozen
dataclass with `to_dict()` / `from_dict()`, so any stage can be snapshot-tested
or persisted independently.

## Repository layout

The repository root *is* the `ava_voice` package.

| Path | Lines | What it is |
| --- | --- | --- |
| `linguistics/` | ~2.4k | Persian normalizer, token graph, rule-based + neural G2P, stress, punctuation, homographs |
| `emotion/` | ~2.0k | Lexicon classifier, valence/arousal/dominance mapping, intensity, context, short-term memory |
| `prosody/` | ~1.7k | Emotion → pitch, tempo, energy, pause, emphasis, contour shape |
| `acoustics/` | ~2.0k | Phoneme alignment, per-phoneme durations, F0/energy curves, pause shaping, elongation |
| `core/` | ~1.4k | Settings, logging, cache, sessions, auth, pipeline primitives, **`plan_speech()` compiler** |
| `corpus/` | ~0.6k | Licensed-only ingest (owned seed + Common Voice `fa` CC0 sentences). Refuses podcast/YouTube rips |
| `main.py` | 231 | FastAPI app: `/health`, `/api/v1/status`, and a WebSocket transport endpoint |
| `tests/` | ~2.2k | Deterministic unit tests across all layers |
| `infra/` | — | Dockerfile + docker-compose (API + Redis) |
| `api/` `tts/` `clone/` `realtime/` `sdk/` `training/` | ~10 each | Reserved placeholders — no runtime logic |

### Neural voice

| Piece | What it is |
| --- | --- |
| `linguistics/ge2pe_numpy.py` | Homo-GE2PE (2+2-layer byte T5) forward pass and KV-cached beam search in NumPy. Output matches the published model, including homographs: `گل زیبا است` → *gol*, `کفشم پر از گل شد` → *gel*, `او دیروز مرد` → *mord*. About 0.1 s per sentence on CPU. It is also the new default backend of `neural_phonemizer`, so no torch is needed. |
| `tts/vits.py` | `PersianVITS`: clause splitting, GE2PE → IPA mapping with stem-final stress (unstressed ezafe and clitics), `و` cliticised as */-o/*, VITS synthesis, `synthesize_plan()` applying the emotion-driven `speaking_rate`. Real-time factor ≈ 0.1 on 2 CPU cores. |
| `tts/models.py` | Voice registry and downloader (`fetch`, `list`). Voices are Persian Piper/VITS models mirrored on the sherpa-onnx GitHub releases. |

### Expressive speech (`tts/expressive.py`, `tts/dsp.py`)

| Ava prosody field | What you hear | How |
| --- | --- | --- |
| `speaking_rate` | faster when happy, slower when sad | VITS `length_scale` |
| `pitch_base` | higher voice for joy, lower for sadness | TD-PSOLA mean shift; formants stay put, so no chipmunk effect |
| `pitch_variance` | lively vs. flat melody | VITS `noise_scale`/`noise_w` + PSOLA widening of the F0 range around its mean |
| `contour_shape` | rising (excited, questioning), falling, oscillating, broken (agitated) | time-varying PSOLA ratio across the sentence |
| `energy_level` | louder / softer | gain |
| `pause_duration`, `pause_density` | longer gaps, more breaths | silence + breath placement |

Breaths (`dsp.breath`) are shaped turbulence through a neutral vocal-tract
envelope. An inhale swells and stops; an exhale (sigh) fades. They sit about
24 dB below speech, before long sentences, after long pauses, and as a sigh
before sad sentences. `ExpressiveVoice.backchannel()` renders cached
reactions that a dialog system can play the moment the other person stops
talking.

Emotion does not cost intelligibility. On the 20-sentence Whisper test,
expressive rendering measured **5.1 % CER** against 5.2 % for plain
(`scripts/eval_intelligibility.py --expressive`). The emotion lexicon now also
covers service and phone vocabulary (`عالیه`، `متأسفانه`، `ببخشید`، `حتماً`،
`نگران نباشید` …).

### Conversational pronunciation and self-correction

People don't talk the way they write, and phone conversations are colloquial.
On 20 everyday sentences (`scripts/data/colloquial_fa.txt`) the Whisper
round-trip error fell from **27 % to 10 %**. Part of the 27 % was Whisper repeating itself on short clips, which is fixed as well. Four changes did the rest:

* **G2P repairs.** The model sometimes emits ASCII digraphs on colloquial
  input (`mishe`, `mikham`). These are mapped back (`mi$e` → *miʃe*).
* **Verb stress.** Verbs with mi-/nemi- are stressed on the prefix
  (*ˈmixɑːm*, *ˈnemidunam*), as in speech, not on the last syllable.
* **Colloquial lexicon** (`linguistics/data/colloquial_lexicon.tsv`, ~150
  entries): `بهت` → *behet*, `خونه` → *xune*, `بریم` → *ˈberim*, `می‌کنی` →
  *mikoni* … Add your own entries in `~/.config/ava-voice/lexicon.tsv`.
* **Self-correction** (`tts/self_check.py`). Ava speaks a sentence, Whisper
  listens, and each misheard word is retried with the G2P's n-best readings
  and vowel variants inside the same sentence. A variant is learned
  (`learned_lexicon.tsv`) only if it at least halves the error *in every
  trial*. Human-curated entries always win. Command:
  `python -m ava_voice.tts.self_check sentences.txt --whisper-dir …`.

**Delivery.** `ExpressiveVoice(speed=…, pitch_semitones=…)` sets the global
pace and pitch on top of emotion. The default voice reads at about 4.3
syllables/s with a median F0 of 103 Hz, which sounds slow and old.
Conversational settings of speed 1.2 and +2 semitones raise that to about 5.2
syllables/s and ~116 Hz, at 6.4 % CER (5.1 % at reading pace).

### Clone a voice (`clone/`)

To build a personal voice: record about 15 minutes in the browser, with a
spoken consent step and live QC (`python -m ava_voice.clone.record`). Next,
build a Piper training set phonemized exactly like Ava speaks
(`clone.dataset`). Then fine-tune Piper's Persian checkpoint on a GPU and
install the result (`clone.install`). The full guide is in
[`clone/TRAINING.md`](clone/TRAINING.md). The training itself cannot run in
this repository's CI, because it needs PyTorch and a GPU.

**How the defaults were chosen.** Each candidate went through synthesize →
Whisper-turbo → character error rate. The test was 20 sentences
(`scripts/eval_intelligibility.py`; Whisper runs in NumPy + onnxruntime via
`asr/whisper_onnx.py`):

| Voice | CER (Piper noise defaults) |
| --- | --- |
| `ganji` (default) | 8.9 % → **5.2–6.2 %** with Ava's tuned noise (0.35 / 0.55) |
| `haaniye` (female, low quality) | 11.1 % |
| `ganji_adabi` | 11.2 % |
| `gyro` | 12.1 % |
| `reza_ibrahim` | 20.4 % |
| `amir` | 23.8 % |

Mapping choices were tested the same way on `gyro`. Stress marks helped
(10.9 % vs. 11.8 % without). Long-marked `iː/uː`, a tapped `ɾ`, and dropping
word-initial `ʔ` all hurt. The CER includes Whisper's own mistakes, so it is an
upper bound on the TTS error. VITS is stochastic, so expect ±1 % between runs.

### Render to audio (formant fallback)

```python
from ava_voice.core.speech_plan import plan_speech
from ava_voice.tts.formant_vocoder import synthesize_plan, to_wav_bytes

wav = to_wav_bytes(synthesize_plan(plan_speech("سلام! قیمت ۲۵۰ هزار تومان است.")))
open("out.wav", "wb").write(wav)
```

or `python -m ava_voice.tts.formant_vocoder "متن" out.wav`. Pure standard
library, deterministic for a fixed `seed`, about 30× faster than real time on
one CPU core.

## Install

Requires Python 3.11+.

```bash
git clone <repo-url> ava_voice && cd ava_voice
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Configuration is environment-driven; copy `.env.example` to `.env` and edit.
Every variable has a default, so `.env` is optional.

## Run

```bash
uvicorn ava_voice.main:app --reload      # or: ava-voice
curl localhost:8000/health
```

Or with Redis, from `infra/`:

```bash
cd infra && docker compose up --build
```

### Authentication

Set `AVA_API_KEY` to require an `X-API-Key` header on protected routes. When it
is unset (the default) auth is disabled and a warning is logged at startup —
convenient locally, and **refused outright in production**: `create_app` raises
rather than serve an unauthenticated API when `AVA_ENVIRONMENT=production`.

```bash
AVA_API_KEY=secret uvicorn ava_voice.main:app
curl localhost:8000/health                                    # public (probes)
curl -H "X-API-Key: secret" localhost:8000/api/v1/status      # protected
```

`/health` stays public by design so container and load-balancer probes work; it
exposes only the service name, version and environment.

WebSockets authenticate at handshake, via the same header or an `api_key` query
parameter for browser clients that cannot set headers (the query form appears in
access logs, so prefer the header). The transport enforces
`AVA_WS_MAX_MESSAGE_BYTES` (oversized frames close with `1009`) and
`AVA_WS_MAX_CONNECTIONS` (handshakes at capacity close with `1008`). The
connection limit is per process — behind multiple workers each has its own.

## Test

```bash
pytest                        # config lives in pyproject.toml
pytest --cov=ava_voice        # with coverage
pytest -m requires_model      # integration tests needing the G2P checkpoint
```

Tests marked `requires_model` are excluded by default (they need a ~229 MB
download); see [Neural G2P](#neural-g2p-homo-ge2pe).

Without an editable install, set the path explicitly:

```bash
cd .. && PYTHONPATH="$PWD" python -m pytest ava_voice/tests -q
```

## Neural G2P (Homo-GE2PE)

The default `legacy` backend is a rule engine with a 10-word lexicon. Where it
cannot resolve a short vowel it emits a `V` placeholder — on common vocabulary
that happens in **66% of words** (`مدرسه` → `mVdVrVsVh`). Nothing downstream ever
resolves those, so the phoneme stream is not speakable.

The `neural` backend replaces that with
[Homo-GE2PE](https://huggingface.co/MahtaFetrat/Homo-GE2PE-Persian) (MIT
licensed) — a byte-level T5 fine-tuned on HomoRich, reporting 3.98% PER and 76.9%
homograph accuracy on SentenceBench. Measured on the same vocabulary: **0
placeholders**, and it also recovers ezafe and verbalizes numerals.

```
             legacy                  neural
مدرسه        mVdVrVsVh               mædrese
بزرگ         bVzVrVg                 bozorg
کشور         kVshvVr                 keshvær
دو           dv                      do
فروش         fVrvVsh                 forush
```

### Install

```bash
pip install -e ".[g2p-neural]"        # torch, transformers, unidecode, nltk

git clone https://huggingface.co/MahtaFetrat/Homo-GE2PE-Persian
unzip -q Homo-GE2PE-Persian/model-weights/homo-ge2pe.zip -d homo-ge2pe

# Optional but recommended — enables numeral verbalization:
unzip -q Homo-GE2PE-Persian/assets/Parsivar.zip
cp Homo-GE2PE-Persian/assets/GE2PE.py .
sed -i 's/from collections import Iterable/from collections.abc import Iterable/' \
    Parsivar/token_merger.py

export AVA_G2P_BACKEND=neural
export AVA_G2P_MODEL_PATH=$PWD/homo-ge2pe
```

`GE2PE.py` and `Parsivar/` must sit in the working directory — Parsivar loads its
resources from relative paths. Without them the checkpoint still loads (straight
through transformers) but numerals are not verbalized; the fallback is automatic
and logged.

### Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `AVA_G2P_BACKEND` | `legacy` | `legacy` (rule-based, no dependencies) or `neural` |
| `AVA_G2P_MODEL_PATH` | — | Directory holding the unpacked checkpoint |

`legacy` is the default so a clean checkout, the test suite and CI all work with
no model download. Selecting `neural` does not load anything at import: the
~229 MB checkpoint loads on the first phonemization (~0.3 s) and is then cached.
Call `NeuralPhonemizer.warmup()` at startup to move that cost out of the first
request. A missing or unreadable model raises `G2PModelUnavailableError` with
install instructions rather than a bare traceback.

### Behaviour worth knowing

- **Sentence-level inference.** Homographs need context: `گل` alone returns
  nonsense (`ole`), but `گل زیبا است` → `gol` (flower) and `کفشم پر از گل شد` →
  `gel` (mud). The token graph therefore issues one model call per sentence.
- **Ezafe is an explicit field**, not a trailing letter. `TOKEN.ezafe` and
  `PHONEME.is_ezafe` carry it, because a final `e` is otherwise ambiguous:
  `خانه` → *xane* (lexical) vs `خانه بزرگ` → *xane-ye* (linker). A written kasra
  (`کتابِ`) forces the flag on — the upstream wrapper strips harakat before
  inference, so the model cannot see it.
- **Non-Persian input never reaches the model.** On empty, punctuation-only or
  Latin input it degenerates into hundreds of tokens of gibberish, so those go to
  the rule engine instead.
- **Every failure degrades, none crashes.** Word-count mismatch (the model may
  merge `می روم` or expand `۱۴۰۳`), empty output, or an inference error all fall
  back to the rule engine per sentence.
- **New phoneme symbol.** The neural backend distinguishes short /æ/ from long
  /ɑː/ and emits `æ` for the former — a distinction the rule engine collapses to
  `a`. Duration, stress and elongation tables treat `æ` as short.

## Licensed corpus

```bash
python -m ava_voice.corpus              # seed + Common Voice fa sentences (CC0)
python -m ava_voice.corpus --offline    # owned seed only
```

Writes `data/corpus.jsonl`, `data/annotations.jsonl`, `data/calibration.json`.
Commercial podcasts and YouTube / Aparat rips are **refused** — see
`PATENT_ROADMAP.md` Phase 1. Audio clips are a later, explicit download.

## Using the layers directly

```python
from ava_voice.linguistics.token_graph import build_token_graph
from ava_voice.emotion.emotion_fusion import analyze_once
from ava_voice.prosody.prosody_engine import generate_prosody
from ava_voice.acoustics.acoustic_engine import AcousticEngine

text = "خیلی خوشحالم! امروز عالی بود."
graph = build_token_graph(text)
emotion = analyze_once(text)
prosody = generate_prosody(graph, emotion.vector)
acoustic = AcousticEngine().generate(graph, prosody.profile)

print(emotion.vector.dominant_emotion)      # PrimaryEmotion.JOY
print(prosody.profile.speaking_rate)        # 1.291
print(acoustic.total_duration_ms)           # 2649.4
```

## Current status

**What works today**

- **Persian normalization** — Arabic↔Persian character unification (`ي`→`ی`,
  `ك`→`ک`), ZWNJ binding, tatweel stripping, digit unification, abbreviation
  expansion, punctuation folding. The most complete module in the project.
- **Number / date / time verbalization** (`linguistics/number_verbalizer.py`) —
  runs inside the normalizer, so no digit glyph reaches G2P: cardinals up to
  10¹⁵, thousands separators, decimals, `%`, ranges, ordinals, clock times,
  Jalali and Gregorian `YYYY/MM/DD` dates, and leading-zero strings (Iranian
  mobile numbers in 4-3-2-2 grouping).
- **Neural Persian voice** — `tts/vits.py`: GE2PE-in-NumPy G2P + VITS on
  onnxruntime, ≈5–6 % Whisper round-trip CER, RTF ≈ 0.1 on CPU.
- **Formant fallback** — `tts/formant_vocoder.py` renders an `AcousticProfile`
  with the plan's exact timing and F0 (tested by autocorrelation) to 16-bit PCM.
- **Token graph** — typed, deterministic, serializable; thread-safe builder.
- **Emotion detection** — 7 channels + valence/arousal/dominance + intensity,
  with negation handling and short-term emotional memory.
- **Prosody + acoustics** — a full symbolic speech plan: per-phoneme durations,
  pause segments, stress boosts, elongation, F0/energy curves on a 10 ms grid,
  **ezafe linker segments** (duration / local F0 / energy by formality × affect),
  and a speaker-baseline hertz curve.
- **`plan_speech()`** — sentence split + emotion memory + the full stack.
- **Licensed corpus ingest** — owned evaluation sentences plus optional Common
  Voice Persian prompts (CC0). Commercial podcasts and YouTube rips are refused.
- **API** — `/health`, an authenticated `/api/v1/status`, and a WebSocket
  transport with API-key handshake auth plus enforced frame-size and
  connection limits.

**What does not work — and why**

1. **The neural voice follows the prosody plan per sentence, not per
   phoneme.** Tempo, pitch level and range, contour, loudness and breathing
   are applied, but Ava's per-phoneme durations and word-level emphasis only
   reach the formant fallback. Conditioning a model on the plan, through a
   fine-tune or an F0-controllable vocoder, is the next research step. Voice
   choice is limited to existing Persian VITS voices, mostly male. A
   high-quality female voice needs a dataset and a fine-tune.
2. **Grapheme-to-phoneme depends on which backend you run.** With the default
   `legacy` rule engine, 66% of common words carry an unresolved `V` placeholder
   and the stream is not speakable. With `neural` that drops to 0% — but it costs
   a torch dependency and a 229 MB download. See
   [Neural G2P](#neural-g2p-homo-ge2pe).
3. **Ezafe only under the neural backend.** `TOKEN.ezafe` is always `False` on
   `legacy`: undiacritized orthography does not mark it and the rule engine has no
   way to infer it.
4. **Verbalization is rule-based.** Numbers are now spelled out ahead of G2P
   (see above), but there is no ezafe-aware ordinal placement, no unit
   expansion (`کیلو`, `km`) and no Latin-script handling.
5. **Emotion coverage is narrow.** ~92 lexicon stems with exact-string matching
   and no morphological analysis, so inflected and colloquial forms (`خوبه`,
   `غمگینم`, `آرومم`) are missed. Sarcasm inverts.
6. **Prosody is uncalibrated.** ~86 hand-picked constants with no reference
   corpus. No speaker F0 baseline exists, so semitone offsets cannot be resolved
   to hertz.
7. **One profile per utterance.** Input is not split into sentences, so a mixed
   passage receives a single emotional reading.
8. **`core/pipeline.py` and `core/router.py` are still unused.** The real
   consumer is `core/speech_plan.py` (`plan_speech`), not those abstractions.

### To reach audible speech

| Priority | Work | Rough size |
| --- | --- | --- |
| 1 | ~~Neural voice~~ — done (GE2PE + VITS). Next: condition it on Ava's F0/duration plan | large |
| 2 | ~~Number / date verbalization ahead of G2P~~ — done; units and Latin script remain | small |
| 3 | Sentence segmentation with per-sentence prosody | small |
| 4 | Broaden the emotion lexicon, or swap in a fine-tuned classifier | medium |
| 5 | Calibrate the prosody constants against a Persian corpus | large |

~~Resolve the `V` placeholder~~ — done via the neural backend.

Two viable routes for (1): drive an existing Persian TTS engine with the
duration and pause timeline as SSML-style control input (fast, weeks), or train a
FastSpeech2-class acoustic model plus a HiFi-GAN vocoder (slow, months, and it
needs the current F0/energy curves reworked to be phoneme-conditioned).

## Conventions

- Python 3.11, `from __future__ import annotations` everywhere, full type hints.
- Domain value objects are `@dataclass(frozen=True)` with bounds clamped in
  `__post_init__`; pydantic is used only for settings.
- Every layer is deterministic — no clocks, no RNG, no I/O in the hot path.
- Stages are constructor-injected so they can be swapped or mocked.

## License

Proprietary — all rights reserved.
