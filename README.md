# Ava Voice

Emotion-aware Persian (Farsi) text-to-speech infrastructure.

Ava Voice turns Persian text into a **symbolic speech plan**: a normalized token
graph, a detected emotional vector, a prosody profile, and a frame-level acoustic
timeline. Every stage is deterministic, rule-based, immutable and serializable —
no model weights, no randomness, no network calls.

> **Status: pre-audio.** The pipeline runs end to end and is covered by tests,
> but it does **not** synthesize sound yet. See [Current status](#current-status).

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
                                               AcousticProfile  ──▶ ✗ no vocoder yet
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
| `core/` | ~1.2k | Settings (pydantic-settings), logging, cache (Redis + in-memory), sessions, API-key auth + WebSocket limits, pipeline/router primitives |
| `main.py` | 231 | FastAPI app: `/health`, `/api/v1/status`, and a WebSocket transport endpoint |
| `tests/` | ~2.2k | Deterministic unit tests across all layers |
| `infra/` | — | Dockerfile + docker-compose (API + Redis) |
| `api/` `tts/` `clone/` `realtime/` `sdk/` `training/` | ~10 each | Reserved placeholders — no runtime logic |

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
- **Token graph** — typed, deterministic, serializable; thread-safe builder.
- **Emotion detection** — 7 channels + valence/arousal/dominance + intensity,
  with negation handling and short-term emotional memory.
- **Prosody + acoustics** — a full symbolic speech plan: per-phoneme durations,
  pause segments, stress boosts, elongation, and F0/energy curves on a 10 ms grid.
- **API** — `/health`, an authenticated `/api/v1/status`, and a WebSocket
  transport with API-key handshake auth plus enforced frame-size and
  connection limits.

**What does not work — and why**

1. **No audio output.** There is no vocoder, no acoustic model, no PCM path. The
   `AcousticProfile` is a JSON control plan, not a waveform. **This is now the
   main blocker.**
2. **Grapheme-to-phoneme depends on which backend you run.** With the default
   `legacy` rule engine, 66% of common words carry an unresolved `V` placeholder
   and the stream is not speakable. With `neural` that drops to 0% — but it costs
   a torch dependency and a 229 MB download. See
   [Neural G2P](#neural-g2p-homo-ge2pe).
3. **Ezafe only under the neural backend.** `TOKEN.ezafe` is always `False` on
   `legacy`: undiacritized orthography does not mark it and the rule engine has no
   way to infer it.
4. **Number verbalization is partial.** The neural backend expands `۱۴۰۳` to
   *hezar o chaharsad o se*, but that is 3 words for 1 input token, which trips the
   word-count guard and falls back. A verbalizer ahead of G2P is still needed.
5. **Emotion coverage is narrow.** ~92 lexicon stems with exact-string matching
   and no morphological analysis, so inflected and colloquial forms (`خوبه`,
   `غمگینم`, `آرومم`) are missed. Sarcasm inverts.
6. **Prosody is uncalibrated.** ~86 hand-picked constants with no reference
   corpus. No speaker F0 baseline exists, so semitone offsets cannot be resolved
   to hertz.
7. **One profile per utterance.** Input is not split into sentences, so a mixed
   passage receives a single emotional reading.
8. **`core/pipeline.py` and `core/router.py` have no consumers.** The domain
   layers are reached by direct import, not through the pipeline abstraction.

### To reach audible speech

| Priority | Work | Rough size |
| --- | --- | --- |
| 1 | Bridge to synthesis — speaker F0 baseline, voiced/unvoiced mask, phoneme id vocabulary | medium |
| 2 | Number / date / abbreviation verbalization ahead of G2P | small |
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
