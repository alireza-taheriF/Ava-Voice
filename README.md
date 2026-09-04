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
| `linguistics/` | ~1.6k | Persian normalizer, token graph, grapheme-to-phoneme, stress, punctuation, homographs |
| `emotion/` | ~2.0k | Lexicon classifier, valence/arousal/dominance mapping, intensity, context, short-term memory |
| `prosody/` | ~1.7k | Emotion → pitch, tempo, energy, pause, emphasis, contour shape |
| `acoustics/` | ~2.0k | Phoneme alignment, per-phoneme durations, F0/energy curves, pause shaping, elongation |
| `core/` | ~1.0k | Settings (pydantic-settings), logging, cache (Redis + in-memory), sessions, pipeline/router primitives |
| `main.py` | 141 | FastAPI app: `/health` and a WebSocket transport endpoint |
| `tests/` | ~1.2k | Deterministic unit tests across all layers |
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
is unset (the default) auth is disabled — convenient locally, **never do this on
a reachable host**. `/health` is always public so container probes work.

```bash
AVA_API_KEY=secret uvicorn ava_voice.main:app
curl -H "X-API-Key: secret" localhost:8000/api/v1/analyze -d '{"text":"سلام"}'
```

## Test

```bash
pytest                 # config lives in pyproject.toml
pytest --cov=ava_voice # with coverage
```

Without an editable install, set the path explicitly:

```bash
cd .. && PYTHONPATH="$PWD" python -m pytest ava_voice/tests -q
```

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
- **API** — `/health`, an authenticated WebSocket transport with enforced frame
  size and connection limits.

**What does not work — and why**

1. **No audio output.** There is no vocoder, no acoustic model, no PCM path.
   `torch` is not a dependency (it was declared but imported nowhere). The
   `AcousticProfile` is a JSON control plan, not a waveform.
2. **Grapheme-to-phoneme is underspecified.** Persian omits short vowels in
   writing. The phonemizer emits a `V` placeholder where it cannot decide, and
   nothing resolves it: on common vocabulary roughly 3 in 4 words contain at
   least one `V` (`مدرسه` → `mVdVrVsVh`, not `madrese`). The pronunciation
   lexicon holds ~10 entries. **This is the main blocker.**
3. **Ezafe is not implemented.** No linking `-e-` is inserted between a noun and
   its modifier, so `کتاب علی` is not pronounced `ketâb-e ali`.
4. **No number verbalization.** Digits reach the phoneme stream as literal
   characters; `۱۴۰۳` is never spoken as «هزار و چهارصد و سه».
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
| 1 | Resolve the `V` placeholder — pronunciation lexicon or a seq2seq G2P model | large |
| 2 | Bridge to synthesis — speaker F0 baseline, voiced/unvoiced mask, phoneme id vocabulary | medium |
| 3 | Ezafe insertion | medium |
| 4 | Number / date / abbreviation verbalization | small |
| 5 | Sentence segmentation with per-sentence prosody | small |

Two viable routes for (2): drive an existing Persian TTS engine with the
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
