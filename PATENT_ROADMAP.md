# Ava Voice — road to a real, specific patent

This is the working plan from the current symbolic pipeline to a **specific,
embodied, evidence-backed** invention. The idea “emotion-aware Persian TTS” is
not patentable. The method below is: a vocoder-agnostic compiler that treats
**ezafe, written drawl, and formality-as-register** as first-class controls of
duration, F0 and energy.

Do not file a complete application until Phase 4 has audio evidence. If the
repository is public, talk to counsel *before* disclosing further algorithm
detail — public disclosure can destroy novelty in many jurisdictions.

---

## What is being invented

**Persian Affective Linker + Written-Prosody Compiler (PAL-WPC)**

A method that:

1. Detects the Persian ezafe linker as a phoneme *distinct* from a lexical
   final `/e/` (and chooses the `e` / `ye` allomorph).
2. Modulates that linker’s duration, local F0 and local energy as a closed-form
   function of **formality × valence × intensity × sadness** — not as a normal
   vowel and not as a global “emotion style”.
3. Compiles authorial writing cues (letter repetition, emphatic punctuation,
   colloquial forms) into the same control plan, gated so ordinary long vowels
   are not drawn out.
4. Emits a serialisable, engine-agnostic `AcousticProfile` (ms durations,
   semitone and hertz F0, energy, ezafe segments) that can drive any vocoder.

Phase 0 (this change) implements (1)–(4) symbolically. Later phases add
calibration, audio, and comparative measurements.

---

## Phase 0 — Close the symbolic loop  ← **this change**

**Goal.** The claimed controls exist in code and tests, without a third-party
model and without copyrighted media.

| Work | Status |
| --- | --- |
| Explicit-kasra ezafe on the legacy G2P (`کتابِ` → linker `/e/` or `/ye/`) | implemented |
| `EzafeModulator`: register-conditioned duration / F0 / energy | implemented |
| Ezafe segments on `AcousticProfile` (auditable) | implemented |
| Speaker F0 baseline; semitone → hertz curve | implemented |
| Sentence segmentation + per-sentence emotion memory | implemented |
| End-to-end `plan_speech()` | implemented |
| Legal corpus ingest (CC0 Common Voice sentences + owned seed) | implemented |

**Exit.** Tests prove: formal ezafe is shorter/flatter than warm ezafe; lexical
`خانه` `/e/` is not treated as ezafe; `ye` allomorph is classified; multi-sentence
plans carry memory; Hz curve matches the semitone curve.

**First corpus measurement** (see `reports/phase0_corpus_summary.json`): 25
owned sentences + 4000 CC0 Common Voice `fa` prompts. Explicit-kasra ezafe
fired 18 times — 17 on the seed, **1 on Common Voice**. Speakers almost never
write kasra. Phase 1 must infer the linker from the *next word*, not only from
a diacritic. Commercial podcasts/YouTube were not used.

---

## Phase 1 — Legal speech data, not pirated media

**Need data? Yes — but only licensed or self-recorded audio.**

Scraping commercial Persian podcasts or YouTube videos to train or calibrate
this system is copyright infringement and would taint a patent file. The
collector **refuses** unsigned sources.

| Allowed now | Not allowed |
| --- | --- |
| Mozilla Common Voice `fa` (CC0) | Commercial podcasts |
| Sentences we wrote (seed) | YouTube / Aparat rips |
| User-supplied files with an SPDX licence in the manifest | “everyone uses it” dumps |
| Self-recorded studio / phone takes | Scraped radio |

**Work.**

1. Ingest Common Voice Persian *transcripts* (done in Phase 0).
2. Download a **validated audio subset** of Common Voice `fa` (CC0 clips) into
   `data/audio/` (gitignored) and pair them with transcripts.
3. Record a **gold set** of 200 utterances we own: ezafe / no-ezafe minimal
   pairs, formal vs colloquial, written drawl, mixed-emotion paragraphs.
4. Optional later: Wikimedia Commons / explicitly CC podcast episodes listed in
   a signed manifest — never a crawler.

**Exit.** ≥ 1 000 licensed transcribed clips on disk + a 200-utt owned gold set
with ezafe/register labels.

---

## Phase 2 — Measure the invention on real speech

**Goal.** Replace hand constants with values fitted on Persian audio.

1. Forced-align gold + Common Voice clips (MFA or a Persian aligner) to
   phoneme intervals.
2. Extract F0 (Hz) and RMS energy on a 10 ms grid.
3. Label ezafe vs lexical `/e/` on the gold set (human, then automated).
4. Fit the modulator’s duration / F0 / energy coefficients; freeze a
   versioned `calibration.json`.
5. Report effect sizes: formal vs warm linker duration, F0 delta, confusion
   with lexical `/e/`.

**Exit.** A calibration file and a short measurement report. This is the first
piece of **technical effect** a patent office can read.

---

## Phase 3 — Audible embodiment

**Goal.** The method produces sound, not only JSON.

Two tracks, in order:

1. **Fast track.** Map `AcousticProfile` (durations, pauses, ezafe, F0 in Hz)
   onto SSML / API controls of an existing Persian TTS engine. Enough to hear
   the linker contrast.
2. **Own track.** Train or adapt a duration+F0-conditioned acoustic model
   (FastSpeech2-class) and a vocoder, using the profile as the conditioning
   plan — not as a lost side channel.

Also in this phase: number/date verbaliser *before* G2P; broader emotion
lexicon (morphology); speaker-id F0 baselines.

**Exit.** A/B audio: same sentence, formal vs warm ezafe, plus no-ezafe
control. Listeners (or a MOS-style test) can hear the difference.

---

## Phase 4 — Evidence package for filing

**Goal.** A complete, narrow specification — not a slogan.

1. One independent claim: the ezafe-as-linker modulation method.
2. Dependent claims: `e`/`ye` allomorph; formality as an axis orthogonal to
   categorical emotion; written-drawl compiler; Hz-resolved speaker baseline;
   engine-agnostic IR.
3. Figures: token graph → vector → profile → ezafe segment → waveform.
4. Comparative table vs SSML-only TTS, vs global style tokens, vs treating
   ezafe as lexical `/e/`.
5. Do **not** claim Homo-GE2PE, VAD psychology, or “emotional TTS” as such.

**Exit.** Counsel reviews; file provisional / first filing **before** any
further public write-up of the fitted coefficients.

---

## Phase 5 — After filing (product, not patent)

IR adapters to a second engine; realtime streaming of the plan; optional
voice clone **conditioned on the plan**. These strengthen the product. They
are not required for the first filing.

---

## Honest scope

Phase 0 does **not** make the patent “ready”. It makes the invention *real in
software* and starts a **legal** corpus. Phases 2–4 are what make it
defensible.
