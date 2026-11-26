# Clone your voice for Ava

There are four steps. Steps 1, 2 and 4 run on your computer in a few minutes.
Step 3 trains the model and needs a GPU, so use a free Google Colab GPU or any
CUDA machine. An Apple-Silicon Mac also works with PyTorch's MPS backend, but
much more slowly.

> Clone only your own voice, or the voice of someone who records their own
> consent, and use it only for that person. The recorder stores a spoken
> consent statement, and the dataset builder refuses to work without one.

## 1. Record (≈ 15 minutes)

```bash
pip install -e ".[neural]"
python -m ava_voice.clone.record --voice ali      # open http://localhost:8765
```

Read the consent sentence, then the ~140 sentences: everyday talk, questions,
emotions, numbers and some formal reading. Use a quiet room, stay at a steady
distance from the microphone, and talk the way you normally talk. That is the
voice you will get. Takes that are clipped, too quiet or noisy are flagged at
once. Re-record them.

More audio gives a better clone. About 10 minutes is the minimum for a
convincing result. With 30–60 minutes, even unusual words sound like you.
Add your own sentences to `clone/data/script_fa.txt` if you want more.

## 2. Build the dataset

```bash
python -m ava_voice.clone.dataset --voice ali --whisper-dir ~/models/sherpa-onnx-whisper-turbo
```

This writes `~/ava-voice-recordings/ali/train/` with `wavs/` (22.05 kHz,
trimmed, normalized), `metadata.csv` (`file|IPA` in Ava's phonemes) and
`report.json`. Whisper rejects takes where you misread the sentence.

## 3. Fine-tune Piper (GPU)

Start from the Persian Piper checkpoint `fa_IR/amir/medium`. Fine-tuning
needs far less audio than training from scratch. In a Colab notebook with a GPU
runtime:

```bash
# — setup (once) —
git clone https://github.com/OHF-Voice/piper1-gpl.git && cd piper1-gpl
pip install -e '.[train]' && ./build_monotonic_align.sh
wget -O base.ckpt "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/fa/fa_IR/amir/medium/epoch=2199-step=897600.ckpt"

# — upload ~/ava-voice-recordings/ali/train/ to /content/train first —
python3 -m piper.train fit \
  --data.voice_name ali \
  --data.csv_path /content/train/metadata.csv \
  --data.audio_dir /content/train/wavs \
  --data.phoneme_type text \
  --model.sample_rate 22050 \
  --data.espeak_voice fa \
  --data.cache_dir /content/cache \
  --data.config_path /content/ali.json \
  --data.batch_size 16 \
  --trainer.max_epochs 2000 \
  --ckpt_path base.ckpt

python3 -m piper.train.export_onnx --checkpoint <lightning_logs/…/checkpoints/last.ckpt> --output-file /content/ali.onnx
```

`--data.phoneme_type text` makes Piper train on the IPA in `metadata.csv`.
These are the same phonemes Ava produces when it speaks, so your voice
inherits Ava's pronunciation fixes. To use espeak-ng instead, point
`--data.csv_path` at `metadata_text.csv` and drop `--data.phoneme_type`.

Piper's guide recommends about 2,000 epochs for a small fine-tune. On 15
minutes of audio that takes roughly 1–2 hours on a Colab T4. Listen to a
checkpoint along the way:
`python3 -m piper.train.infer_torch --checkpoint … --config /content/ali.json`.

## 4. Install into Ava

```bash
python -m ava_voice.clone.install --onnx ali.onnx --config ali.json --name ali
# → ~/.cache/ava-voice/voice-ali/ (+ test.wav)
```

Then use the folder in Ava (`AVA_TTS_VOICE_DIR=~/.cache/ava-voice/voice-ali`)
or in PVAP (`[voice] voice_dir`). For your own voice, set `pitch_semitones = 0`
and `speed = 1.0` to keep its natural pitch and pace. Emotion, breathing and
the learned lexicon apply exactly as with the stock voices.
