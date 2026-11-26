"""Text-to-Speech synthesis.

* ``vits`` — **the voice**: Ava's Persian frontend (normalization, number
  verbalization) → Homo-GE2PE G2P in NumPy → a Persian VITS voice on
  onnxruntime. Download models with ``python -m ava_voice.tts.models fetch``.
* ``formant_vocoder`` — dependency-free rule-based fallback that renders a
  symbolic speech plan directly. Robotic; kept for environments without models
  and as a reference renderer for the plan's timing and F0.
"""

from __future__ import annotations

__all__: list[str] = ["vits", "models", "formant_vocoder"]
