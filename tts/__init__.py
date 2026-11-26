"""Text-to-Speech synthesis.

``formant_vocoder`` is the first working path from a speech plan to PCM: a
rule-based source–filter baseline (robotic, intelligibility unmeasured) that
will be replaced by a trained acoustic model + neural vocoder behind the same
``synthesize_profile`` / ``synthesize_plan`` signatures.
"""

from __future__ import annotations

__all__: list[str] = ["formant_vocoder"]
