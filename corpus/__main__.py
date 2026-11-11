"""``python -m ava_voice.corpus`` — ingest, annotate, calibrate."""

from __future__ import annotations

import argparse

from ava_voice.corpus.annotate import annotate_corpus
from ava_voice.corpus.calibrate import write_calibration
from ava_voice.corpus.ingest import ingest


def main() -> None:
    """CLI entry: build the licensed corpus and run the symbolic pass."""
    parser = argparse.ArgumentParser(
        description="Ingest licensed Persian text and annotate it with Ava Voice."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip the Common Voice fetch; write the owned seed only.",
    )
    parser.add_argument(
        "--max-remote",
        type=int,
        default=4000,
        help="Maximum Common Voice sentences to keep (default 4000).",
    )
    parser.add_argument(
        "--limit-annotate",
        type=int,
        default=None,
        help="Annotate at most N rows (default: all).",
    )
    args = parser.parse_args()
    ingest(include_common_voice=not args.offline, max_remote=args.max_remote)
    if args.limit_annotate is not None:
        annotate_corpus(limit=args.limit_annotate)
        write_calibration(annotate=False)
    else:
        write_calibration(annotate=True)


if __name__ == "__main__":
    main()
