#!/usr/bin/env python3
"""Message Notification Router — production entry point.

Reads participant-facing files under ``dataset/`` and writes ``output.csv``.
Never loads ``sample_messages.csv``; that file is evaluation-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_code_on_path() -> Path:
    code_root = Path(__file__).resolve().parent
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    return code_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Route WhatsApp messages into notify/digest/mute decisions."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to the dataset directory (default: <repo>/dataset)",
    )
    parser.add_argument(
        "--messages",
        default="messages.csv",
        help="Message CSV basename inside the dataset directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <dataset>/output.csv)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_code_on_path()
    from router.baseline import route_dataset
    from router.data import load_dataset
    from router.output import write_output_csv

    args = build_parser().parse_args(argv)
    code_root = Path(__file__).resolve().parent
    repo_root = code_root.parent
    dataset_root = (args.dataset or (repo_root / "dataset")).resolve()
    output_path = (args.output or (dataset_root / "output.csv")).resolve()

    if args.messages == "sample_messages.csv":
        print(
            "Refusing to use sample_messages.csv in the production entry point. "
            "Run code/evaluation/main.py for sample evaluation.",
            file=sys.stderr,
        )
        return 2

    dataset = load_dataset(dataset_root, messages_file=args.messages)
    for warning in dataset.media_warnings:
        print(f"warning: {warning}", file=sys.stderr)

    from router.media import build_media_interpreter

    interpreter = build_media_interpreter()
    predictions = route_dataset(dataset, interpreter=interpreter)
    history_ids = {item.message_id for item in dataset.message_history}
    write_output_csv(
        output_path,
        predictions,
        required_message_ids=[message.message_id for message in dataset.messages],
        history_ids=history_ids,
    )

    from collections import Counter

    actions = Counter(prediction.action for prediction in predictions)
    types = Counter(prediction.message_type for prediction in predictions)
    print(f"Wrote {len(predictions)} predictions to {output_path}")
    print(f"actions={dict(actions)}")
    print(f"types={dict(types)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
