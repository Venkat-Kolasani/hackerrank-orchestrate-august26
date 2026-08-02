#!/usr/bin/env python3
"""Sample-set evaluation CLI (labels used only as gold, never as features).

Production ``code/main.py`` must not load ``sample_messages.csv``. This
evaluator may, but it strips label columns before routing and compares
predictions to held-out gold afterward.

Diagnostics are written under ``code/evaluation/diagnostics/`` (outside
``dataset/``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _CODE_ROOT.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from evaluation.metrics import PairResult, summarize_pairs
from router.baseline import route_dataset
from router.data import load_dataset
from router.media import build_media_interpreter
from router.output import validate_prediction
from router.types import ACTIONS, MESSAGE_TYPES, Prediction

INPUT_COLUMNS = (
    "message_id",
    "user_id",
    "conversation_type",
    "group_id",
    "business_id",
    "sender_user_id",
    "created_at",
    "message_text",
    "media_type",
    "media_id",
    "forwarded_count",
)
LABEL_COLUMNS = (
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
)


def _read_sample_gold(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split sample CSV into feature rows (no labels) and gold label rows."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    features: list[dict[str, str]] = []
    gold: list[dict[str, str]] = []
    for row in rows:
        features.append({col: row.get(col, "") for col in INPUT_COLUMNS})
        gold.append(
            {
                "message_id": row["message_id"],
                **{col: row.get(col, "") for col in LABEL_COLUMNS},
            }
        )
    return features, gold


def _write_feature_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def evaluate(
    *,
    dataset_root: Path,
    sample_file: Path,
    diagnostics_dir: Path,
    provider_media: str = "offline",
    provider_decision: str = "offline",
) -> dict:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    feature_rows, gold_rows = _read_sample_gold(sample_file)
    # Materialize unlabeled inputs outside dataset/ so the router never sees labels.
    unlabeled_path = diagnostics_dir / "sample_inputs_unlabeled.csv"
    _write_feature_csv(unlabeled_path, feature_rows)

    # Copy unlabeled file into a temp messages slot under diagnostics and point
    # load_dataset at a shim directory that reuses dataset context CSVs via
    # the real dataset root for joins, but messages from unlabeled file.
    # Simplest: load full dataset context, then replace messages by loading
    # unlabeled CSV through the same loader API from diagnostics as root is
    # wrong (missing users.csv). Instead symlink/copy approach:
    # load_dataset(dataset_root, messages_file=...) cannot read outside root.
    # So write unlabeled messages into diagnostics and pass messages via a
    # thin wrapper: load context from dataset_root, then overwrite messages.

    dataset = load_dataset(dataset_root, messages_file="messages.csv")
    # Replace production messages with unlabeled sample feature rows so label
    # columns never enter the router feature path.
    from router import data as data_mod

    dataset.messages = [data_mod._message_from_row(row) for row in feature_rows]

    import os

    os.environ["ROUTER_MEDIA_PROVIDER"] = provider_media
    os.environ["ROUTER_DECISION_PROVIDER"] = provider_decision
    interpreter = build_media_interpreter(provider=provider_media, cache=True)
    predictions = route_dataset(dataset, interpreter=interpreter)

    gold_by_id = {row["message_id"]: row for row in gold_rows}
    history_ids = {item.message_id for item in dataset.message_history}
    pairs: list[PairResult] = []
    pred_rows: list[dict[str, str]] = []

    for pred in predictions:
        gold = gold_by_id[pred.message_id]
        invalid = validate_prediction(
            pred,
            allowed_message_ids={pred.message_id},
            history_ids=history_ids,
        )
        pairs.append(
            PairResult(
                message_id=pred.message_id,
                gold_action=gold["action"],
                pred_action=pred.action,
                gold_type=gold["message_type"],
                pred_type=pred.message_type,
                gold_evidence=gold["evidence_message_ids"],
                pred_evidence=pred.evidence_message_ids,
                gold_confidence=_safe_float(gold["confidence"]),
                pred_confidence=float(pred.confidence),
                reason=pred.reason,
                invalid=invalid,
            )
        )
        pred_rows.append(
            {
                "message_id": pred.message_id,
                "pred_action": pred.action,
                "gold_action": gold["action"],
                "pred_message_type": pred.message_type,
                "gold_message_type": gold["message_type"],
                "pred_confidence": f"{pred.confidence:.4f}",
                "gold_confidence": gold["confidence"],
                "pred_evidence": pred.evidence_message_ids,
                "gold_evidence": gold["evidence_message_ids"],
                "reason": pred.reason,
                "action_match": str(pred.action == gold["action"]),
                "type_match": str(pred.message_type == gold["message_type"]),
            }
        )

    type_labels = sorted({p.gold_type for p in pairs} | {p.pred_type for p in pairs} | set(MESSAGE_TYPES))
    summary = summarize_pairs(pairs, type_labels)
    summary["sample_file"] = str(sample_file)
    summary["unlabeled_inputs"] = str(unlabeled_path)
    summary["providers"] = {
        "media": provider_media,
        "decision": provider_decision,
    }

    # Persist artifacts outside dataset/
    preds_path = diagnostics_dir / "sample_predictions.csv"
    with preds_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pred_rows[0].keys()) if pred_rows else ["message_id"])
        writer.writeheader()
        writer.writerows(pred_rows)

    errors = [row for row in pred_rows if row["action_match"] != "True" or row["type_match"] != "True"]
    errors_path = diagnostics_dir / "sample_errors.csv"
    with errors_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(errors[0].keys()) if errors else ["message_id"],
        )
        writer.writeheader()
        writer.writerows(errors)

    summary_path = diagnostics_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    summary["artifacts"] = {
        "predictions": str(preds_path),
        "errors": str(errors_path),
        "summary": str(summary_path),
        "unlabeled_inputs": str(unlabeled_path),
    }
    return summary


def _print_summary(summary: dict) -> None:
    print(f"n={summary['n']}")
    print(
        f"action_acc={summary['action_accuracy']:.4f}  "
        f"type_acc={summary['type_accuracy']:.4f}  "
        f"action_macro_f1={summary['action_f1']['macro']:.4f}  "
        f"type_macro_f1={summary['type_f1']['macro']:.4f}"
    )
    print(
        f"evidence exact={summary['evidence_exact']}/{summary['n']} "
        f"({summary['evidence_exact_rate']:.4f})  "
        f"partial={summary['evidence_partial']}/{summary['n']} "
        f"({summary['evidence_partial_rate']:.4f})  "
        f"mean_jaccard={summary['evidence_mean_jaccard']:.4f}"
    )
    print(f"invalid_outputs={summary['invalid_output_count']}")
    print("action_confusion (rows=gold, cols=pred):")
    labels = list(ACTIONS)
    header = "gold\\pred".ljust(12) + "".join(lab.rjust(10) for lab in labels)
    print(header)
    for row in labels:
        cells = "".join(str(summary["action_confusion"][row].get(col, 0)).rjust(10) for col in labels)
        print(row.ljust(12) + cells)
    print("action_error_patterns:", summary["action_error_patterns"])
    print("type_error_patterns:", summary["type_error_patterns"])
    print("confidence_buckets:")
    for bucket in summary["confidence_buckets"]:
        print(
            f"  {bucket['bucket']}: n={bucket['count']} "
            f"acc={bucket['action_accuracy']:.4f} "
            f"mean_conf={bucket['mean_confidence']:.4f}"
        )
    print("artifacts:", summary.get("artifacts"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate router on sample_messages.csv")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_REPO_ROOT / "dataset",
        help="Dataset directory with context CSVs",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=None,
        help="Labeled sample CSV (default: <dataset>/sample_messages.csv)",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=_CODE_ROOT / "evaluation" / "diagnostics",
        help="Output directory outside dataset/",
    )
    parser.add_argument("--media-provider", default="offline")
    parser.add_argument("--decision-provider", default="offline")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = args.dataset.resolve()
    sample = (args.sample or (dataset_root / "sample_messages.csv")).resolve()
    diagnostics = args.diagnostics.resolve()
    if "dataset" in diagnostics.parts and diagnostics.parts[-2:] == ("dataset", "diagnostics"):
        print("Refusing to write diagnostics inside dataset/", file=sys.stderr)
        return 2
    if not sample.is_file():
        print(f"Sample file missing: {sample}", file=sys.stderr)
        return 2

    summary = evaluate(
        dataset_root=dataset_root,
        sample_file=sample,
        diagnostics_dir=diagnostics,
        provider_media=args.media_provider,
        provider_decision=args.decision_provider,
    )
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
