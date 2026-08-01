"""Exact output schema validation and CSV writer."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

from .types import ACTIONS, MESSAGE_TYPES, OUTPUT_COLUMNS, Prediction


class OutputValidationError(ValueError):
    """Raised when predictions violate the challenge output contract."""


def format_confidence(value: float) -> str:
    clipped = max(0.0, min(1.0, float(value)))
    text = f"{clipped:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def format_evidence(evidence_ids: Sequence[str] | str | None) -> str:
    if evidence_ids is None:
        return "none"
    if isinstance(evidence_ids, str):
        text = evidence_ids.strip()
        return text if text else "none"
    cleaned = [item.strip() for item in evidence_ids if item and item.strip()]
    return ";".join(cleaned) if cleaned else "none"


def validate_prediction(
    prediction: Prediction,
    *,
    allowed_message_ids: set[str] | None = None,
    history_ids: set[str] | None = None,
) -> list[str]:
    """Return a list of validation problems for one prediction."""
    problems: list[str] = []
    if not prediction.message_id:
        problems.append("missing message_id")
    elif allowed_message_ids is not None and prediction.message_id not in allowed_message_ids:
        problems.append(f"unknown message_id {prediction.message_id}")

    if prediction.action not in ACTIONS:
        problems.append(f"invalid action {prediction.action!r}")
    if prediction.message_type not in MESSAGE_TYPES:
        problems.append(f"invalid message_type {prediction.message_type!r}")

    reason = (prediction.reason or "").strip()
    if not reason:
        problems.append("empty reason")
    elif len(reason) > 280:
        problems.append("reason exceeds 280 characters")

    try:
        confidence = float(prediction.confidence)
    except (TypeError, ValueError):
        problems.append(f"non-numeric confidence {prediction.confidence!r}")
        confidence = None
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        problems.append(f"confidence out of range: {confidence}")

    evidence = format_evidence(prediction.evidence_message_ids)
    if evidence != "none":
        parts = [part for part in evidence.split(";") if part]
        if not parts:
            problems.append("empty evidence_message_ids")
        if history_ids is not None:
            for part in parts:
                if part not in history_ids:
                    problems.append(f"evidence id not in history: {part}")
    return problems


def validate_predictions(
    predictions: Sequence[Prediction],
    *,
    required_message_ids: Sequence[str],
    history_ids: set[str] | None = None,
) -> list[str]:
    problems: list[str] = []
    required = list(required_message_ids)
    required_set = set(required)
    seen: set[str] = set()

    if len(predictions) != len(required):
        problems.append(
            f"expected {len(required)} predictions, got {len(predictions)}"
        )

    for prediction in predictions:
        if prediction.message_id in seen:
            problems.append(f"duplicate message_id {prediction.message_id}")
        seen.add(prediction.message_id)
        problems.extend(
            f"{prediction.message_id}: {issue}"
            for issue in validate_prediction(
                prediction,
                allowed_message_ids=required_set,
                history_ids=history_ids,
            )
        )

    missing = [message_id for message_id in required if message_id not in seen]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        problems.append(f"missing predictions for: {preview}{suffix}")
    return problems


def predictions_to_rows(predictions: Iterable[Prediction]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for prediction in predictions:
        rows.append(
            {
                "message_id": prediction.message_id,
                "action": prediction.action,
                "message_type": prediction.message_type,
                "reason": prediction.reason.strip(),
                "confidence": format_confidence(prediction.confidence),
                "evidence_message_ids": format_evidence(prediction.evidence_message_ids),
            }
        )
    return rows


def write_output_csv(
    path: Path | str,
    predictions: Sequence[Prediction],
    *,
    required_message_ids: Sequence[str],
    history_ids: set[str] | None = None,
) -> Path:
    """Validate then write predictions with the exact required column order."""
    problems = validate_predictions(
        predictions,
        required_message_ids=required_message_ids,
        history_ids=history_ids,
    )
    if problems:
        joined = "\n".join(f"- {item}" for item in problems[:20])
        extra = "" if len(problems) <= 20 else f"\n- ... {len(problems) - 20} more"
        raise OutputValidationError(
            f"Output validation failed with {len(problems)} issue(s):\n{joined}{extra}"
        )

    # Preserve input message order.
    by_id = {prediction.message_id: prediction for prediction in predictions}
    ordered = [by_id[message_id] for message_id in required_message_ids]
    rows = predictions_to_rows(ordered)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def read_output_csv(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(OUTPUT_COLUMNS):
            raise OutputValidationError(
                f"Unexpected columns: {reader.fieldnames!r}; expected {list(OUTPUT_COLUMNS)!r}"
            )
        return list(reader)
