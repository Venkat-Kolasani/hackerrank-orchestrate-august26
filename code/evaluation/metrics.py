"""Evaluation metrics for sample-set diagnostics.

Used only by ``code/evaluation/`` — never imported by the production router.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


ACTIONS = ("notify", "digest", "mute")


@dataclass
class PairResult:
    message_id: str
    gold_action: str
    pred_action: str
    gold_type: str
    pred_type: str
    gold_evidence: str
    pred_evidence: str
    gold_confidence: float
    pred_confidence: float
    reason: str
    invalid: list[str] = field(default_factory=list)


def _parse_evidence(value: str) -> set[str]:
    text = (value or "").strip()
    if not text or text == "none":
        return set()
    return {part.strip() for part in text.split(";") if part.strip()}


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict[str, float]:
    """Per-label and macro F1."""
    per: dict[str, float] = {}
    for label in labels:
        tp = fp = fn = 0
        for truth, pred in zip(y_true, y_pred, strict=True):
            if pred == label and truth == label:
                tp += 1
            elif pred == label and truth != label:
                fp += 1
            elif truth == label and pred != label:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per[label] = round(f1, 4)
    macro = round(sum(per.values()) / len(labels), 4) if labels else 0.0
    return {"macro": macro, **per}


def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> dict[str, dict[str, int]]:
    matrix = {row: {col: 0 for col in labels} for row in labels}
    for truth, pred in zip(y_true, y_pred, strict=True):
        t = truth if truth in matrix else None
        p = pred if pred in matrix.get(truth, {}) else None
        if t is None:
            continue
        if p is None:
            # Map unknown pred into a bucket if needed
            if pred not in matrix[t]:
                matrix[t][pred] = 0
            matrix[t][pred] += 1
        else:
            matrix[t][p] += 1
    return matrix


def evidence_overlap(gold: str, pred: str) -> tuple[bool, bool, float]:
    """Return (exact_match, partial_match, jaccard)."""
    g = _parse_evidence(gold)
    p = _parse_evidence(pred)
    if not g and not p:
        return True, True, 1.0
    if not g or not p:
        return False, False, 0.0
    exact = g == p
    partial = bool(g & p)
    jaccard = len(g & p) / len(g | p)
    return exact, partial, jaccard


def confidence_buckets(
    pairs: Sequence[PairResult],
    *,
    edges: Sequence[float] = (0.0, 0.5, 0.7, 0.85, 1.01),
) -> list[dict[str, float | int | str]]:
    """Calibration-style buckets: predicted confidence vs action accuracy."""
    rows: list[dict[str, float | int | str]] = []
    for lo, hi in zip(edges, edges[1:]):
        members = [p for p in pairs if lo <= p.pred_confidence < hi]
        if not members:
            rows.append(
                {
                    "bucket": f"[{lo:.2f},{hi:.2f})",
                    "count": 0,
                    "action_accuracy": 0.0,
                    "mean_confidence": 0.0,
                }
            )
            continue
        acc = sum(1 for p in members if p.pred_action == p.gold_action) / len(members)
        mean_c = sum(p.pred_confidence for p in members) / len(members)
        rows.append(
            {
                "bucket": f"[{lo:.2f},{hi:.2f})",
                "count": len(members),
                "action_accuracy": round(acc, 4),
                "mean_confidence": round(mean_c, 4),
            }
        )
    return rows


def summarize_pairs(
    pairs: Sequence[PairResult],
    type_labels: Sequence[str],
) -> dict:
    gold_actions = [p.gold_action for p in pairs]
    pred_actions = [p.pred_action for p in pairs]
    gold_types = [p.gold_type for p in pairs]
    pred_types = [p.pred_type for p in pairs]

    exact = partial = 0
    jaccards: list[float] = []
    for pair in pairs:
        ex, pa, jac = evidence_overlap(pair.gold_evidence, pair.pred_evidence)
        exact += int(ex)
        partial += int(pa)
        jaccards.append(jac)

    invalid_count = sum(1 for p in pairs if p.invalid)
    action_acc = (
        sum(1 for p in pairs if p.pred_action == p.gold_action) / len(pairs)
        if pairs
        else 0.0
    )
    type_acc = (
        sum(1 for p in pairs if p.pred_type == p.gold_type) / len(pairs) if pairs else 0.0
    )

    # Aggregate confusion patterns for diagnostics.
    action_errors = Counter(
        f"{p.gold_action}->{p.pred_action}"
        for p in pairs
        if p.gold_action != p.pred_action
    )
    type_errors = Counter(
        f"{p.gold_type}->{p.pred_type}"
        for p in pairs
        if p.gold_type != p.pred_type
    )

    return {
        "n": len(pairs),
        "action_accuracy": round(action_acc, 4),
        "type_accuracy": round(type_acc, 4),
        "action_f1": macro_f1(gold_actions, pred_actions, ACTIONS),
        "type_f1": macro_f1(gold_types, pred_types, list(type_labels)),
        "action_confusion": confusion_matrix(gold_actions, pred_actions, ACTIONS),
        "type_confusion": confusion_matrix(gold_types, pred_types, list(type_labels)),
        "evidence_exact": exact,
        "evidence_partial": partial,
        "evidence_exact_rate": round(exact / len(pairs), 4) if pairs else 0.0,
        "evidence_partial_rate": round(partial / len(pairs), 4) if pairs else 0.0,
        "evidence_mean_jaccard": round(sum(jaccards) / len(jaccards), 4) if jaccards else 0.0,
        "confidence_buckets": confidence_buckets(pairs),
        "invalid_output_count": invalid_count,
        "action_error_patterns": dict(action_errors.most_common()),
        "type_error_patterns": dict(type_errors.most_common(15)),
    }
