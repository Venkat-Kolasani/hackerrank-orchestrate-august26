"""Ground TECHNICAL_RISK_FLOOR in message_history + message_events.

Computes assess_risk for every historical message with domain_mismatch,
user_reports_30d, and forwarded_count wired in, then finds the confirmed_risk
threshold that best separates muted/reported outcomes from everything else.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _CODE_ROOT.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from router.safety import assess_risk
from router.types import ContentSummary


@dataclass(frozen=True)
class ScoredMessage:
    message_id: str
    user_id: str
    confirmed_risk: float
    negative_outcome: bool  # muted_after_message or message_reported
    muted: bool
    reported: bool
    domain_mismatch: bool
    user_reports_30d: int
    forwarded_count: int
    base_score: float
    injection_score: float
    metadata_score: float


def _load_businesses(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["business_id"]: row for row in csv.DictReader(fh)}


def _domain_mismatch(biz: dict[str, str] | None) -> bool:
    if not biz:
        return False
    official = (biz.get("official_domain") or "").strip().lower()
    used = (biz.get("domain_used_by_sender") or "").strip().lower()
    if not official or not used:
        return False
    return official != used


def score_history(dataset_dir: Path) -> list[ScoredMessage]:
    businesses = _load_businesses(dataset_dir / "business_accounts.csv")
    with (dataset_dir / "message_events.csv").open(newline="", encoding="utf-8") as fh:
        events = {
            (row["user_id"], row["message_id"]): row for row in csv.DictReader(fh)
        }

    scored: list[ScoredMessage] = []
    with (dataset_dir / "message_history.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["user_id"], row["message_id"])
            event = events.get(key)
            if event is None:
                continue

            biz_id = (row.get("business_id") or "").strip()
            biz = businesses.get(biz_id) if biz_id else None
            mismatch = _domain_mismatch(biz)
            reports = int(biz["user_reports_30d"]) if biz else 0
            forwards = int(row.get("forwarded_count") or 0)

            assessment = assess_risk(
                ContentSummary(message_text=row.get("message_text") or ""),
                domain_mismatch=mismatch,
                user_reports_30d=reports,
                forwarded_count=forwards,
            )
            muted = event["muted_after_message"] == "1"
            reported = event["message_reported"] == "1"
            scored.append(
                ScoredMessage(
                    message_id=row["message_id"],
                    user_id=row["user_id"],
                    confirmed_risk=assessment.risk_score,
                    negative_outcome=muted or reported,
                    muted=muted,
                    reported=reported,
                    domain_mismatch=mismatch,
                    user_reports_30d=reports,
                    forwarded_count=forwards,
                    base_score=assessment.base_score,
                    injection_score=assessment.injection_score,
                    metadata_score=assessment.metadata_score,
                )
            )
    return scored


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def summarize(label: str, values: list[float]) -> None:
    vals = sorted(values)
    print(f"{label}  n={len(vals)}")
    if not vals:
        return
    buckets = Counter(round(v, 2) for v in vals)
    print(
        "  min={:.2f} p25={:.2f} p50={:.2f} p75={:.2f} p90={:.2f} max={:.2f} mean={:.3f}".format(
            vals[0],
            _percentile(vals, 0.25),
            _percentile(vals, 0.50),
            _percentile(vals, 0.75),
            _percentile(vals, 0.90),
            vals[-1],
            sum(vals) / len(vals),
        )
    )
    print("  value histogram (rounded to 0.05):")
    hist = Counter(round(v * 20) / 20 for v in vals)
    for edge in sorted(hist):
        bar = "#" * hist[edge]
        print(f"    {edge:>4.2f}: {hist[edge]:>4} {bar}")


def best_threshold(scored: list[ScoredMessage]) -> tuple[float, dict[str, float]]:
    """Maximize Youden's J = TPR - FPR; break ties by higher F1 then lower threshold."""
    risks = sorted({round(s.confirmed_risk, 4) for s in scored})
    # Also evaluate midpoints and a fine grid so we are not stuck on discrete masses.
    candidates = set(risks)
    for a, b in zip(risks, risks[1:]):
        candidates.add(round((a + b) / 2, 4))
    for i in range(0, 101):
        candidates.add(round(i / 100, 2))
    candidates.add(0.35)

    y_true = [1 if s.negative_outcome else 0 for s in scored]
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos

    best: tuple[float, float, float, float] | None = None
    # (J, f1, -threshold, threshold) for ranking
    ranking: list[tuple[float, float, float, float, dict[str, float]]] = []

    for thr in sorted(candidates):
        tp = fp = tn = fn = 0
        for s, y in zip(scored, y_true):
            pred = 1 if s.confirmed_risk >= thr else 0
            if pred == 1 and y == 1:
                tp += 1
            elif pred == 1 and y == 0:
                fp += 1
            elif pred == 0 and y == 0:
                tn += 1
            else:
                fn += 1
        tpr = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tpr
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        j = tpr - fpr
        metrics = {
            "threshold": thr,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "tpr": tpr,
            "fpr": fpr,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "youden_j": j,
        }
        ranking.append((j, f1, -thr, thr, metrics))

    ranking.sort(reverse=True)
    return ranking[0][3], ranking[0][4]


def main() -> None:
    dataset_dir = _REPO_ROOT / "dataset"
    scored = score_history(dataset_dir)
    neg = [s for s in scored if s.negative_outcome]
    pos = [s for s in scored if not s.negative_outcome]

    print(f"scored messages: {len(scored)}")
    print(
        f"negative outcomes (muted_after_message OR message_reported): {len(neg)}"
    )
    print(f"  muted_after_message=1: {sum(1 for s in scored if s.muted)}")
    print(f"  message_reported=1: {sum(1 for s in scored if s.reported)}")
    print(f"everything else: {len(pos)}")
    print(
        f"domain_mismatch among scored: {sum(1 for s in scored if s.domain_mismatch)}"
    )
    print()

    summarize(
        "confirmed_risk | muted/reported",
        [s.confirmed_risk for s in neg],
    )
    print()
    summarize(
        "confirmed_risk | everything else",
        [s.confirmed_risk for s in pos],
    )
    print()

    thr, metrics = best_threshold(scored)
    print("Best separating threshold (max Youden J = TPR-FPR; tie-break F1, then lower thr):")
    for key in (
        "threshold",
        "youden_j",
        "f1",
        "precision",
        "recall",
        "tpr",
        "fpr",
        "tp",
        "fp",
        "tn",
        "fn",
    ):
        val = metrics[key]
        if isinstance(val, float):
            print(f"  {key}={val:.4f}")
        else:
            print(f"  {key}={val}")

    # Metrics at the previous heuristic 0.35 for comparison.
    y_true = [1 if s.negative_outcome else 0 for s in scored]
    tp = fp = tn = fn = 0
    for s, y in zip(scored, y_true):
        pred = 1 if s.confirmed_risk >= 0.35 else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 0:
            tn += 1
        else:
            fn += 1
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    tpr = tp / n_pos if n_pos else 0.0
    fpr = fp / n_neg if n_neg else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * prec * tpr / (prec + tpr)) if (prec + tpr) else 0.0
    print()
    print("Comparison at previous heuristic threshold 0.35:")
    print(
        f"  youden_j={tpr - fpr:.4f} f1={f1:.4f} precision={prec:.4f} "
        f"recall={tpr:.4f} tpr={tpr:.4f} fpr={fpr:.4f} "
        f"tp={tp} fp={fp} tn={tn} fn={fn}"
    )
    print()
    if abs(thr - 0.35) < 1e-9:
        print("VERDICT: data-driven threshold is still 0.35")
    else:
        print(f"VERDICT: data-driven threshold is {thr:.4f} (NOT 0.35)")


if __name__ == "__main__":
    main()
