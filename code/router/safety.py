"""Risk scoring and hard safety overrides.

Injection scanning and metadata risk signals run here, independently of any
LLM call. Hits raise ``risk_score`` before a model ever sees the message.
The hard mute gate can only move decisions toward mute, never toward notify.

``confirmed_risk`` in priority.py is fed from ``risk_score``. Domain mismatch,
report volume, and forward volume are first-class scored inputs — not flags
that only matter if a keyword also fires — so the score-only notify ceiling
actually arms when those signals are present alone.
"""

from __future__ import annotations

import re
from typing import Optional

from .injection import injection_score, scan_content
from .types import ContentSummary, RiskAssessment

# Hard mute when risk reaches this threshold. Injection alone can force mute
# when multiple override signals fire; otherwise it compounds with scam bait.
HARD_MUTE_THRESHOLD = 0.75

# Metadata risk weights. domain_mismatch alone must clear the notify ceiling
# floor (TECHNICAL_RISK_FLOOR, currently 0.18) without keyword/injection hits.
DOMAIN_MISMATCH_WEIGHT = 0.40
REPORT_WEIGHT = 0.35
REPORT_SATURATION_COUNT = 20  # user_reports_30d at which report contrib maxes
FORWARD_WEIGHT = 0.25
FORWARD_SATURATION_COUNT = 10  # forwarded_count at which forward contrib maxes

_SCAM_PATTERNS: tuple[tuple[str, float, re.Pattern[str]], ...] = (
    (
        "otp_request",
        0.35,
        re.compile(
            r"\b(?:otp|one[-\s]?time\s+password|verification\s+code)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "wallet_verification",
        0.25,
        re.compile(
            r"\b(?:wallet|account|payment).{0,30}\b(?:verif(?:y|ication)|failed|suspend|locked)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "credential_harvest",
        0.30,
        re.compile(
            r"\b(?:reply with|send|share|enter)\b.{0,40}\b(?:otp|password|pin|cvv|card\s+number)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "urgent_payment_link",
        0.25,
        re.compile(
            r"\b(?:pay(?:ment)?|fee|reattempt)\b.{0,40}\b(?:http|www\.|\.in\b|\.com\b)",
            re.IGNORECASE,
        ),
    ),
)


def _base_scam_score(text: str) -> tuple[float, list[str]]:
    if not text.strip():
        return 0.0, []
    score = 0.0
    signals: list[str] = []
    for signal_id, weight, pattern in _SCAM_PATTERNS:
        if pattern.search(text):
            score += weight
            signals.append(signal_id)
    return min(score, 1.0), signals


def _metadata_score(
    *,
    domain_mismatch: bool,
    user_reports_30d: int,
    forwarded_count: int,
) -> tuple[float, list[str], dict[str, float]]:
    """Score sender/channel metadata independently of message keywords."""
    score = 0.0
    signals: list[str] = []
    parts: dict[str, float] = {
        "domain_mismatch": 0.0,
        "report_count": 0.0,
        "forwarded_count": 0.0,
    }

    if domain_mismatch:
        parts["domain_mismatch"] = DOMAIN_MISMATCH_WEIGHT
        score += DOMAIN_MISMATCH_WEIGHT
        signals.append("domain_mismatch")

    reports = max(0, int(user_reports_30d))
    if reports > 0:
        report_contrib = REPORT_WEIGHT * min(1.0, reports / REPORT_SATURATION_COUNT)
        parts["report_count"] = round(report_contrib, 4)
        score += report_contrib
        signals.append(f"report_count:{reports}")

    forwards = max(0, int(forwarded_count))
    if forwards > 0:
        forward_contrib = FORWARD_WEIGHT * min(
            1.0, forwards / FORWARD_SATURATION_COUNT
        )
        parts["forwarded_count"] = round(forward_contrib, 4)
        score += forward_contrib
        signals.append(f"forwarded_count:{forwards}")

    return min(score, 1.0), signals, parts


def assess_risk(
    content: ContentSummary,
    *,
    domain_mismatch: bool = False,
    user_reports_30d: int = 0,
    forwarded_count: int = 0,
    apply_injection: bool = True,
    hard_mute_threshold: float = HARD_MUTE_THRESHOLD,
) -> RiskAssessment:
    """Compute explainable risk before any LLM decision.

    Parameters
    ----------
    domain_mismatch:
        True when sender domain disagrees with the official/known domain.
        Scored on its own — does not require a keyword or injection hit.
    user_reports_30d:
        Recent report volume for the business/sender channel.
    forwarded_count:
        Forwarding fan-out on this message.
    apply_injection:
        When False, skip the deterministic injection scanner. Used by tests
        to show before/after risk scores for the same content.
    """
    joined = content.joined_text()
    base_score, scam_signals = _base_scam_score(joined)

    hits = scan_content(content) if apply_injection else []
    inj_score = injection_score(hits) if hits else 0.0

    meta_score, meta_signals, meta_parts = _metadata_score(
        domain_mismatch=domain_mismatch,
        user_reports_30d=user_reports_30d,
        forwarded_count=forwarded_count,
    )

    # Saturating combination of keyword, injection, and metadata risk.
    risk_score = round(min(1.0, base_score + inj_score + meta_score), 4)
    forced_mute = risk_score >= hard_mute_threshold

    reasons: list[str] = []
    if hits:
        channels = sorted({hit.channel for hit in hits})
        patterns = sorted({hit.pattern_id for hit in hits})
        reasons.append(
            "Deterministic injection scanner matched "
            f"{', '.join(patterns)} in {', '.join(channels)}"
        )
    if scam_signals:
        reasons.append(f"Scam-like content signals: {', '.join(scam_signals)}")
    if meta_signals:
        reasons.append(
            "Metadata risk signals: "
            + ", ".join(
                f"{name}={meta_parts[name]:.2f}"
                for name in ("domain_mismatch", "report_count", "forwarded_count")
                if meta_parts[name] > 0
            )
        )
    if forced_mute:
        reasons.append(
            f"risk_score {risk_score:.2f} >= hard mute threshold "
            f"{hard_mute_threshold:.2f}"
        )

    return RiskAssessment(
        risk_score=risk_score,
        base_score=round(base_score, 4),
        injection_score=inj_score,
        metadata_score=round(meta_score, 4),
        forced_mute=forced_mute,
        reasons=reasons,
        injection_hits=list(hits),
        scam_signals=scam_signals,
        metadata_signals=meta_signals,
        metadata_parts=meta_parts,
    )


def should_force_mute(assessment: RiskAssessment) -> bool:
    return assessment.forced_mute
