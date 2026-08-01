"""Domain mismatch / report / forward metadata must score in assess_risk alone.

Re-runs the earlier high-history admin domain-mismatch cases with clean text
(no OTP/keyword/injection) so confirmed_risk reflects the mismatch by itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from router.priority import (
    DEFAULT_WEIGHTS,
    HISTORY_HEAVY_WEIGHTS,
    MENTION_AND_RISK_WEIGHTS,
    TECHNICAL_RISK_FLOOR,
    PriorityTerms,
    decide_action,
    technical_notify_ceiling_active,
)
from router.safety import (
    DOMAIN_MISMATCH_WEIGHT,
    assess_risk,
)
from router.types import ContentSummary

# Benign operational text — must not trip keyword scam or injection scanners.
_CLEAN_ADMIN_TEXT = (
    "Maintenance window moved to 6 PM. Please use the portal link from the "
    "society office notice board."
)


def _print_decision(label: str, terms: PriorityTerms) -> None:
    print(f"\n=== {label} ===")
    print(f"  confirmed_risk={terms.confirmed_risk:.4f}")
    print(f"  ceiling_active={technical_notify_ceiling_active(terms)}")
    print(
        f"  domain_mismatch={terms.domain_mismatch} "
        f"credential_flag={terms.credential_otp_or_payment_request}"
    )
    for profile, weights in (
        ("DEFAULT", DEFAULT_WEIGHTS),
        ("MENTION_AND_RISK", MENTION_AND_RISK_WEIGHTS),
        ("HISTORY_HEAVY", HISTORY_HEAVY_WEIGHTS),
    ):
        decision = decide_action(terms, weights)
        print(
            f"  {profile}: priority={decision.priority:+.4f} "
            f"raw={decision.raw_action} -> {decision.action} "
            f"(ceiling_applied={decision.technical_ceiling_applied}, "
            f"mention_applied={decision.mention_override_applied})"
        )
        assert decision.action != "notify", (
            f"{label}/{profile}: domain-mismatch risk must not notify"
        )


def test_domain_mismatch_alone_scores_and_arms_ceiling() -> None:
    assessment = assess_risk(
        ContentSummary(message_text=_CLEAN_ADMIN_TEXT),
        domain_mismatch=True,
        user_reports_30d=0,
        forwarded_count=0,
        apply_injection=True,
    )

    print("\ndomain_mismatch alone (no keywords, no injection, no reports/forwards)")
    print(f"  DOMAIN_MISMATCH_WEIGHT={DOMAIN_MISMATCH_WEIGHT}")
    print(f"  base_score={assessment.base_score:.4f}")
    print(f"  injection_score={assessment.injection_score:.4f}")
    print(f"  metadata_score={assessment.metadata_score:.4f}")
    print(f"  metadata_parts={assessment.metadata_parts}")
    print(f"  risk_score/confirmed_risk={assessment.risk_score:.4f}")
    print(f"  scam_signals={assessment.scam_signals}")
    print(f"  injection_hits={[h.pattern_id for h in assessment.injection_hits]}")

    assert assessment.base_score == 0.0
    assert assessment.injection_score == 0.0
    assert assessment.scam_signals == []
    assert assessment.injection_hits == []
    assert assessment.metadata_parts["domain_mismatch"] == DOMAIN_MISMATCH_WEIGHT
    assert assessment.risk_score == DOMAIN_MISMATCH_WEIGHT
    assert assessment.risk_score >= TECHNICAL_RISK_FLOOR


def test_admin_high_history_domain_mismatch_from_assess_risk() -> None:
    """Replay admin_high_history_domain_mismatch with scored mismatch alone."""
    assessment = assess_risk(
        ContentSummary(message_text=_CLEAN_ADMIN_TEXT),
        domain_mismatch=True,
        user_reports_30d=0,
        forwarded_count=0,
    )
    terms = PriorityTerms(
        urgency=0.6,
        direct_mention=0.0,
        sender_trust=0.95,
        personal_relevance=0.7,
        positive_history=0.95,
        repetition=0.0,
        low_engagement=0.05,
        quiet_hour_cost=0.0,
        confirmed_risk=assessment.risk_score,
        domain_mismatch=True,
        credential_otp_or_payment_request=False,
        sender_is_group_admin=True,
    )
    assert terms.confirmed_risk == DOMAIN_MISMATCH_WEIGHT
    assert technical_notify_ceiling_active(terms)
    _print_decision("admin_high_history_domain_mismatch (mismatch-only risk)", terms)


def test_medium_risk_known_admin_domain_mismatch_variant() -> None:
    """medium_risk_known_admin variant: trusted admin + domain mismatch only."""
    assessment = assess_risk(
        ContentSummary(message_text=_CLEAN_ADMIN_TEXT),
        domain_mismatch=True,
        user_reports_30d=0,
        forwarded_count=0,
    )
    terms = PriorityTerms(
        urgency=0.5,
        direct_mention=0.0,
        sender_trust=0.85,
        personal_relevance=0.5,
        positive_history=0.8,
        repetition=0.0,
        low_engagement=0.1,
        quiet_hour_cost=0.0,
        confirmed_risk=assessment.risk_score,
        domain_mismatch=True,
        credential_otp_or_payment_request=False,
        sender_is_group_admin=True,
    )
    assert terms.confirmed_risk == DOMAIN_MISMATCH_WEIGHT
    assert technical_notify_ceiling_active(terms)
    _print_decision("medium_risk_known_admin domain-mismatch variant", terms)


def test_report_and_forward_weights_contribute() -> None:
    assessment = assess_risk(
        ContentSummary(message_text=_CLEAN_ADMIN_TEXT),
        domain_mismatch=False,
        user_reports_30d=20,
        forwarded_count=10,
    )
    print("\nreport_count + forwarded_count alone")
    print(f"  metadata_parts={assessment.metadata_parts}")
    print(f"  risk_score={assessment.risk_score:.4f}")
    assert assessment.base_score == 0.0
    assert assessment.injection_score == 0.0
    assert assessment.metadata_parts["report_count"] > 0
    assert assessment.metadata_parts["forwarded_count"] > 0
    assert assessment.risk_score == assessment.metadata_score


if __name__ == "__main__":
    test_domain_mismatch_alone_scores_and_arms_ceiling()
    test_admin_high_history_domain_mismatch_from_assess_risk()
    test_medium_risk_known_admin_domain_mismatch_variant()
    test_report_and_forward_weights_contribute()
