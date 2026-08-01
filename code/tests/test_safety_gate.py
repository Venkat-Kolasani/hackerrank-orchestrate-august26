"""Stage 3 safety gate: trusted payments vs untrusted OTP/payment lures.

General rules only — no hardcoded sample_msg IDs as answers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from router.priority import (
    DEFAULT_WEIGHTS,
    HISTORY_HEAVY_WEIGHTS,
    TECHNICAL_RISK_FLOOR,
    PriorityTerms,
    decide_action,
)
from router.safety import (
    ACCOUNT_AGE_WEIGHT,
    DOMAIN_AGE_WEIGHT,
    HARD_MUTE_THRESHOLD,
    assess_risk,
)
from router.types import ContentSummary


# Legitimate transactional updates — must not hard-mute.
_TRUSTED_PAYMENTS: tuple[str, ...] = (
    "Your UPI payment of Rs. 450 to BigBasket was successful. "
    "Order #12345 will be delivered today.",
    "HDFC Bank: Rs 2,500 debited from A/c XX1234 on 01-Aug for UPI/merchant. "
    "Available bal Rs 12,340.",
    "Payment of INR 899 for your electricity bill was received. "
    "Receipt has been emailed to you.",
    "Your refund of Rs. 120 has been credited for order #99881. "
    "No further action is needed.",
)


# Untrusted credential / payment requests — elevate risk / mute path.
_UNTRUSTED_LURES: tuple[str, ...] = (
    "Your wallet verification failed. Reply with the OTP immediately "
    "to avoid account suspension.",
    "URGENT: payment failed. Reattempt fee at http://pay-secure-wallet.com "
    "and enter OTP to keep access active.",
    "Account will be locked within 2 hours. Send your OTP and password "
    "to restore wallet verification.",
    "Act now to avoid permanent suspension. Share the verification code "
    "and pay the unlock fee via http://unlock-now.biz",
)


@pytest.mark.parametrize("text", _TRUSTED_PAYMENTS, ids=lambda t: t[:40])
def test_trusted_legitimate_payment_not_hard_muted(text: str) -> None:
    assessment = assess_risk(
        ContentSummary(message_text=text),
        domain_mismatch=False,
        user_reports_30d=0,
        forwarded_count=0,
        account_age_days=400,
        domain_age_days=400,
    )
    assert not assessment.forced_mute, (
        f"trusted payment must not hard-mute (score={assessment.risk_score}, "
        f"signals={assessment.scam_signals})"
    )
    assert assessment.risk_score < HARD_MUTE_THRESHOLD
    # Legitimate receipts should not trip phishing keyword patterns.
    assert "credential_harvest" not in assessment.scam_signals
    assert "otp_request" not in assessment.scam_signals
    assert "urgent_payment_link" not in assessment.scam_signals


@pytest.mark.parametrize("text", _UNTRUSTED_LURES, ids=lambda t: t[:40])
def test_untrusted_otp_payment_elevates_risk(text: str) -> None:
    assessment = assess_risk(
        ContentSummary(message_text=text),
        domain_mismatch=True,
        user_reports_30d=4,
        forwarded_count=1,
        account_age_days=7,
        domain_age_days=3,
    )
    assert assessment.risk_score >= TECHNICAL_RISK_FLOOR
    assert assessment.scam_signals, "expected keyword scam signals"
    assert any(
        signal in assessment.scam_signals
        for signal in (
            "otp_request",
            "credential_harvest",
            "wallet_verification",
            "urgent_payment_link",
            "urgency_pressure",
        )
    )
    # Strong untrusted lures should hard-mute (or at least clear the ceiling).
    assert assessment.risk_score >= TECHNICAL_RISK_FLOOR
    if assessment.risk_score >= HARD_MUTE_THRESHOLD:
        assert assessment.forced_mute


def test_untrusted_otp_hard_mute_beats_personalization() -> None:
    """High trust / history / mention cannot restore notify past hard mute."""
    text = (
        "Wallet verification failed. Reply with the OTP immediately to avoid "
        "suspension. Pay unlock fee at http://fake-pay.example"
    )
    assessment = assess_risk(
        ContentSummary(message_text=text),
        domain_mismatch=True,
        user_reports_30d=8,
        forwarded_count=2,
        account_age_days=5,
        domain_age_days=2,
    )
    assert assessment.forced_mute
    assert assessment.risk_score >= HARD_MUTE_THRESHOLD

    terms = PriorityTerms(
        urgency=1.0,
        direct_mention=1.0,
        sender_trust=1.0,
        personal_relevance=1.0,
        positive_history=1.0,
        repetition=0.0,
        low_engagement=0.0,
        quiet_hour_cost=0.0,
        confirmed_risk=assessment.risk_score,
        domain_mismatch=True,
        credential_otp_or_payment_request=True,
        sender_is_group_admin=True,
    )
    for label, weights in (
        ("DEFAULT", DEFAULT_WEIGHTS),
        ("HISTORY_HEAVY", HISTORY_HEAVY_WEIGHTS),
    ):
        decision = decide_action(
            terms, weights, hard_blocked_by_safety=assessment.forced_mute
        )
        assert decision.action == "mute", f"{label} must stay mute under hard block"
        assert decision.hard_blocked_by_safety
        assert not decision.mention_override_applied


def test_young_account_and_domain_age_contribute_metadata_risk() -> None:
    clean = ContentSummary(
        message_text="Society maintenance reminder for next Tuesday evening."
    )
    mature = assess_risk(
        clean,
        account_age_days=400,
        domain_age_days=400,
    )
    young = assess_risk(
        clean,
        account_age_days=0,
        domain_age_days=0,
    )
    assert mature.metadata_score == 0.0
    assert young.metadata_score > mature.metadata_score
    assert young.metadata_parts["young_account"] == pytest.approx(ACCOUNT_AGE_WEIGHT)
    assert young.metadata_parts["young_domain"] == pytest.approx(DOMAIN_AGE_WEIGHT)
    # Brand-new ages alone should arm the notify ceiling, not hard-mute.
    assert young.risk_score >= TECHNICAL_RISK_FLOOR
    assert young.risk_score < HARD_MUTE_THRESHOLD
    assert not young.forced_mute


def test_urgency_pressure_signal_without_false_positive_on_ops() -> None:
    pressure = assess_risk(
        ContentSummary(
            message_text=(
                "Act now to avoid permanent suspension of your account access."
            )
        )
    )
    assert "urgency_pressure" in pressure.scam_signals
    assert pressure.base_score > 0

    ops = assess_risk(
        ContentSummary(
            message_text=(
                "Tower B water tanker arrives in 20 mins. Please store drinking water."
            )
        )
    )
    assert "urgency_pressure" not in ops.scam_signals
    assert ops.risk_score == 0.0


def test_spam_path_high_forward_metadata_without_phishing_keywords() -> None:
    """Heavy forwarding + reports raise risk without needing OTP keywords."""
    assessment = assess_risk(
        ContentSummary(message_text="Check this out when free."),
        domain_mismatch=False,
        user_reports_30d=20,
        forwarded_count=10,
        account_age_days=200,
        domain_age_days=200,
    )
    assert assessment.base_score == 0.0
    assert assessment.scam_signals == []
    assert assessment.metadata_score > 0
    assert assessment.risk_score >= TECHNICAL_RISK_FLOOR
