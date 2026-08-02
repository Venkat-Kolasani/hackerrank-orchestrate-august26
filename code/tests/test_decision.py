"""Stage 5 decision-layer regression tests."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

_CODE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _CODE_ROOT.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from router.baseline import route_dataset
from router.data import load_dataset
from router.decision import (
    DecisionContext,
    apply_safety_constraints,
    build_reason,
    decide,
    deterministic_decide,
    filter_evidence_ids,
    validate_decision_payload,
)
from router.features import MessageFeatures
from router.media import build_media_interpreter
from router.priority import PriorityDecision, PriorityTerms, decide_action, DEFAULT_WEIGHTS
from router.types import ContentSummary, MessageRecord, RiskAssessment


def _message(**overrides) -> MessageRecord:
    base = dict(
        message_id="msg_test",
        user_id="u_001",
        conversation_type="group",
        group_id="g_001",
        business_id=None,
        sender_user_id="u_002",
        created_at=datetime(2026, 7, 30, 12, 0),
        message_text="Please review the maintenance schedule for tomorrow.",
        media_type=None,
        media_id=None,
        forwarded_count=0,
    )
    base.update(overrides)
    return MessageRecord(**base)


def _risk(**overrides) -> RiskAssessment:
    base = dict(
        risk_score=0.0,
        base_score=0.0,
        injection_score=0.0,
        forced_mute=False,
        metadata_score=0.0,
        reasons=[],
        scam_signals=[],
    )
    base.update(overrides)
    return RiskAssessment(**base)


def _features(**overrides) -> MessageFeatures:
    feats = MessageFeatures(
        urgency=0.2,
        sender_trust=0.5,
        personal_relevance=0.4,
        positive_history=0.4,
        text="hello",
    )
    for key, value in overrides.items():
        setattr(feats, key, value)
    return feats


def _priority(action: str, *, ceiling: bool = False, raw: str | None = None) -> PriorityDecision:
    return PriorityDecision(
        priority=0.5 if action == "notify" else (0.2 if action == "digest" else 0.0),
        action=action,
        raw_action=raw or action,
        technical_ceiling_active=ceiling,
        technical_ceiling_applied=ceiling and action == "digest" and (raw or action) == "notify",
        mention_override_active=False,
        mention_override_applied=False,
        hard_blocked_by_safety=False,
        technical_indicators=(),
    )


def test_safety_cannot_weaken_hard_mute():
    action, clamped = apply_safety_constraints(
        "notify",
        forced_mute=True,
        notify_ceiling_active=False,
        mention_override_active=True,  # even with mention armed, mute wins
    )
    assert action == "mute"
    assert clamped is True


def test_notify_ceiling_clamps_model_notify_to_digest():
    action, clamped = apply_safety_constraints(
        "notify",
        forced_mute=False,
        notify_ceiling_active=True,
        mention_override_active=True,  # ceiling blocks mention override
    )
    assert action == "digest"
    assert clamped is True


def test_mention_override_forces_notify_post_hoc():
    action, clamped = apply_safety_constraints(
        "digest",
        forced_mute=False,
        notify_ceiling_active=False,
        mention_override_active=True,
    )
    assert action == "notify"
    assert clamped is True


def test_finalize_model_decision_clamps_raw_notify_under_ceiling():
    from router.decision import _finalize_model_decision

    ctx = DecisionContext(
        message=_message(),
        content=ContentSummary(message_text="Flash sale today only — buy now"),
        features=_features(urgency=0.8, domain_mismatch=True),
        risk=_risk(risk_score=0.5, reasons=["Metadata risk signals: domain_mismatch=0.40"]),
        priority=_priority("digest", ceiling=True, raw="notify"),
        allowed_evidence_ids=["hist_1"],
    )
    raw = {
        "action": "notify",
        "message_type": "promotion",
        "reason": "Verified-looking promo with same-day urgency language.",
        "confidence": 0.91,
        "evidence_message_ids": "hist_1",
    }
    result = _finalize_model_decision(ctx, raw=raw, source="test")
    assert result is not None
    assert result.model_proposed_action == "notify"
    assert result.action == "digest"
    assert result.clamped_by_safety is True
    assert result.raw_model_json == raw


def test_filter_evidence_rejects_unknown_ids():
    assert filter_evidence_ids("hist_1;evil_x", ["hist_1", "hist_2"]) == "hist_1"
    assert filter_evidence_ids("none", ["hist_1"]) == "none"
    assert filter_evidence_ids("evil_x", ["hist_1"]) == "none"


def test_validate_rejects_generic_reason_and_bad_enums():
    assert (
        validate_decision_payload(
            {
                "action": "notify",
                "message_type": "urgent",
                "reason": "based on analysis",
                "confidence": 0.9,
                "evidence_message_ids": "none",
            },
            allowed_evidence_ids=[],
        )
        is None
    )
    assert (
        validate_decision_payload(
            {
                "action": "ping",
                "message_type": "urgent",
                "reason": "Admin deadline today.",
                "confidence": 0.9,
                "evidence_message_ids": "none",
            },
            allowed_evidence_ids=[],
        )
        is None
    )


def test_deterministic_mute_boundary_for_forced_risk():
    ctx = DecisionContext(
        message=_message(message_text="Reply with OTP to unlock wallet http://evil.test"),
        content=ContentSummary(message_text="Reply with OTP to unlock wallet http://evil.test"),
        features=_features(urgency=0.9, sender_trust=0.9, domain_mismatch=True),
        risk=_risk(
            risk_score=0.9,
            forced_mute=True,
            reasons=["risk_score 0.90 >= hard mute threshold 0.75"],
            scam_signals=["otp_request", "credential_harvest"],
        ),
        priority=_priority("notify"),  # personalization would have wanted notify
        allowed_evidence_ids=["hist_1"],
    )
    result = deterministic_decide(ctx)
    assert result.action == "mute"
    assert result.message_type in {"scam", "spam"}
    assert "risk" in result.reason.lower() or "otp" in result.reason.lower() or "mute" in result.reason.lower()


def test_deterministic_notify_vs_digest_quiet_hours():
    # Urgent admin mention-like: high urgency + admin.
    notify_ctx = DecisionContext(
        message=_message(message_text="Water cut today at 3pm — prepare storage."),
        content=ContentSummary(message_text="Water cut today at 3pm — prepare storage."),
        features=_features(
            urgency=0.85,
            sender_trust=0.7,
            sender_is_group_admin=True,
            direct_mention=0.0,
            quiet_hours=False,
            quiet_hour_cost=0.0,
            text="Water cut today at 3pm — prepare storage.",
        ),
        risk=_risk(risk_score=0.0),
        priority=decide_action(
            PriorityTerms(
                urgency=0.85,
                sender_trust=0.7,
                sender_is_group_admin=True,
                personal_relevance=0.4,
                positive_history=0.4,
            ),
            DEFAULT_WEIGHTS,
        ),
        allowed_evidence_ids=[],
    )
    notify_result = deterministic_decide(notify_ctx)
    assert notify_result.action in {"notify", "digest"}
    assert "admin" in notify_result.reason.lower() or "urgent" in notify_result.reason.lower() or "time-sensitive" in notify_result.reason.lower() or "interrupt" in notify_result.reason.lower()

    digest_ctx = DecisionContext(
        message=_message(
            message_text="Weekly society newsletter with garage sale tips.",
            created_at=datetime(2026, 7, 30, 23, 30),
        ),
        content=ContentSummary(message_text="Weekly society newsletter with garage sale tips."),
        features=_features(
            urgency=0.1,
            quiet_hours=True,
            quiet_hour_cost=0.85,
            personal_relevance=0.35,
            text="Weekly society newsletter with garage sale tips.",
            same_entity_history_count=2,
        ),
        risk=_risk(risk_score=0.0),
        priority=decide_action(
            PriorityTerms(
                urgency=0.1,
                quiet_hour_cost=0.85,
                personal_relevance=0.35,
                positive_history=0.3,
                sender_trust=0.4,
            ),
            DEFAULT_WEIGHTS,
        ),
        allowed_evidence_ids=["hist_a"],
    )
    digest_result = deterministic_decide(digest_ctx)
    assert digest_result.action in {"digest", "mute"}
    assert digest_result.action != "notify"
    assert "quiet" in digest_result.reason.lower() or "later" in digest_result.reason.lower() or "dismiss" in digest_result.reason.lower() or "promotion" in digest_result.reason.lower() or "review" in digest_result.reason.lower()


def test_model_payload_cannot_override_hard_mute(monkeypatch):
    """Simulate a model saying notify; safety must force mute."""
    calls = {}

    def fake_openai(ctx, *, model):
        from router.decision import DecisionResult, apply_safety_constraints

        action, clamped = apply_safety_constraints(
            "notify",
            forced_mute=ctx.risk.forced_mute,
            notify_ceiling_active=ctx.priority.technical_ceiling_active,
            mention_override_active=ctx.priority.mention_override_active,
        )
        calls["hit"] = True
        return DecisionResult(
            action=action,
            message_type="urgent",
            reason="Model wanted notify but safety may clamp.",
            confidence=0.9,
            evidence_message_ids="none",
            source="openai",
            clamped_by_safety=clamped,
        )

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ROUTER_DECISION_PROVIDER", "openai")
    monkeypatch.setattr("router.decision._openai_decide", fake_openai)

    ctx = DecisionContext(
        message=_message(),
        content=ContentSummary(message_text="hi"),
        features=_features(),
        risk=_risk(forced_mute=True, risk_score=0.9, reasons=["forced"]),
        priority=_priority("mute"),
        allowed_evidence_ids=[],
    )
    result = decide(ctx)
    assert calls.get("hit") is True
    assert result.action == "mute"
    assert result.clamped_by_safety is True


def test_build_reason_is_signal_specific():
    reason = build_reason(
        action="mute",
        message_type="scam",
        risk_reasons=["Scam-like content signals: otp_request"],
        features=_features(domain_mismatch=True),
        evidence="none",
    )
    assert "otp" in reason.lower() or "scam" in reason.lower()
    assert "based on analysis" not in reason.lower()


def test_router_offline_decision_end_to_end():
    dataset = load_dataset(_REPO_ROOT / "dataset")
    interpreter = build_media_interpreter(provider="offline", cache=False)
    predictions = route_dataset(dataset, interpreter=interpreter)
    assert len(predictions) == len(dataset.messages)
    actions = {p.action for p in predictions}
    assert actions <= {"notify", "digest", "mute"}
    for pred in predictions:
        assert pred.reason.strip()
        assert "based on analysis" not in pred.reason.lower()
        assert 0.0 <= pred.confidence <= 1.0
