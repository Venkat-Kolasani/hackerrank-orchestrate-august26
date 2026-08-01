"""Deterministic offline router (Stages 1–2).

Stage 1: typed loading + exact output contract.
Stage 2: recipient-specific historical features + evidence retrieval.
Safety/priority modules remain authoritative for mute/notify ceilings.
"""

from __future__ import annotations

import re

from .data import Dataset
from .evidence import retrieve_evidence
from .features import MessageFeatures, compute_features, in_quiet_hours
from .priority import DEFAULT_WEIGHTS, PriorityTerms, decide_action
from .safety import assess_risk
from .types import ContentSummary, MessageRecord, Prediction

# Re-export for existing Stage 1 tests.
__all__ = ["in_quiet_hours", "route_dataset", "route_message"]

_PROMO_RE = re.compile(
    r"\b("
    r"sale|discount|off\b|promo|deal|coupon|limited\s+offer|"
    r"flash\s+sale|buy\s+now|free\s+shipping|unsubscribe"
    r")\b",
    re.IGNORECASE,
)
_PAYMENT_RE = re.compile(
    r"\b("
    r"payment|invoice|upi|refund|order\s+#?\d+|delivered|shipment|"
    r"booking|reservation|amount\s*(?:rs\.?|₹)|pay\s+now"
    r")\b",
    re.IGNORECASE,
)
_EVENT_RE = re.compile(
    r"\b("
    r"event|invite|invitation|rsvp|tomorrow|schedule|meeting|"
    r"webinar|party|gathering|assembly|pta|school"
    r")\b",
    re.IGNORECASE,
)
_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening)|namaste)\b",
    re.IGNORECASE,
)
_URGENT_RE = re.compile(
    r"\b("
    r"urgent|asap|immediately|right\s+now|today\s+only|same[-\s]?day|"
    r"deadline|expires?\s+today|last\s+chance|emergency|due\s+(today|tonight|now)"
    r")\b",
    re.IGNORECASE,
)
_OTP_RE = re.compile(
    r"\b(?:otp|one[-\s]?time\s+password|verification\s+code|cvv|card\s+number)\b",
    re.IGNORECASE,
)


def _infer_message_type(
    *,
    text: str,
    action: str,
    conversation_type: str,
    forced_mute: bool,
    scam_signals: list[str],
    forwarded_count: int,
) -> str:
    if forced_mute or any(
        signal in {"otp_request", "credential_harvest", "wallet_verification"}
        for signal in scam_signals
    ):
        return "scam"
    if action == "mute" and forwarded_count >= 5:
        return "spam"
    if forwarded_count >= 3 and not text.strip():
        return "forward"
    if _OTP_RE.search(text) and action != "notify":
        return "scam"
    if _GREETING_RE.search(text) and len(text) < 80:
        return "greeting"
    if _PROMO_RE.search(text):
        return "promotion"
    if _PAYMENT_RE.search(text):
        return "payment" if conversation_type != "business" else "business_update"
    if _URGENT_RE.search(text):
        return "urgent"
    if _EVENT_RE.search(text):
        return "event"
    if conversation_type == "business":
        return "business_update"
    if conversation_type == "personal":
        return "personal"
    if forwarded_count > 0:
        return "forward"
    return "unknown"


def _build_reason(
    *,
    action: str,
    message_type: str,
    risk_reasons: list[str],
    features: MessageFeatures,
    evidence: str,
) -> str:
    parts: list[str] = []
    if risk_reasons:
        parts.append(risk_reasons[0].rstrip("."))
    elif features.domain_mismatch:
        parts.append("Sender domain does not match the official business domain")
    elif action == "mute" and features.opted_out:
        parts.append("User opted out of promotions from this business")
    elif action == "mute" and features.repeated_template:
        parts.append("Recipient previously ignored a near-duplicate of this message")
    elif action == "mute" and features.muted_group:
        parts.append("Recipient has muted this group and the message is not high urgency")
    elif action == "mute" and features.channel_dismiss_rate >= 0.5:
        parts.append("Recipient usually dismisses messages on this conversation")
    elif action == "notify" and features.verified_business:
        parts.append("Verified business context with a time-sensitive operational signal")
    elif action == "notify" and features.sender_is_group_admin:
        parts.append("Group admin sent a high-priority update for this recipient")
    elif action == "notify":
        parts.append("Trusted or urgent context warrants an immediate interrupt")
    elif action == "digest" and features.quiet_hours:
        parts.append("Useful update deferred because it arrived during quiet hours")
    elif action == "digest" and features.same_entity_history_count > 0:
        parts.append("Useful for later review based on this recipient's channel history")
    elif action == "digest":
        parts.append("Useful for later review but not interrupt-worthy")
    else:
        parts.append(f"Routed as {action} with type {message_type}")

    if evidence != "none":
        parts.append("supported by prior recipient reactions")
    if features.media_uninterpreted and not features.text.strip():
        parts.append("media was present but left uninterpreted in the offline baseline")

    reason = ". ".join(parts).strip()
    if not reason.endswith("."):
        reason += "."
    return reason[:280]


def _confidence_from_decision(
    *,
    action: str,
    raw_action: str,
    priority: float,
    forced_mute: bool,
    media_uninterpreted: bool,
    risk_score: float,
    evidence: str,
    features: MessageFeatures,
) -> float:
    if forced_mute:
        base = 0.82 + min(0.12, risk_score * 0.1)
    elif action == "notify":
        base = 0.55 + min(0.30, max(0.0, priority) * 0.5)
    elif action == "digest":
        base = 0.50 + min(0.25, abs(priority) * 0.4)
    else:
        base = 0.55 + min(0.25, abs(priority) * 0.35)

    if action != raw_action:
        base -= 0.08
    if media_uninterpreted:
        base -= 0.12
    if evidence != "none":
        base += 0.05
    elif features.same_entity_history_count == 0:
        base -= 0.04
    return round(max(0.35, min(0.95, base)), 4)


def route_message(dataset: Dataset, message: MessageRecord) -> Prediction:
    """Route one message using Stage 2 features and evidence."""
    features = compute_features(dataset, message)
    content = ContentSummary(
        message_text=message.message_text,
        media_type=message.media_type,
        media_id=message.media_id,
    )
    risk = assess_risk(
        content,
        domain_mismatch=features.domain_mismatch,
        user_reports_30d=features.user_reports_30d,
        forwarded_count=message.forwarded_count,
    )

    terms = PriorityTerms(
        urgency=features.urgency,
        direct_mention=features.direct_mention,
        sender_trust=features.sender_trust,
        personal_relevance=features.personal_relevance,
        positive_history=features.positive_history,
        repetition=features.repetition,
        low_engagement=features.low_engagement,
        quiet_hour_cost=features.quiet_hour_cost,
        confirmed_risk=risk.risk_score,
        domain_mismatch=features.domain_mismatch,
        credential_otp_or_payment_request=bool(
            set(risk.scam_signals)
            & {"otp_request", "credential_harvest", "urgent_payment_link"}
        ),
        sender_is_group_admin=features.sender_is_group_admin,
    )
    decision = decide_action(
        terms,
        DEFAULT_WEIGHTS,
        hard_blocked_by_safety=risk.forced_mute,
    )

    evidence = retrieve_evidence(
        dataset,
        message,
        decision.action,
        features,
    )

    message_type = _infer_message_type(
        text=features.text,
        action=decision.action,
        conversation_type=message.conversation_type,
        forced_mute=risk.forced_mute,
        scam_signals=risk.scam_signals,
        forwarded_count=message.forwarded_count,
    )
    reason = _build_reason(
        action=decision.action,
        message_type=message_type,
        risk_reasons=risk.reasons,
        features=features,
        evidence=evidence,
    )
    confidence = _confidence_from_decision(
        action=decision.action,
        raw_action=decision.raw_action,
        priority=decision.priority,
        forced_mute=risk.forced_mute,
        media_uninterpreted=features.media_uninterpreted,
        risk_score=risk.risk_score,
        evidence=evidence,
        features=features,
    )

    return Prediction(
        message_id=message.message_id,
        action=decision.action,
        message_type=message_type,
        reason=reason,
        confidence=confidence,
        evidence_message_ids=evidence,
    )


def route_dataset(dataset: Dataset) -> list[Prediction]:
    """Route every incoming message in deterministic input order."""
    return [route_message(dataset, message) for message in dataset.messages]
