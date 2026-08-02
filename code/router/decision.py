"""Constrained contextual decision layer.

Consumes normalized content, deterministic features, and ranked evidence
summaries only. Returns validated action/type/reason/confidence/evidence.
An optional model path (temperature 0) is post-clamped by the safety gate;
the deterministic fallback is used when no API key is configured.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

from . import prompts
from .features import MessageFeatures
from .output import format_evidence
from .priority import PriorityDecision
from .types import (
    ACTIONS,
    MESSAGE_TYPES,
    ContentSummary,
    MessageRecord,
    Prediction,
    RiskAssessment,
)

logger = logging.getLogger(__name__)

ENV_API_KEY = "OPENAI_API_KEY"
ENV_DECISION_PROVIDER = "ROUTER_DECISION_PROVIDER"  # auto | offline | openai
ENV_DECISION_MODEL = "ROUTER_DECISION_MODEL"
DEFAULT_DECISION_MODEL = "gpt-4o-mini"

ACTION_RANK = {"mute": 0, "digest": 1, "notify": 2}

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
    r"deadline|expires?\s+today|last\s+chance|emergency|due\s+(today|tonight|now)|"
    r"for\s+today|leaving\s+early|mins?\s+early|expected\s+to\s+reach|"
    r"delivery\s+today|heads-?up|tanker|bus\s+is\s+leaving|packed\s+and|\d+\s*mins?\b"
    r")\b",
    re.IGNORECASE,
)
_OTP_RE = re.compile(
    r"\b(?:otp|one[-\s]?time\s+password|verification\s+code|cvv|card\s+number)\b",
    re.IGNORECASE,
)
_PHISHING_SIGNALS = frozenset(
    {
        "otp_request",
        "credential_harvest",
        "wallet_verification",
        "urgent_payment_link",
        "urgency_pressure",
    }
)
_GENERIC_REASON_RE = re.compile(
    r"^(based on (the )?analysis|important message|needs attention|"
    r"routed as \w+|as per (the )?system)\.?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvidenceSummary:
    message_id: str
    snippet: str
    reaction: str


@dataclass
class DecisionContext:
    message: MessageRecord
    content: ContentSummary
    features: MessageFeatures
    risk: RiskAssessment
    priority: PriorityDecision
    allowed_evidence_ids: Sequence[str]
    evidence_summaries: Sequence[EvidenceSummary] = field(default_factory=list)
    media_source: str = "offline"


@dataclass(frozen=True)
class DecisionResult:
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: str
    source: str = "fallback"
    clamped_by_safety: bool = False

    def to_prediction(self, message_id: str) -> Prediction:
        return Prediction(
            message_id=message_id,
            action=self.action,
            message_type=self.message_type,
            reason=self.reason,
            confidence=self.confidence,
            evidence_message_ids=self.evidence_message_ids,
        )


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def safer_action(left: str, right: str) -> str:
    """Return the less-interruptive action."""
    if ACTION_RANK.get(left, 1) <= ACTION_RANK.get(right, 1):
        return left if left in ACTION_RANK else "digest"
    return right if right in ACTION_RANK else "digest"


def apply_safety_constraints(
    proposed_action: str,
    *,
    forced_mute: bool,
    notify_ceiling_active: bool,
) -> tuple[str, bool]:
    """Clamp action so the model cannot weaken hard mute or notify ceiling.

    Returns (action, clamped).
    """
    action = proposed_action if proposed_action in ACTIONS else "digest"
    clamped = False
    if forced_mute:
        if action != "mute":
            clamped = True
        return "mute", clamped
    if notify_ceiling_active and action == "notify":
        return "digest", True
    return action, clamped


def infer_message_type(
    *,
    text: str,
    action: str,
    conversation_type: str,
    forced_mute: bool,
    scam_signals: Sequence[str],
    forwarded_count: int,
    injection_hit: bool = False,
) -> str:
    phishing = bool(_PHISHING_SIGNALS.intersection(scam_signals))
    if forced_mute:
        if phishing or injection_hit:
            return "scam"
        if forwarded_count >= 5:
            return "spam"
        return "scam"
    if phishing:
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
    if conversation_type == "group":
        return "personal"
    if forwarded_count > 0:
        return "forward"
    return "unknown"


def build_reason(
    *,
    action: str,
    message_type: str,
    risk_reasons: Sequence[str],
    features: MessageFeatures,
    evidence: str,
    media_source: str = "offline",
) -> str:
    """Signal-based reason; avoids generic boilerplate."""
    parts: list[str] = []
    if risk_reasons:
        parts.append(str(risk_reasons[0]).rstrip("."))
    elif features.domain_mismatch:
        parts.append("Sender domain does not match the official business domain")
    elif action == "mute" and features.opted_out:
        parts.append("User opted out of promotions from this business")
    elif action == "mute" and features.repeated_template:
        parts.append("Recipient previously ignored a near-duplicate of this message")
    elif action == "mute" and features.muted_group and features.urgency < 0.5:
        parts.append("Recipient has muted this group and the message is not high urgency")
    elif action == "mute" and features.channel_dismiss_rate >= 0.5:
        parts.append("Recipient usually dismisses messages on this conversation")
    elif action == "mute" and features.repetition >= 0.7:
        parts.append("Near-duplicate content for this recipient with low engagement")
    elif action == "notify" and features.sender_is_group_admin and features.urgency >= 0.5:
        parts.append("Group admin sent a time-sensitive update that should interrupt now")
    elif action == "notify" and features.verified_business and features.urgency >= 0.5:
        parts.append("Verified business sent a same-day operational update matching recent activity")
    elif action == "notify" and features.direct_mention >= 0.5 and features.urgency >= 0.5:
        parts.append("Direct mention with urgent operational content warrants an interrupt")
    elif action == "notify" and features.verified_business:
        parts.append("Verified business context with a time-sensitive operational signal")
    elif action == "notify" and features.sender_is_group_admin:
        parts.append("Group admin sent a high-priority update for this recipient")
    elif action == "notify":
        parts.append("Trusted sender with urgency signals warrants an immediate interrupt")
    elif action == "digest" and features.quiet_hours and features.urgency < 0.7:
        parts.append("Useful update deferred because it arrived during quiet hours")
    elif action == "digest" and features.same_entity_history_count > 0:
        parts.append("Useful for later review based on this recipient's channel history")
    elif action == "digest" and message_type == "promotion":
        parts.append("Promotional content is relevant enough to keep but not interrupt-worthy")
    elif action == "digest":
        parts.append("Useful for later review but not interrupt-worthy given current signals")
    else:
        parts.append(
            f"{message_type.replace('_', ' ')} content routed to {action} from channel signals"
        )

    if evidence != "none":
        parts.append("supported by prior recipient reactions")
    if features.media_uninterpreted:
        if media_source == "missing":
            parts.append("referenced media file was unavailable so OCR/ASR stayed empty")
        else:
            parts.append("media was present but left uninterpreted offline")

    reason = ". ".join(parts).strip()
    if not reason.endswith("."):
        reason += "."
    return reason[:280]


def calibrate_confidence(
    *,
    action: str,
    raw_action: str,
    priority: float,
    forced_mute: bool,
    media_uninterpreted: bool,
    risk_score: float,
    evidence: str,
    features: MessageFeatures,
    clamped: bool = False,
) -> float:
    if forced_mute:
        base = 0.82 + min(0.12, risk_score * 0.1)
    elif action == "notify":
        base = 0.55 + min(0.30, max(0.0, priority) * 0.5)
    elif action == "digest":
        base = 0.50 + min(0.25, abs(priority) * 0.4)
    else:
        base = 0.55 + min(0.25, abs(priority) * 0.35)

    if action != raw_action or clamped:
        base -= 0.08
    if media_uninterpreted:
        base -= 0.12
    if evidence != "none":
        base += 0.05
    elif features.same_entity_history_count == 0:
        base -= 0.04
    if features.domain_mismatch and action != "mute":
        base -= 0.05
    return round(max(0.35, min(0.95, base)), 4)


def filter_evidence_ids(
    proposed: str | Sequence[str] | None,
    allowed: Sequence[str],
) -> str:
    allowed_set = {item for item in allowed if item and item != "none"}
    formatted = format_evidence(proposed)
    if formatted == "none":
        return "none"
    kept = [part for part in formatted.split(";") if part in allowed_set]
    return format_evidence(kept)


def deterministic_decide(ctx: DecisionContext) -> DecisionResult:
    """No-API fallback: priority action + signal-based type/reason/confidence."""
    action, clamped = apply_safety_constraints(
        ctx.priority.action,
        forced_mute=ctx.risk.forced_mute,
        notify_ceiling_active=ctx.priority.technical_ceiling_active,
    )
    # Priority already applied hard mute / ceiling; keep safer of the two.
    action = safer_action(action, ctx.priority.action)
    if ctx.risk.forced_mute:
        action = "mute"
        clamped = True
    else:
        # Soft uplift: avoid muting weak-positive / domain-suspicious-but-not-hard
        # cases that still deserve a digest glance.
        text = ctx.content.joined_text()
        promo_like = bool(_PROMO_RE.search(text))
        greeting_like = bool(_GREETING_RE.search(text) and len(text) < 120)
        phishing = bool(_PHISHING_SIGNALS.intersection(ctx.risk.scam_signals))
        if action == "mute" and not phishing and ctx.risk.risk_score < 0.75:
            if greeting_like:
                action = "digest"
                clamped = True
            elif (
                ctx.features.domain_mismatch
                and promo_like
                and ctx.features.personal_relevance >= 0.35
            ):
                action = "digest"
                clamped = True
            elif (
                ctx.features.verified_business
                and not ctx.features.domain_mismatch
                and ctx.features.personal_relevance >= 0.45
                and ctx.risk.risk_score < 0.35
            ):
                action = "digest"
                clamped = True

    text = ctx.content.joined_text()
    message_type = infer_message_type(
        text=text,
        action=action,
        conversation_type=ctx.message.conversation_type,
        forced_mute=ctx.risk.forced_mute,
        scam_signals=ctx.risk.scam_signals,
        forwarded_count=ctx.message.forwarded_count,
        injection_hit=bool(ctx.risk.injection_hits),
    )
    evidence = filter_evidence_ids(list(ctx.allowed_evidence_ids), ctx.allowed_evidence_ids)
    reason = build_reason(
        action=action,
        message_type=message_type,
        risk_reasons=ctx.risk.reasons,
        features=ctx.features,
        evidence=evidence,
        media_source=ctx.media_source,
    )
    confidence = calibrate_confidence(
        action=action,
        raw_action=ctx.priority.raw_action,
        priority=ctx.priority.priority,
        forced_mute=ctx.risk.forced_mute,
        media_uninterpreted=ctx.features.media_uninterpreted,
        risk_score=ctx.risk.risk_score,
        evidence=evidence,
        features=ctx.features,
        clamped=clamped,
    )
    return DecisionResult(
        action=action,
        message_type=message_type,
        reason=reason,
        confidence=confidence,
        evidence_message_ids=evidence,
        source="fallback",
        clamped_by_safety=clamped,
    )


def _context_payload(ctx: DecisionContext) -> dict[str, Any]:
    features = ctx.features
    return {
        "message_id": ctx.message.message_id,
        "conversation_type": ctx.message.conversation_type,
        "forwarded_count": ctx.message.forwarded_count,
        "content": {
            "message_text": prompts.wrap_untrusted_text(
                "MESSAGE_TEXT", ctx.content.message_text
            ),
            "ocr_text": prompts.wrap_untrusted_text("OCR_TEXT", ctx.content.ocr_text),
            "asr_transcript": prompts.wrap_untrusted_text(
                "ASR_TRANSCRIPT", ctx.content.asr_transcript
            ),
            "caption": prompts.wrap_untrusted_text("CAPTION", ctx.content.caption),
            "media_type": ctx.content.media_type,
            "media_id": ctx.content.media_id,
        },
        "features": {
            "urgency": features.urgency,
            "direct_mention": features.direct_mention,
            "sender_trust": features.sender_trust,
            "personal_relevance": features.personal_relevance,
            "positive_history": features.positive_history,
            "repetition": features.repetition,
            "low_engagement": features.low_engagement,
            "quiet_hour_cost": features.quiet_hour_cost,
            "domain_mismatch": features.domain_mismatch,
            "sender_is_group_admin": features.sender_is_group_admin,
            "muted_group": features.muted_group,
            "opted_out": features.opted_out,
            "verified_business": features.verified_business,
            "quiet_hours": features.quiet_hours,
            "media_uninterpreted": features.media_uninterpreted,
            "signals": list(features.signals),
            "channel_open_rate": features.channel_open_rate,
            "channel_dismiss_rate": features.channel_dismiss_rate,
        },
        "safety": {
            "risk_score": ctx.risk.risk_score,
            "forced_mute": ctx.risk.forced_mute,
            "scam_signals": list(ctx.risk.scam_signals),
            "notify_ceiling_active": ctx.priority.technical_ceiling_active,
            "priority_action": ctx.priority.action,
            "priority_score": ctx.priority.priority,
        },
        "allowed_evidence_ids": list(ctx.allowed_evidence_ids) or ["none"],
        "evidence_summaries": [asdict(item) for item in ctx.evidence_summaries],
    }


def validate_decision_payload(
    payload: Mapping[str, Any],
    *,
    allowed_evidence_ids: Sequence[str],
) -> Optional[dict[str, Any]]:
    """Return cleaned fields or None if the payload is unusable."""
    action = str(payload.get("action") or "").strip().lower()
    message_type = str(payload.get("message_type") or "").strip().lower()
    reason = str(payload.get("reason") or "").strip()
    if action not in ACTIONS or message_type not in MESSAGE_TYPES:
        return None
    if not reason or _GENERIC_REASON_RE.match(reason):
        return None
    if len(reason) > 280:
        reason = reason[:280]
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        return None
    if not 0.0 <= confidence <= 1.0:
        return None
    evidence = filter_evidence_ids(payload.get("evidence_message_ids"), allowed_evidence_ids)
    return {
        "action": action,
        "message_type": message_type,
        "reason": reason if reason.endswith(".") else reason + ".",
        "confidence": round(confidence, 4),
        "evidence_message_ids": evidence,
    }


def _openai_decide(ctx: DecisionContext, *, model: str) -> Optional[DecisionResult]:
    api_key = os.environ.get(ENV_API_KEY, "")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package missing; using deterministic decision fallback")
        return None

    client = OpenAI(api_key=api_key)
    payload = json.dumps(_context_payload(ctx), ensure_ascii=True, sort_keys=True)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompts.DECISION_SYSTEM_PROMPT},
            {"role": "user", "content": prompts.decision_user_prompt(payload)},
        ],
    )
    raw = _extract_json_object(response.choices[0].message.content or "")
    cleaned = validate_decision_payload(
        raw, allowed_evidence_ids=ctx.allowed_evidence_ids
    )
    if cleaned is None:
        return None

    action, clamped = apply_safety_constraints(
        cleaned["action"],
        forced_mute=ctx.risk.forced_mute,
        notify_ceiling_active=ctx.priority.technical_ceiling_active,
    )
    message_type = cleaned["message_type"]
    if ctx.risk.forced_mute and message_type not in {"scam", "spam"}:
        message_type = infer_message_type(
            text=ctx.content.joined_text(),
            action="mute",
            conversation_type=ctx.message.conversation_type,
            forced_mute=True,
            scam_signals=ctx.risk.scam_signals,
            forwarded_count=ctx.message.forwarded_count,
            injection_hit=bool(ctx.risk.injection_hits),
        )
    evidence = cleaned["evidence_message_ids"]
    if action != cleaned["action"]:
        # Re-filter is already subset; keep unless empty after clamp to mute.
        evidence = filter_evidence_ids(evidence, ctx.allowed_evidence_ids)

    confidence = cleaned["confidence"]
    if clamped or action != cleaned["action"]:
        confidence = min(confidence, calibrate_confidence(
            action=action,
            raw_action=ctx.priority.raw_action,
            priority=ctx.priority.priority,
            forced_mute=ctx.risk.forced_mute,
            media_uninterpreted=ctx.features.media_uninterpreted,
            risk_score=ctx.risk.risk_score,
            evidence=evidence,
            features=ctx.features,
            clamped=True,
        ))

    reason = cleaned["reason"]
    if clamped and "safety" not in reason.lower() and "mute" not in reason.lower():
        reason = (
            f"{reason.rstrip('.')} Safety gate forced a safer {action} outcome."
        )[:280]

    return DecisionResult(
        action=action,
        message_type=message_type,
        reason=reason,
        confidence=round(float(confidence), 4),
        evidence_message_ids=evidence,
        source="openai",
        clamped_by_safety=clamped or action != cleaned["action"],
    )


def decide(ctx: DecisionContext, *, provider: Optional[str] = None) -> DecisionResult:
    """Produce a validated decision; always falls back deterministically."""
    mode = (provider or os.environ.get(ENV_DECISION_PROVIDER) or "auto").strip().lower()
    if mode == "auto":
        mode = "openai" if os.environ.get(ENV_API_KEY) else "offline"

    if mode == "openai":
        model = os.environ.get(ENV_DECISION_MODEL, DEFAULT_DECISION_MODEL)
        try:
            result = _openai_decide(ctx, model=model)
            if result is not None:
                return result
            logger.warning("model decision invalid/unavailable; using deterministic fallback")
        except Exception as exc:  # noqa: BLE001 - keep router runnable
            logger.warning(
                "model decision failed (%s); using deterministic fallback",
                type(exc).__name__,
            )

    return deterministic_decide(ctx)
