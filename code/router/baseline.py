"""Deterministic offline router (Stages 1–5).

Stage 5: constrained contextual decision (model optional; safety clamps).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from .data import Dataset
from .decision import DecisionContext, EvidenceSummary, decide
from .evidence import retrieve_evidence
from .features import compute_features, in_quiet_hours
from .media import MediaInterpreter, build_media_interpreter, interpret_message
from .priority import DEFAULT_WEIGHTS, PriorityTerms, decide_action
from .safety import assess_risk
from .types import MessageRecord, Prediction

# Re-export for existing Stage 1 tests.
__all__ = ["in_quiet_hours", "route_dataset", "route_message"]


def _evidence_summaries(
    dataset: Dataset,
    message: MessageRecord,
    evidence: str,
    *,
    limit_chars: int = 120,
) -> list[EvidenceSummary]:
    if evidence == "none":
        return []
    summaries: list[EvidenceSummary] = []
    history = {
        item.message_id: item
        for item in dataset.history_by_user.get(message.user_id, [])
    }
    for message_id in evidence.split(";"):
        hist = history.get(message_id)
        if hist is None:
            continue
        event = dataset.get_event(message.user_id, message_id)
        reaction_bits: list[str] = []
        if event:
            if event.message_reported:
                reaction_bits.append("reported")
            if event.muted_after_message:
                reaction_bits.append("muted")
            if event.notification_dismissed:
                reaction_bits.append("dismissed")
            if event.message_replied:
                reaction_bits.append("replied")
            elif event.message_opened:
                reaction_bits.append("opened")
        snippet = (hist.message_text or "").replace("\n", " ").strip()
        if len(snippet) > limit_chars:
            snippet = snippet[: limit_chars - 3] + "..."
        summaries.append(
            EvidenceSummary(
                message_id=message_id,
                snippet=snippet,
                reaction=",".join(reaction_bits) or "unknown",
            )
        )
    return summaries


def route_message(
    dataset: Dataset,
    message: MessageRecord,
    interpreter: Optional[MediaInterpreter] = None,
) -> Prediction:
    """Route one message through perception → features → safety → decision."""
    media = interpreter or build_media_interpreter()
    content, perception = interpret_message(dataset, message, media)
    features = compute_features(dataset, message, content)
    risk = assess_risk(
        content,
        domain_mismatch=features.domain_mismatch,
        user_reports_30d=features.user_reports_30d,
        forwarded_count=message.forwarded_count,
        account_age_days=features.account_age_days,
        domain_age_days=features.domain_age_days,
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
            & {
                "otp_request",
                "credential_harvest",
                "urgent_payment_link",
                "wallet_verification",
                "urgency_pressure",
            }
        ),
        sender_is_group_admin=features.sender_is_group_admin,
    )
    priority = decide_action(
        terms,
        DEFAULT_WEIGHTS,
        hard_blocked_by_safety=risk.forced_mute,
    )

    evidence = retrieve_evidence(
        dataset,
        message,
        priority.action,
        features,
    )
    allowed_ids = [] if evidence == "none" else evidence.split(";")
    summaries = _evidence_summaries(dataset, message, evidence)

    result = decide(
        DecisionContext(
            message=message,
            content=content,
            features=features,
            risk=risk,
            priority=priority,
            allowed_evidence_ids=allowed_ids,
            evidence_summaries=summaries,
            media_source=perception.source,
        )
    )

    # If the final action differs, refresh evidence for the safer action.
    if result.action != priority.action:
        refreshed = retrieve_evidence(dataset, message, result.action, features)
        result = replace(result, evidence_message_ids=refreshed)

    return result.to_prediction(message.message_id)


def route_dataset(
    dataset: Dataset,
    interpreter: Optional[MediaInterpreter] = None,
) -> list[Prediction]:
    """Route every incoming message in deterministic input order."""
    media = interpreter or build_media_interpreter()
    return [route_message(dataset, message, interpreter=media) for message in dataset.messages]
