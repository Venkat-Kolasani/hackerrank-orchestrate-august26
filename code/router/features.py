"""Deterministic recipient-specific feature engine.

Joins ``message_history`` to ``message_events`` by ``(user_id, message_id)``
and derives auditable signals from users, groups, memberships, businesses,
user-business history, and daily notification load.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Optional

from .data import Dataset
from .types import ContentSummary, MessageEventRecord, MessageRecord

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_URGENT_RE = re.compile(
    r"\b("
    r"urgent|asap|immediately|right\s+now|today\s+only|same[-\s]?day|"
    r"deadline|expires?\s+today|last\s+chance|before\s+\d|"
    r"emergency|evacuate|hospital|ambulance|power\s+cut|water\s+cut|"
    r"meeting\s+in\s+\d+|due\s+(today|tonight|now)"
    r")\b",
    re.IGNORECASE,
)
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
_MENTION_RE = re.compile(r"(?<!\w)@[\w.]+")
_TEMPLATE_NOISE = re.compile(r"\d+|https?://\S+|www\.\S+")


def _parse_hhmm(value: str) -> Optional[time]:
    parts = value.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return time(hour=hour, minute=minute)


def in_quiet_hours(window: str, when: Optional[datetime]) -> bool:
    """True when ``when`` is inside a DND window, including midnight-crossing."""
    if not window or when is None or "-" not in window:
        return False
    start_raw, end_raw = window.split("-", 1)
    start = _parse_hhmm(start_raw)
    end = _parse_hhmm(end_raw)
    if start is None or end is None:
        return False
    current = when.time().replace(second=0, microsecond=0)
    if start <= end:
        return start <= current < end
    return current >= start or current < end


def tokenize(text: str) -> list[str]:
    normalized = _TEMPLATE_NOISE.sub(" ", (text or "").lower())
    return _TOKEN_RE.findall(normalized)


def template_hash(text: str) -> str:
    tokens = tokenize(text)
    blob = " ".join(sorted(set(tokens)))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class HistoryItem:
    message: MessageRecord
    event: Optional[MessageEventRecord]

    @property
    def message_id(self) -> str:
        return self.message.message_id

    def supports_notify(self) -> bool:
        if not self.event:
            return False
        if self.event.message_reported or self.event.muted_after_message:
            return False
        if self.event.message_replied:
            return True
        if self.event.message_opened and not self.event.notification_dismissed:
            rt = self.event.reaction_time_minutes
            return rt is not None and rt <= 30
        return False

    def supports_digest(self) -> bool:
        if not self.event:
            return False
        if self.event.message_reported or self.event.muted_after_message:
            return False
        return self.event.message_opened and not self.event.message_replied

    def supports_mute(self) -> bool:
        if not self.event:
            return False
        return (
            self.event.notification_dismissed
            or self.event.muted_after_message
            or self.event.message_reported
        )


@dataclass
class MessageFeatures:
    """Auditable feature bundle for one incoming message."""

    urgency: float = 0.0
    direct_mention: float = 0.0
    sender_trust: float = 0.0
    personal_relevance: float = 0.0
    positive_history: float = 0.0
    repetition: float = 0.0
    low_engagement: float = 0.0
    quiet_hour_cost: float = 0.0
    domain_mismatch: bool = False
    sender_is_group_admin: bool = False
    muted_group: bool = False
    opted_out: bool = False
    verified_business: bool = False
    quiet_hours: bool = False
    media_uninterpreted: bool = False
    media_missing: bool = False
    user_reports_30d: int = 0
    account_age_days: Optional[int] = None
    domain_age_days: Optional[int] = None
    notification_load: float = 0.0
    channel_open_rate: float = 0.0
    channel_dismiss_rate: float = 0.0
    channel_report_rate: float = 0.0
    same_entity_history_count: int = 0
    repeated_template: bool = False
    text: str = ""
    signals: list[str] = field(default_factory=list)
    history_for_user: list[HistoryItem] = field(default_factory=list)


def recipient_history(dataset: Dataset, user_id: str) -> list[HistoryItem]:
    """Join recipient history rows to events; stable newest-last order."""
    items: list[HistoryItem] = []
    for message in dataset.history_by_user.get(user_id, []):
        event = dataset.get_event(user_id, message.message_id)
        items.append(HistoryItem(message=message, event=event))
    return items


def same_entity(message: MessageRecord, other: MessageRecord) -> bool:
    if message.conversation_type == "group" and message.group_id:
        return other.group_id == message.group_id
    if message.conversation_type == "business" and message.business_id:
        return other.business_id == message.business_id
    if message.conversation_type == "personal" and message.sender_user_id:
        return (
            other.conversation_type == "personal"
            and other.sender_user_id == message.sender_user_id
        )
    return False


def _channel_stats(items: list[HistoryItem]) -> tuple[float, float, float, int]:
    if not items:
        return 0.0, 0.0, 0.0, 0
    opened = dismissed = reported = 0
    counted = 0
    for item in items:
        if not item.event:
            continue
        counted += 1
        opened += int(item.event.message_opened)
        dismissed += int(item.event.notification_dismissed or item.event.muted_after_message)
        reported += int(item.event.message_reported)
    if counted == 0:
        return 0.0, 0.0, 0.0, 0
    return opened / counted, dismissed / counted, reported / counted, counted


def _daily_load(dataset: Dataset, user_id: str, when: Optional[datetime]) -> float:
    if when is None:
        return 0.0
    day = when.date().isoformat()
    for row in dataset.daily_notification_summary:
        if row.user_id == user_id and row.date == day:
            sent = max(0, row.notifications_sent)
            dismissed = max(0, row.notifications_dismissed)
            # Normalize: 8+ notifications in a day is heavy.
            load = min(1.0, sent / 8.0)
            if sent > 0 and dismissed / sent >= 0.5:
                load = min(1.0, load + 0.2)
            return load
    # Fall back to recent 3-day average if exact day missing.
    recent = [
        row
        for row in dataset.daily_notification_summary
        if row.user_id == user_id
    ]
    if not recent:
        return 0.0
    # Prefer rows within 3 days of message time when parseable.
    scored: list[tuple[int, float]] = []
    for row in recent:
        try:
            row_day = datetime.strptime(row.date, "%Y-%m-%d").date()
        except ValueError:
            continue
        delta = abs((when.date() - row_day).days)
        if delta <= 3:
            scored.append((delta, min(1.0, row.notifications_sent / 8.0)))
    if not scored:
        return 0.0
    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def _direct_mention_score(text: str, user_id: str) -> float:
    if not text:
        return 0.0
    if _MENTION_RE.search(text):
        return 1.0
    if user_id and re.search(rf"(?<!\w){re.escape(user_id)}(?!\w)", text, re.IGNORECASE):
        return 0.8
    return 0.0


def _urgency_score(text: str) -> float:
    if not text:
        return 0.0
    if _URGENT_RE.search(text):
        return 0.85
    if _PAYMENT_RE.search(text) and re.search(
        r"\b(today|now|due|overdue|failed)\b", text, re.IGNORECASE
    ):
        return 0.65
    return 0.0


def _repetition_score(
    message: MessageRecord,
    history: list[HistoryItem],
) -> tuple[float, bool]:
    tokens = set(tokenize(message.message_text))
    if not tokens and not message.media_id:
        return 0.0, False
    msg_hash = template_hash(message.message_text)
    best = 0.0
    template_hit = False
    cutoff = None
    if message.created_at is not None:
        cutoff = message.created_at - timedelta(days=30)

    for item in history:
        other = item.message
        if cutoff and other.created_at and other.created_at < cutoff:
            continue
        if other.message_id == message.message_id:
            continue
        other_tokens = set(tokenize(other.message_text))
        sim = jaccard(tokens, other_tokens)
        if message.media_id and other.media_id and message.media_id == other.media_id:
            sim = max(sim, 0.9)
        if template_hash(other.message_text) == msg_hash and msg_hash:
            template_hit = True
            sim = max(sim, 0.95)
        # Stronger weight when prior reaction was dismiss/mute/report.
        if sim >= 0.55 and item.supports_mute():
            sim = min(1.0, sim + 0.15)
        best = max(best, sim)
    return clip01(best), template_hit


def compute_features(
    dataset: Dataset,
    message: MessageRecord,
    content: Optional[ContentSummary] = None,
) -> MessageFeatures:
    """Build recipient-specific features for ``message``.

    When ``content`` includes OCR/ASR channels, text-based signals use the
    joined untrusted content. ``media_uninterpreted`` is False only when media
    perception actually filled a media channel.
    """
    user = dataset.get_user(message.user_id)
    business = dataset.get_business(message.business_id)
    membership = dataset.get_group_member(message.group_id, message.user_id)
    sender_membership = dataset.get_group_member(
        message.group_id, message.sender_user_id or ""
    )
    relationship = dataset.get_user_business(message.user_id, message.business_id)
    media_ref = dataset.resolve_media(message.media_type, message.media_id)
    history = recipient_history(dataset, message.user_id)
    entity_history = [item for item in history if same_entity(message, item.message)]

    if content is None:
        content = ContentSummary(
            message_text=message.message_text or "",
            media_type=message.media_type,
            media_id=message.media_id,
        )
    text = content.joined_text()
    signals: list[str] = []

    domain_mismatch = bool(business.domain_mismatch) if business else False
    verified_business = bool(business and business.verified)
    muted_group = bool(membership and membership.group_muted_by_user)
    opted_out = bool(
        relationship
        and (
            relationship.promotions_opted_out_at is not None
            or not relationship.allows_promotions
        )
    )
    sender_is_admin = bool(sender_membership and sender_membership.role == "admin")
    quiet = in_quiet_hours(
        user.do_not_disturb_window if user else "",
        message.created_at,
    )

    urgency = _urgency_score(text)
    mention = _direct_mention_score(text, message.user_id)
    if mention >= 0.5:
        signals.append("direct_mention")
    if urgency >= 0.5:
        signals.append("urgency")

    # Sender trust
    sender_trust = 0.2
    if verified_business and not domain_mismatch:
        sender_trust = 0.75
        signals.append("verified_business")
    elif sender_is_admin:
        sender_trust = 0.7
        signals.append("sender_is_group_admin")
    elif message.conversation_type == "personal":
        # Prior positive personal replies raise trust.
        personal_items = [
            item
            for item in history
            if item.message.conversation_type == "personal"
            and item.message.sender_user_id == message.sender_user_id
        ]
        open_rate, dismiss_rate, _, count = _channel_stats(personal_items)
        if count:
            sender_trust = clip01(0.35 + open_rate * 0.45 - dismiss_rate * 0.35)
        else:
            sender_trust = 0.55
    account_age_days: Optional[int] = None
    domain_age_days: Optional[int] = None
    if business:
        account_age_days = business.account_age_days
        domain_age_days = business.domain_used_by_sender_age_days
        age_factor = clip01(business.account_age_days / 365.0)
        domain_age_factor = clip01(business.domain_used_by_sender_age_days / 365.0)
        sender_trust = clip01(sender_trust * 0.7 + 0.3 * min(age_factor, domain_age_factor))
        if business.account_age_days < 90 or business.domain_used_by_sender_age_days < 90:
            signals.append("young_sender_age")
        if business.user_reports_30d >= 10:
            sender_trust = min(sender_trust, 0.25)
            signals.append("high_report_volume")
    if domain_mismatch:
        sender_trust = min(sender_trust, 0.15)
        signals.append("domain_mismatch")

    # Personal relevance
    personal_relevance = 0.2
    if relationship:
        personal_relevance = clip01(
            0.30
            + min(0.45, relationship.activity_count_180d / 40.0)
            + min(0.20, relationship.messages_opened_30d / 20.0)
            - min(0.25, relationship.messages_dismissed_30d / 20.0)
        )
        why = (relationship.why_user_knows_account or "").lower()
        if any(token in why for token in ("opted_out", "ignored", "old_", "abandoned")):
            personal_relevance = min(personal_relevance, 0.25)
            signals.append("weak_business_relationship")
        elif any(
            token in why
            for token in ("active_", "recent_", "today", "booking", "order", "payment")
        ):
            personal_relevance = max(personal_relevance, 0.65)
            signals.append("active_business_relationship")
    if membership:
        member_activity = membership.messages_read_30d + membership.replies_sent_30d
        personal_relevance = max(
            personal_relevance,
            clip01(member_activity / 40.0),
        )
    if _PROMO_RE.search(text) and opted_out:
        personal_relevance = min(personal_relevance, 0.1)
    if opted_out:
        signals.append("promotions_opted_out")

    # Positive / low engagement from recipient + same-entity history
    open_rate, dismiss_rate, report_rate, entity_count = _channel_stats(entity_history)
    positive_history = 0.0
    if user:
        opens = user.messages_opened_30d
        replies = user.messages_replied_30d
        dismissals = user.notifications_dismissed_30d
        total = max(1, opens + dismissals)
        positive_history = clip01((opens + 2 * replies) / (total + 2 * replies))
    if entity_count:
        positive_history = clip01(0.5 * positive_history + 0.5 * open_rate)

    low_engagement = 0.0
    if muted_group:
        low_engagement = max(low_engagement, 0.7)
        signals.append("muted_group")
    if opted_out:
        low_engagement = max(low_engagement, 0.8)
    if entity_count:
        low_engagement = max(low_engagement, dismiss_rate)
        if report_rate > 0:
            low_engagement = max(low_engagement, clip01(0.5 + report_rate))
            signals.append("channel_reports")
    if user and user.notifications_dismissed_30d > user.messages_opened_30d:
        low_engagement = max(low_engagement, 0.45)

    notification_load = _daily_load(dataset, message.user_id, message.created_at)
    if notification_load >= 0.6:
        low_engagement = max(low_engagement, 0.35)
        signals.append("high_notification_load")

    repetition, repeated_template = _repetition_score(message, history)
    if repetition >= 0.7:
        signals.append("repetition")
    if repeated_template:
        signals.append("repeated_template")

    quiet_hour_cost = 0.0
    if quiet:
        quiet_hour_cost = 0.85 if urgency < 0.7 else 0.25
        signals.append("quiet_hours")

    media_channels_filled = bool(
        (content.ocr_text and content.ocr_text.strip())
        or (content.asr_transcript and content.asr_transcript.strip())
        or (content.caption and content.caption.strip())
    )
    media_uninterpreted = bool(message.media_type) and not media_channels_filled
    media_missing = bool(message.media_type and (media_ref is None or not media_ref.available))
    if media_uninterpreted and not (message.message_text or "").strip():
        urgency = min(urgency, 0.2)
        personal_relevance = min(personal_relevance, 0.25)
        signals.append("media_uninterpreted")
    elif media_uninterpreted:
        signals.append("media_uninterpreted")
    if media_missing:
        personal_relevance = min(personal_relevance, 0.2)
        signals.append("media_missing")

    return MessageFeatures(
        urgency=clip01(urgency),
        direct_mention=clip01(mention),
        sender_trust=clip01(sender_trust),
        personal_relevance=clip01(personal_relevance),
        positive_history=clip01(positive_history),
        repetition=clip01(repetition),
        low_engagement=clip01(low_engagement),
        quiet_hour_cost=clip01(quiet_hour_cost),
        domain_mismatch=domain_mismatch,
        sender_is_group_admin=sender_is_admin,
        muted_group=muted_group,
        opted_out=opted_out,
        verified_business=verified_business,
        quiet_hours=quiet,
        media_uninterpreted=media_uninterpreted,
        media_missing=media_missing,
        user_reports_30d=business.user_reports_30d if business else 0,
        account_age_days=account_age_days,
        domain_age_days=domain_age_days,
        notification_load=notification_load,
        channel_open_rate=open_rate,
        channel_dismiss_rate=dismiss_rate,
        channel_report_rate=report_rate,
        same_entity_history_count=entity_count,
        repeated_template=repeated_template,
        text=text,
        signals=signals,
        history_for_user=history,
    )
