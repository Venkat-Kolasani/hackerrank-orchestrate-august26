"""Shared router datatypes and allowed enums."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

ACTIONS = ("notify", "digest", "mute")
MESSAGE_TYPES = (
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
)
CONVERSATION_TYPES = ("personal", "group", "business")
MEDIA_TYPES = ("", "image", "voice")

OUTPUT_COLUMNS = (
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
)


@dataclass(frozen=True)
class ContentSummary:
    """Normalized multimodal content consumed by safety and decision stages.

    All fields are untrusted data. Injection scanning must cover every
    non-empty channel (original text, OCR, ASR), not just message_text.
    """

    message_text: str = ""
    ocr_text: str = ""
    asr_transcript: str = ""
    caption: str = ""
    media_type: Optional[str] = None
    media_id: Optional[str] = None

    def channels(self) -> dict[str, str]:
        """Return non-empty content channels keyed by source name."""
        mapping = {
            "message_text": self.message_text,
            "ocr_text": self.ocr_text,
            "asr_transcript": self.asr_transcript,
            "caption": self.caption,
        }
        return {name: text for name, text in mapping.items() if text and text.strip()}

    def joined_text(self) -> str:
        """Concatenate all channels for scanners that need a single blob."""
        return "\n".join(self.channels().values())


@dataclass(frozen=True)
class InjectionHit:
    """One deterministic match of an override / label-bait phrase."""

    pattern_id: str
    category: str
    channel: str
    matched_text: str
    weight: float


@dataclass
class RiskAssessment:
    """Explainable risk breakdown produced before any LLM call."""

    risk_score: float
    base_score: float
    injection_score: float
    forced_mute: bool
    metadata_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    injection_hits: list[InjectionHit] = field(default_factory=list)
    scam_signals: list[str] = field(default_factory=list)
    metadata_signals: list[str] = field(default_factory=list)
    metadata_parts: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MessageRecord:
    """One incoming or historical message row."""

    message_id: str
    user_id: str
    conversation_type: str
    group_id: Optional[str]
    business_id: Optional[str]
    sender_user_id: Optional[str]
    created_at: Optional[datetime]
    message_text: str
    media_type: Optional[str]
    media_id: Optional[str]
    forwarded_count: int


@dataclass(frozen=True)
class UserRecord:
    user_id: str
    do_not_disturb_window: str
    messages_opened_30d: int
    messages_replied_30d: int
    notifications_dismissed_30d: int
    messages_reported_30d: int


@dataclass(frozen=True)
class GroupRecord:
    group_id: str
    group_name: str
    group_type: str
    member_count: int
    admin_count: int
    created_at: Optional[datetime]
    messages_30d: int


@dataclass(frozen=True)
class GroupMemberRecord:
    group_id: str
    user_id: str
    role: str
    joined_at: Optional[datetime]
    messages_sent_30d: int
    messages_read_30d: int
    replies_sent_30d: int
    notifications_dismissed_30d: int
    group_muted_by_user: bool


@dataclass(frozen=True)
class BusinessRecord:
    business_id: str
    display_name: str
    brand_name: str
    category: str
    verified: bool
    official_domain: str
    domain_used_by_sender: str
    account_age_days: int
    messages_sent_30d: int
    user_reports_30d: int
    domain_used_by_sender_age_days: int

    @property
    def domain_mismatch(self) -> bool:
        official = (self.official_domain or "").strip().lower()
        used = (self.domain_used_by_sender or "").strip().lower()
        if not official or not used:
            return False
        return official != used


@dataclass(frozen=True)
class UserBusinessHistoryRecord:
    user_id: str
    business_id: str
    why_user_knows_account: str
    last_activity_at: Optional[datetime]
    allows_promotions: bool
    promotions_opted_out_at: Optional[datetime]
    activity_count_180d: int
    messages_opened_30d: int
    messages_dismissed_30d: int
    messages_replied_30d: int
    last_reply_at: Optional[datetime]


@dataclass(frozen=True)
class MessageEventRecord:
    user_id: str
    message_id: str
    message_opened: bool
    message_replied: bool
    reaction_time_minutes: Optional[float]
    notification_dismissed: bool
    muted_after_message: bool
    message_reported: bool


@dataclass(frozen=True)
class DailyNotificationSummaryRecord:
    user_id: str
    date: str
    notifications_sent: int
    notifications_dismissed: int


@dataclass(frozen=True)
class MediaRef:
    media_id: str
    file_path: str
    absolute_path: Optional[str]
    available: bool


@dataclass(frozen=True)
class Prediction:
    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: str

    def as_row(self) -> dict[str, str]:
        return {
            "message_id": self.message_id,
            "action": self.action,
            "message_type": self.message_type,
            "reason": self.reason,
            "confidence": f"{self.confidence:.4f}".rstrip("0").rstrip(".")
            if "." in f"{self.confidence:.4f}"
            else f"{self.confidence:.4f}",
            "evidence_message_ids": self.evidence_message_ids,
        }
