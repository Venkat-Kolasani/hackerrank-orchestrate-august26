"""Load and validate participant-facing dataset files.

Production code must never load ``sample_messages.csv``. That file belongs
only in ``code/evaluation/``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .types import (
    BusinessRecord,
    DailyNotificationSummaryRecord,
    GroupMemberRecord,
    GroupRecord,
    MediaRef,
    MessageEventRecord,
    MessageRecord,
    UserBusinessHistoryRecord,
    UserRecord,
)

TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
)


def _blank_to_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = value.strip()
    return text if text else None


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    return default


def _parse_int(value: Optional[str], default: int = 0) -> int:
    text = _blank_to_none(value)
    if text is None:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def _parse_float(value: Optional[str]) -> Optional[float]:
    text = _blank_to_none(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    text = _blank_to_none(value)
    if text is None:
        return None
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset file missing: {path}")
    return path


def _read_dicts(path: Path) -> list[dict[str, str]]:
    with _require_file(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _message_from_row(row: dict[str, str]) -> MessageRecord:
    media_type = _blank_to_none(row.get("media_type"))
    return MessageRecord(
        message_id=_blank_to_none(row.get("message_id")) or "",
        user_id=_blank_to_none(row.get("user_id")) or "",
        conversation_type=(_blank_to_none(row.get("conversation_type")) or "personal"),
        group_id=_blank_to_none(row.get("group_id")),
        business_id=_blank_to_none(row.get("business_id")),
        sender_user_id=_blank_to_none(row.get("sender_user_id")),
        created_at=_parse_datetime(row.get("created_at")),
        message_text=(row.get("message_text") or "").strip(),
        media_type=media_type,
        media_id=_blank_to_none(row.get("media_id")),
        forwarded_count=_parse_int(row.get("forwarded_count"), 0),
    )


@dataclass
class Dataset:
    """In-memory indexes for participant-facing CSVs and media refs."""

    root: Path
    messages: list[MessageRecord] = field(default_factory=list)
    users: dict[str, UserRecord] = field(default_factory=dict)
    groups: dict[str, GroupRecord] = field(default_factory=dict)
    group_members: dict[tuple[str, str], GroupMemberRecord] = field(default_factory=dict)
    businesses: dict[str, BusinessRecord] = field(default_factory=dict)
    user_business_history: dict[tuple[str, str], UserBusinessHistoryRecord] = field(
        default_factory=dict
    )
    message_history: list[MessageRecord] = field(default_factory=list)
    message_events: dict[tuple[str, str], MessageEventRecord] = field(default_factory=dict)
    history_by_user: dict[str, list[MessageRecord]] = field(default_factory=dict)
    images: dict[str, MediaRef] = field(default_factory=dict)
    voice_notes: dict[str, MediaRef] = field(default_factory=dict)
    daily_notification_summary: list[DailyNotificationSummaryRecord] = field(
        default_factory=list
    )
    media_warnings: list[str] = field(default_factory=list)

    def get_user(self, user_id: str) -> Optional[UserRecord]:
        return self.users.get(user_id)

    def get_group(self, group_id: Optional[str]) -> Optional[GroupRecord]:
        return self.groups.get(group_id) if group_id else None

    def get_group_member(
        self, group_id: Optional[str], user_id: str
    ) -> Optional[GroupMemberRecord]:
        if not group_id:
            return None
        return self.group_members.get((group_id, user_id))

    def get_business(self, business_id: Optional[str]) -> Optional[BusinessRecord]:
        return self.businesses.get(business_id) if business_id else None

    def get_user_business(
        self, user_id: str, business_id: Optional[str]
    ) -> Optional[UserBusinessHistoryRecord]:
        if not business_id:
            return None
        return self.user_business_history.get((user_id, business_id))

    def get_event(self, user_id: str, message_id: str) -> Optional[MessageEventRecord]:
        return self.message_events.get((user_id, message_id))

    def resolve_media(self, media_type: Optional[str], media_id: Optional[str]) -> Optional[MediaRef]:
        if not media_id:
            return None
        if media_type == "image":
            return self.images.get(media_id)
        if media_type == "voice":
            return self.voice_notes.get(media_id)
        return self.images.get(media_id) or self.voice_notes.get(media_id)


def _load_media_map(
    rows: Iterable[dict[str, str]],
    *,
    id_field: str,
    dataset_root: Path,
    kind: str,
) -> tuple[dict[str, MediaRef], list[str]]:
    mapping: dict[str, MediaRef] = {}
    warnings: list[str] = []
    for row in rows:
        media_id = _blank_to_none(row.get(id_field))
        rel = _blank_to_none(row.get("file_path"))
        if not media_id or not rel:
            continue
        absolute = (dataset_root / rel).resolve()
        available = absolute.is_file()
        if not available:
            warnings.append(f"{kind} {media_id} missing at {rel}")
        mapping[media_id] = MediaRef(
            media_id=media_id,
            file_path=rel,
            absolute_path=str(absolute) if available else None,
            available=available,
        )
    return mapping, warnings


def load_dataset(
    dataset_root: Path | str,
    *,
    messages_file: str = "messages.csv",
) -> Dataset:
    """Load participant-facing files under ``dataset_root``.

    Parameters
    ----------
    messages_file:
        Basename of the message input CSV. Evaluation may pass an alternate
        input file name, but never the labeled sample columns as features.
    """
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    messages = [_message_from_row(row) for row in _read_dicts(root / messages_file)]
    if not messages:
        raise ValueError(f"No messages found in {root / messages_file}")
    for msg in messages:
        if not msg.message_id or not msg.user_id:
            raise ValueError(
                f"Message row missing message_id/user_id near {msg.message_id!r}"
            )

    users = {
        row["user_id"]: UserRecord(
            user_id=row["user_id"],
            do_not_disturb_window=(row.get("do_not_disturb_window") or "").strip(),
            messages_opened_30d=_parse_int(row.get("messages_opened_30d")),
            messages_replied_30d=_parse_int(row.get("messages_replied_30d")),
            notifications_dismissed_30d=_parse_int(row.get("notifications_dismissed_30d")),
            messages_reported_30d=_parse_int(row.get("messages_reported_30d")),
        )
        for row in _read_dicts(root / "users.csv")
        if _blank_to_none(row.get("user_id"))
    }

    groups = {
        row["group_id"]: GroupRecord(
            group_id=row["group_id"],
            group_name=(row.get("group_name") or "").strip(),
            group_type=(row.get("group_type") or "").strip(),
            member_count=_parse_int(row.get("member_count")),
            admin_count=_parse_int(row.get("admin_count")),
            created_at=_parse_datetime(row.get("created_at")),
            messages_30d=_parse_int(row.get("messages_30d")),
        )
        for row in _read_dicts(root / "groups.csv")
        if _blank_to_none(row.get("group_id"))
    }

    group_members: dict[tuple[str, str], GroupMemberRecord] = {}
    for row in _read_dicts(root / "group_members.csv"):
        group_id = _blank_to_none(row.get("group_id"))
        user_id = _blank_to_none(row.get("user_id"))
        if not group_id or not user_id:
            continue
        group_members[(group_id, user_id)] = GroupMemberRecord(
            group_id=group_id,
            user_id=user_id,
            role=(row.get("role") or "member").strip().lower(),
            joined_at=_parse_datetime(row.get("joined_at")),
            messages_sent_30d=_parse_int(row.get("messages_sent_30d")),
            messages_read_30d=_parse_int(row.get("messages_read_30d")),
            replies_sent_30d=_parse_int(row.get("replies_sent_30d")),
            notifications_dismissed_30d=_parse_int(
                row.get("notifications_dismissed_30d")
            ),
            group_muted_by_user=_parse_bool(row.get("group_muted_by_user")),
        )

    businesses = {
        row["business_id"]: BusinessRecord(
            business_id=row["business_id"],
            display_name=(row.get("display_name") or "").strip(),
            brand_name=(row.get("brand_name") or "").strip(),
            category=(row.get("category") or "").strip(),
            verified=_parse_bool(row.get("verified")),
            official_domain=(row.get("official_domain") or "").strip(),
            domain_used_by_sender=(row.get("domain_used_by_sender") or "").strip(),
            account_age_days=_parse_int(row.get("account_age_days")),
            messages_sent_30d=_parse_int(row.get("messages_sent_30d")),
            user_reports_30d=_parse_int(row.get("user_reports_30d")),
            domain_used_by_sender_age_days=_parse_int(
                row.get("domain_used_by_sender_age_days")
            ),
        )
        for row in _read_dicts(root / "business_accounts.csv")
        if _blank_to_none(row.get("business_id"))
    }

    user_business_history: dict[tuple[str, str], UserBusinessHistoryRecord] = {}
    for row in _read_dicts(root / "user_business_history.csv"):
        user_id = _blank_to_none(row.get("user_id"))
        business_id = _blank_to_none(row.get("business_id"))
        if not user_id or not business_id:
            continue
        user_business_history[(user_id, business_id)] = UserBusinessHistoryRecord(
            user_id=user_id,
            business_id=business_id,
            why_user_knows_account=(row.get("why_user_knows_account") or "").strip(),
            last_activity_at=_parse_datetime(row.get("last_activity_at")),
            allows_promotions=_parse_bool(row.get("allows_promotions"), default=True),
            promotions_opted_out_at=_parse_datetime(row.get("promotions_opted_out_at")),
            activity_count_180d=_parse_int(row.get("activity_count_180d")),
            messages_opened_30d=_parse_int(row.get("messages_opened_30d")),
            messages_dismissed_30d=_parse_int(row.get("messages_dismissed_30d")),
            messages_replied_30d=_parse_int(row.get("messages_replied_30d")),
            last_reply_at=_parse_datetime(row.get("last_reply_at")),
        )

    history = [_message_from_row(row) for row in _read_dicts(root / "message_history.csv")]
    history_by_user: dict[str, list[MessageRecord]] = {}
    for item in history:
        history_by_user.setdefault(item.user_id, []).append(item)
    for items in history_by_user.values():
        items.sort(key=lambda m: m.created_at or datetime.min)

    events: dict[tuple[str, str], MessageEventRecord] = {}
    for row in _read_dicts(root / "message_events.csv"):
        user_id = _blank_to_none(row.get("user_id"))
        message_id = _blank_to_none(row.get("message_id"))
        if not user_id or not message_id:
            continue
        events[(user_id, message_id)] = MessageEventRecord(
            user_id=user_id,
            message_id=message_id,
            message_opened=_parse_bool(row.get("message_opened")),
            message_replied=_parse_bool(row.get("message_replied")),
            reaction_time_minutes=_parse_float(row.get("reaction_time_minutes")),
            notification_dismissed=_parse_bool(row.get("notification_dismissed")),
            muted_after_message=_parse_bool(row.get("muted_after_message")),
            message_reported=_parse_bool(row.get("message_reported")),
        )

    images, image_warnings = _load_media_map(
        _read_dicts(root / "images.csv"),
        id_field="image_id",
        dataset_root=root,
        kind="image",
    )
    voice_notes, voice_warnings = _load_media_map(
        _read_dicts(root / "voice_notes.csv"),
        id_field="voice_note_id",
        dataset_root=root,
        kind="voice_note",
    )

    daily = [
        DailyNotificationSummaryRecord(
            user_id=row["user_id"],
            date=(row.get("date") or "").strip(),
            notifications_sent=_parse_int(row.get("notifications_sent")),
            notifications_dismissed=_parse_int(row.get("notifications_dismissed")),
        )
        for row in _read_dicts(root / "daily_notification_summary.csv")
        if _blank_to_none(row.get("user_id"))
    ]

    return Dataset(
        root=root,
        messages=messages,
        users=users,
        groups=groups,
        group_members=group_members,
        businesses=businesses,
        user_business_history=user_business_history,
        message_history=history,
        message_events=events,
        history_by_user=history_by_user,
        images=images,
        voice_notes=voice_notes,
        daily_notification_summary=daily,
        media_warnings=image_warnings + voice_warnings,
    )
