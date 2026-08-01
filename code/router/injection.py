"""Deterministic prompt-injection scanner.

This module is the primary defense against messages that try to rewrite the
router's instructions. It does not depend on an LLM "treating content as data".
It scans every normalized content channel (message text, OCR, ASR, captions)
and returns weighted hits that feed directly into ``risk_score`` before any
model call.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from .types import ContentSummary, InjectionHit

# Per-hit weight. A single high-severity hit is enough to push past the hard
# mute threshold when combined with typical scam bait; two independent hits
# force mute on their own.
OVERRIDE_WEIGHT = 0.55
ROLEPLAY_WEIGHT = 0.45
LABEL_BAIT_WEIGHT = 0.60
FAKE_OUTPUT_WEIGHT = 0.65
DELIMITER_WEIGHT = 0.40

# Patterns are compiled once. Each entry is (pattern_id, category, weight, regex).
# Matching is case-insensitive. Patterns deliberately cover paraphrases of
# sample_msg_053 and common jailbreak / label-bait framings.
_PATTERN_SPECS: Sequence[tuple[str, str, float, str]] = (
    (
        "ignore_previous",
        "override",
        OVERRIDE_WEIGHT,
        r"\b(?:ignore|disregard|forget|override)\b.{0,40}\b(?:previous|prior|above|all)\b.{0,40}\b(?:instructions?|rules?|routing|prompts?|guidelines?|orders?)\b",
    ),
    (
        "ignore_routing_rules",
        "override",
        OVERRIDE_WEIGHT,
        r"\bignore\b.{0,30}\b(?:routing\s+)?rules?\b",
    ),
    (
        "new_instructions",
        "override",
        OVERRIDE_WEIGHT,
        r"\b(?:new|updated)\s+instructions?\b.{0,20}\b(?:are|follow|:)\b",
    ),
    (
        "system_role",
        "roleplay",
        ROLEPLAY_WEIGHT,
        r"(?:^|[\n\r\s\[\{])system\s*:",
    ),
    (
        "developer_role",
        "roleplay",
        ROLEPLAY_WEIGHT,
        r"(?:^|[\n\r\s\[\{])(?:developer|assistant)\s*:",
    ),
    (
        "you_are_now_admin",
        "roleplay",
        ROLEPLAY_WEIGHT,
        r"\byou\s+are\s+now\b.{0,30}\b(?:admin|administrator|root|system|router|classifier)\b",
    ),
    (
        "act_as_router",
        "roleplay",
        ROLEPLAY_WEIGHT,
        r"\b(?:act|behave|function)\s+as\b.{0,20}\b(?:router|classifier|notification\s+system)\b",
    ),
    (
        "mark_as_action",
        "label_bait",
        LABEL_BAIT_WEIGHT,
        r"\b(?:mark|classify|label|set|force|route)\b.{0,30}\b(?:this|the\s+message|it)\b.{0,30}\b(?:as|to)\b.{0,15}\b(?:notify|digest|mute)\b",
    ),
    (
        "must_be_action",
        "label_bait",
        LABEL_BAIT_WEIGHT,
        r"\b(?:must|should|always)\b.{0,20}\b(?:be|return|output)\b.{0,20}\b(?:notify|digest|mute)\b",
    ),
    (
        "action_equals",
        "label_bait",
        LABEL_BAIT_WEIGHT,
        r"[\"']?action[\"']?\s*(?:=|:|is|to)\s*[\"']?(?:notify|digest|mute)\b",
    ),
    (
        "fake_json_response",
        "fake_output",
        FAKE_OUTPUT_WEIGHT,
        r"\{\s*[\"'](?:action|message_type|confidence|role)[\"']\s*:",
    ),
    (
        "fake_system_role_json",
        "fake_output",
        FAKE_OUTPUT_WEIGHT,
        r"\{\s*[\"']role[\"']\s*:\s*[\"']system[\"']",
    ),
    (
        "fake_csv_row",
        "fake_output",
        FAKE_OUTPUT_WEIGHT,
        r"\b(?:notify|digest|mute)\s*,\s*(?:personal|urgent|event|payment|business_update|promotion|greeting|forward|spam|scam|unknown)\s*,",
    ),
    (
        "end_of_prompt_delimiter",
        "delimiter",
        DELIMITER_WEIGHT,
        r"(?:</?(?:system|instructions?|prompt)>|\[/?INST\]|<<SYS>>)",
    ),
)

_COMPILED = tuple(
    (pattern_id, category, weight, re.compile(regex, re.IGNORECASE | re.DOTALL))
    for pattern_id, category, weight, regex in _PATTERN_SPECS
)


def scan_text(text: str, *, channel: str = "message_text") -> list[InjectionHit]:
    """Scan a single text blob and return all injection hits."""
    if not text or not text.strip():
        return []

    hits: list[InjectionHit] = []
    seen: set[str] = set()
    for pattern_id, category, weight, regex in _COMPILED:
        match = regex.search(text)
        if not match:
            continue
        key = f"{channel}:{pattern_id}"
        if key in seen:
            continue
        seen.add(key)
        matched = match.group(0)
        matched = " ".join(matched.split())
        if len(matched) > 120:
            matched = matched[:117] + "..."
        hits.append(
            InjectionHit(
                pattern_id=pattern_id,
                category=category,
                channel=channel,
                matched_text=matched,
                weight=weight,
            )
        )
    return hits


def scan_content(content: ContentSummary) -> list[InjectionHit]:
    """Scan every non-empty normalized content channel.

    OCR and ASR are first-class attack surfaces: a poster with embedded
    override text is the same class of attack as sample_msg_053.
    """
    hits: list[InjectionHit] = []
    for channel, text in content.channels().items():
        hits.extend(scan_text(text, channel=channel))
    return hits


def injection_score(hits: Iterable[InjectionHit]) -> float:
    """Combine hit weights into a [0, 1] injection contribution.

    Uses saturating sum so multiple independent signals compound without
    needing an LLM. Cap at 1.0.
    """
    total = 0.0
    for hit in hits:
        total += hit.weight
        if total >= 1.0:
            return 1.0
    return round(total, 4)


def has_injection(hits: Sequence[InjectionHit]) -> bool:
    return bool(hits)
