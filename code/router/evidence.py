"""Evidence retrieval for routing decisions.

Priority:
1. Same recipient + same conversation entity (group / business / sender)
2. Same recipient + matching template / category
3. Recipient-local similar content (deterministic token cosine)

Candidates are ranked by relevance and whether their observed reaction
supports the chosen action. Unrelated IDs are never used as filler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .data import Dataset
from .features import (
    HistoryItem,
    MessageFeatures,
    jaccard,
    recipient_history,
    same_entity,
    template_hash,
    tokenize,
)
from .output import format_evidence
from .types import MessageRecord

MAX_EVIDENCE = 3
MIN_SUPPORT_SCORE = 0.55


@dataclass(frozen=True)
class EvidenceCandidate:
    message_id: str
    score: float
    tier: int
    reason: str


def _token_cosine(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    tf_a: dict[str, int] = {}
    tf_b: dict[str, int] = {}
    for token in a:
        tf_a[token] = tf_a.get(token, 0) + 1
    for token in b:
        tf_b[token] = tf_b.get(token, 0) + 1
    # Simple TF cosine (no global IDF) for deterministic recipient-local ranking.
    dot = 0.0
    for token, weight in tf_a.items():
        if token in tf_b:
            dot += weight * tf_b[token]
    norm_a = math.sqrt(sum(weight * weight for weight in tf_a.values()))
    norm_b = math.sqrt(sum(weight * weight for weight in tf_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _reaction_boost(item: HistoryItem, action: str) -> float:
    if action == "notify":
        if item.supports_notify():
            return 0.35
        if item.event and item.event.message_opened:
            return 0.1
        return -0.25 if item.supports_mute() else 0.0
    if action == "digest":
        if item.supports_digest():
            return 0.30
        if item.supports_notify():
            return 0.15
        return -0.20 if item.supports_mute() else 0.0
    if item.supports_mute():
        return 0.40
    if item.supports_notify():
        return -0.30
    return 0.0


def _category_match(message: MessageRecord, other: MessageRecord) -> float:
    score = 0.0
    if message.media_type and message.media_type == other.media_type:
        score += 0.15
    if message.media_id and message.media_id == other.media_id:
        score += 0.55
    if message.conversation_type == other.conversation_type:
        score += 0.1
    if message.message_text.strip() and template_hash(message.message_text) == template_hash(
        other.message_text
    ):
        score += 0.45
    return min(1.0, score)


def _supports_action(item: HistoryItem, action: str) -> bool:
    if action == "notify":
        return item.supports_notify()
    if action == "digest":
        return item.supports_digest() or item.supports_notify()
    return item.supports_mute()


def retrieve_evidence(
    dataset: Dataset,
    message: MessageRecord,
    action: str,
    features: Optional[MessageFeatures] = None,
    *,
    limit: int = MAX_EVIDENCE,
) -> str:
    """Return semicolon-separated evidence IDs or ``none``."""
    history = (
        features.history_for_user
        if features is not None
        else recipient_history(dataset, message.user_id)
    )
    if not history:
        return "none"

    query_tokens = tokenize(message.message_text)
    query_set = set(query_tokens)
    candidates: list[EvidenceCandidate] = []
    history_ids = {item.message_id for item in dataset.message_history}

    for item in history:
        other = item.message
        if other.message_id == message.message_id:
            continue
        if other.message_id not in history_ids:
            continue

        entity_match = same_entity(message, other)
        category = _category_match(message, other)
        cosine = _token_cosine(query_tokens, tokenize(other.message_text))
        overlap = jaccard(query_set, set(tokenize(other.message_text)))
        content_sim = max(cosine, overlap)
        if message.media_id and other.media_id and message.media_id == other.media_id:
            content_sim = max(content_sim, 0.9)

        if entity_match:
            tier = 1
            base = 0.55 + 0.25 * content_sim + 0.15 * category
        elif category >= 0.45 or content_sim >= 0.55:
            tier = 2 if category >= content_sim else 3
            base = 0.35 + 0.40 * max(category, content_sim)
        elif content_sim >= 0.40:
            tier = 3
            base = 0.30 + 0.35 * content_sim
        else:
            continue

        boost = _reaction_boost(item, action)
        score = base + boost
        supports = _supports_action(item, action)
        strong_same = entity_match and content_sim >= 0.5 and boost >= 0.0
        if not supports and not strong_same:
            continue
        if score < MIN_SUPPORT_SCORE:
            continue

        candidates.append(
            EvidenceCandidate(
                message_id=other.message_id,
                score=round(score, 4),
                tier=tier,
                reason=(
                    f"tier{tier}|sim={content_sim:.2f}|boost={boost:.2f}"
                    f"|support={int(supports)}"
                ),
            )
        )

    if not candidates:
        return "none"

    candidates.sort(key=lambda c: (-c.score, c.tier, c.message_id))
    chosen: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.message_id in seen:
            continue
        seen.add(candidate.message_id)
        chosen.append(candidate.message_id)
        if len(chosen) >= limit:
            break
    return format_evidence(chosen)
