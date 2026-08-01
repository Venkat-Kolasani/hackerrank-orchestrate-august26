"""Shared router datatypes used by safety and injection scanners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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
