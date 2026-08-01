"""Router package for the Message Notification Router."""

from .injection import has_injection, injection_score, scan_content, scan_text
from .safety import (
    DOMAIN_MISMATCH_WEIGHT,
    FORWARD_WEIGHT,
    HARD_MUTE_THRESHOLD,
    REPORT_WEIGHT,
    assess_risk,
    should_force_mute,
)
from .types import ContentSummary, InjectionHit, RiskAssessment

__all__ = [
    "DOMAIN_MISMATCH_WEIGHT",
    "FORWARD_WEIGHT",
    "HARD_MUTE_THRESHOLD",
    "REPORT_WEIGHT",
    "ContentSummary",
    "InjectionHit",
    "RiskAssessment",
    "assess_risk",
    "has_injection",
    "injection_score",
    "scan_content",
    "scan_text",
    "should_force_mute",
]
