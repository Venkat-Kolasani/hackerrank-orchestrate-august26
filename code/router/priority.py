"""Interrupt priority: normalized terms, explicit weights, overrides.

Every continuous term is forced into [0, 1] before weighting. Positive terms
raise interrupt priority; penalty terms lower it.

Safety boundaries are not expressed as weight tweaks:

1. Hard mute from ``safety.py`` wins entirely (no priority notify/digest).
2. Notify ceiling: ``confirmed_risk >= TECHNICAL_RISK_FLOOR`` alone caps max
   action at digest — no named-indicator allowlist. Injection, keyword scam
   signals, and future report/forward features all feed that score; history
   cannot buy back notify once it clears the floor.
3. Compound mention override: direct_mention AND (urgent OR group admin)
   forces notify only when not safety-blocked and not under the notify
   ceiling. Bare, non-urgent mentions in dismissed groups stay digest —
   that is intentional personalization, not a gap. Do not raise
   ``direct_mention`` weight globally to chase those cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class PriorityTerms:
    """Interrupt signals.

    Continuous terms are clipped to [0, 1]. Boolean flags arm hard overrides /
    ceilings and are not part of the linear weighted sum.
    """

    urgency: float = 0.0
    direct_mention: float = 0.0
    sender_trust: float = 0.0
    personal_relevance: float = 0.0
    positive_history: float = 0.0
    repetition: float = 0.0
    low_engagement: float = 0.0
    quiet_hour_cost: float = 0.0
    confirmed_risk: float = 0.0
    # Explainability flags (reasons/evidence). Not required to arm the notify
    # ceiling — that keys off confirmed_risk alone (see technical_notify_ceiling_active).
    domain_mismatch: bool = False
    credential_otp_or_payment_request: bool = False
    # Compound mention-override input (not weighted)
    sender_is_group_admin: bool = False

    def __post_init__(self) -> None:
        for name in (
            "urgency",
            "direct_mention",
            "sender_trust",
            "personal_relevance",
            "positive_history",
            "repetition",
            "low_engagement",
            "quiet_hour_cost",
            "confirmed_risk",
        ):
            object.__setattr__(self, name, clip01(getattr(self, name)))


@dataclass(frozen=True)
class PriorityDecision:
    priority: float
    action: str
    raw_action: str
    technical_ceiling_active: bool
    technical_ceiling_applied: bool
    mention_override_active: bool
    mention_override_applied: bool
    hard_blocked_by_safety: bool
    technical_indicators: tuple[str, ...]


POSITIVE_TERMS = (
    "urgency",
    "direct_mention",
    "sender_trust",
    "personal_relevance",
    "positive_history",
)

PENALTY_TERMS = (
    "repetition",
    "low_engagement",
    "quiet_hour_cost",
    "confirmed_risk",
)

WEIGHT_REASONS: Mapping[str, str] = {
    "urgency": "Same-day operational deadlines are the main legitimate interrupt class in the samples.",
    "direct_mention": "Weak linear nudge only; compound override handles mention+urgency/admin cases.",
    "sender_trust": "Admin/verified/known senders are more often actionable, but trust alone is not urgency.",
    "personal_relevance": "Interest/relationship match separates useful digests from generic noise.",
    "positive_history": "Prior open/reply rates personalize without inventing urgency.",
    "repetition": "Near-duplicates the user already saw should not re-interrupt.",
    "low_engagement": "Repeated dismiss/mute/report on this channel is a strong anti-notify signal.",
    "quiet_hour_cost": "Non-critical interrupts during DND should wait; never overrides hard safety.",
    "confirmed_risk": "Residual scam/injection soft-risk must suppress interrupts, not compete with history.",
}

# Keep direct_mention at the same weight across profiles — do not raise it
# globally to chase muted-group mentions.
DEFAULT_WEIGHTS: Mapping[str, float] = {
    "urgency": 0.20,
    "direct_mention": 0.25,
    "sender_trust": 0.12,
    "personal_relevance": 0.10,
    "positive_history": 0.10,
    "repetition": 0.15,
    "low_engagement": 0.15,
    "quiet_hour_cost": 0.10,
    "confirmed_risk": 0.30,
}

# Risk-leaning profile: higher confirmed_risk penalty, mention weight unchanged.
MENTION_AND_RISK_WEIGHTS: Mapping[str, float] = {
    "urgency": 0.15,
    "direct_mention": 0.25,
    "sender_trust": 0.08,
    "personal_relevance": 0.07,
    "positive_history": 0.05,
    "repetition": 0.12,
    "low_engagement": 0.12,
    "quiet_hour_cost": 0.08,
    "confirmed_risk": 0.40,
}

HISTORY_HEAVY_WEIGHTS: Mapping[str, float] = {
    "urgency": 0.12,
    "direct_mention": 0.25,
    "sender_trust": 0.20,
    "personal_relevance": 0.18,
    "positive_history": 0.30,
    "repetition": 0.12,
    "low_engagement": 0.18,
    "quiet_hour_cost": 0.08,
    "confirmed_risk": 0.15,
}

NOTIFY_THRESHOLD = 0.45
# Borderline greetings / weak-positive history should stay visible in digest.
DIGEST_THRESHOLD = 0.10
# Data-driven notify ceiling floor from message_history × message_events
# calibration (Youden J on muted_after_message|message_reported vs rest).
# Recompute with: python code/evaluation/calibrate_risk_threshold.py
TECHNICAL_RISK_FLOOR = 0.18
# Urgency must clear this floor to arm the compound mention override.
# Below it, a bare @mention is only a weak linear term (often digest).
URGENCY_MENTION_FLOOR = 0.50
DIRECT_MENTION_FLOOR = 0.50


def technical_indicators(terms: PriorityTerms) -> tuple[str, ...]:
    flags: list[str] = []
    if terms.domain_mismatch:
        flags.append("domain_mismatch")
    if terms.credential_otp_or_payment_request:
        flags.append("credential_otp_or_payment_request")
    return tuple(flags)


def technical_notify_ceiling_active(terms: PriorityTerms) -> bool:
    """Arm notify ceiling from aggregated risk score alone.

    Do not require a named indicator allowlist. Injection, keyword scam
    signals, and (once wired) report/forward features all feed
    ``confirmed_risk``; an incomplete indicator list would fail open when a
    high score arrives without domain_mismatch or OTP keywords.
    """
    return terms.confirmed_risk >= TECHNICAL_RISK_FLOOR


def compound_mention_override_active(terms: PriorityTerms) -> bool:
    """direct_mention AND (urgency OR group admin).

    Does not consult safety or the technical ceiling; callers gate those.
    """
    if terms.direct_mention < DIRECT_MENTION_FLOOR:
        return False
    urgent = terms.urgency >= URGENCY_MENTION_FLOOR
    return urgent or terms.sender_is_group_admin


def compound_admin_ops_override_active(terms: PriorityTerms) -> bool:
    """Trusted group admin + clear operational urgency → interrupt.

    Complements the mention override for same-day admin ops that may not
    literally @mention the recipient (common in society/school groups).
    """
    if not terms.sender_is_group_admin:
        return False
    return terms.urgency >= URGENCY_MENTION_FLOOR


def score_priority(
    terms: PriorityTerms,
    weights: Mapping[str, float],
) -> float:
    total = 0.0
    for name in POSITIVE_TERMS:
        total += weights[name] * getattr(terms, name)
    for name in PENALTY_TERMS:
        total -= weights[name] * getattr(terms, name)
    return round(total, 4)


def action_from_priority(priority: float) -> str:
    if priority >= NOTIFY_THRESHOLD:
        return "notify"
    if priority >= DIGEST_THRESHOLD:
        return "digest"
    return "mute"


def decide_action(
    terms: PriorityTerms,
    weights: Mapping[str, float],
    *,
    hard_blocked_by_safety: bool = False,
) -> PriorityDecision:
    """Apply linear score, then hard ceilings/overrides in precedence order.

    Precedence:
    1. safety.py hard mute → mute (no notify path)
    2. weighted priority → raw_action
    3. technical notify ceiling → cap at digest
    4. compound mention override → force notify if armed and still allowed
    """
    if hard_blocked_by_safety:
        return PriorityDecision(
            priority=score_priority(terms, weights),
            action="mute",
            raw_action="mute",
            technical_ceiling_active=technical_notify_ceiling_active(terms),
            technical_ceiling_applied=False,
            mention_override_active=compound_mention_override_active(terms),
            mention_override_applied=False,
            hard_blocked_by_safety=True,
            technical_indicators=technical_indicators(terms),
        )

    priority = score_priority(terms, weights)
    raw_action = action_from_priority(priority)
    action = raw_action

    ceiling = technical_notify_ceiling_active(terms)
    ceiling_applied = False
    if ceiling and action == "notify":
        action = "digest"
        ceiling_applied = True

    mention_active = compound_mention_override_active(terms)
    admin_ops_active = compound_admin_ops_override_active(terms)
    mention_applied = False
    # Mention/admin-ops override cannot pierce safety hard-block (already returned)
    # or the technical notify ceiling.
    # Deliberate: these overrides bypass quiet_hour_cost — urgent @mentions /
    # admin ops interrupt during DND by design, not oversight.
    if (mention_active or admin_ops_active) and not ceiling and action != "notify":
        action = "notify"
        mention_applied = True

    return PriorityDecision(
        priority=priority,
        action=action,
        raw_action=raw_action,
        technical_ceiling_active=ceiling,
        technical_ceiling_applied=ceiling_applied,
        mention_override_active=mention_active or admin_ops_active,
        mention_override_applied=mention_applied,
        hard_blocked_by_safety=False,
        technical_indicators=technical_indicators(terms),
    )


def explain(
    terms: PriorityTerms,
    weights: Mapping[str, float],
) -> list[tuple[str, float, float, float]]:
    rows: list[tuple[str, float, float, float]] = []
    for name in POSITIVE_TERMS:
        value = getattr(terms, name)
        weight = weights[name]
        rows.append((name, value, weight, round(weight * value, 4)))
    for name in PENALTY_TERMS:
        value = getattr(terms, name)
        weight = weights[name]
        rows.append((name, value, weight, round(-weight * value, 4)))
    return rows


SCENARIOS: Mapping[str, PriorityTerms] = {
    # Bare @mention, non-urgent, chronically dismissed group → stay digest.
    "muted_group_bare_mention": PriorityTerms(
        urgency=0.2,
        direct_mention=1.0,
        sender_trust=0.3,
        personal_relevance=0.2,
        positive_history=0.1,
        repetition=0.0,
        low_engagement=0.6,
        quiet_hour_cost=0.0,
        confirmed_risk=0.0,
        sender_is_group_admin=False,
    ),
    # Compound override: @mention + urgent content → force notify.
    "mention_with_urgency": PriorityTerms(
        urgency=0.8,
        direct_mention=1.0,
        sender_trust=0.3,
        personal_relevance=0.2,
        positive_history=0.1,
        repetition=0.0,
        low_engagement=0.6,
        quiet_hour_cost=0.0,
        confirmed_risk=0.0,
        sender_is_group_admin=False,
    ),
    # Compound override: @mention + group admin (even if urgency is low).
    "mention_from_group_admin": PriorityTerms(
        urgency=0.2,
        direct_mention=1.0,
        sender_trust=0.7,
        personal_relevance=0.3,
        positive_history=0.2,
        repetition=0.0,
        low_engagement=0.5,
        quiet_hour_cost=0.0,
        confirmed_risk=0.0,
        sender_is_group_admin=True,
    ),
    "trusted_promo_always_dismissed": PriorityTerms(
        urgency=0.1,
        direct_mention=0.0,
        sender_trust=0.9,
        personal_relevance=0.4,
        positive_history=0.2,
        repetition=0.7,
        low_engagement=0.9,
        quiet_hour_cost=0.0,
        confirmed_risk=0.1,
    ),
    "medium_risk_known_admin": PriorityTerms(
        urgency=0.5,
        direct_mention=0.0,
        sender_trust=0.85,
        personal_relevance=0.5,
        positive_history=0.8,
        repetition=0.0,
        low_engagement=0.1,
        quiet_hour_cost=0.0,
        confirmed_risk=0.55,
        credential_otp_or_payment_request=True,
        sender_is_group_admin=True,
    ),
    "new_sender_urgent_language": PriorityTerms(
        urgency=0.9,
        direct_mention=0.0,
        sender_trust=0.05,
        personal_relevance=0.0,
        positive_history=0.0,
        repetition=0.0,
        low_engagement=0.0,
        quiet_hour_cost=0.0,
        confirmed_risk=0.35,
    ),
    "admin_high_history_domain_mismatch": PriorityTerms(
        urgency=0.6,
        direct_mention=0.0,
        sender_trust=0.95,
        personal_relevance=0.7,
        positive_history=0.95,
        repetition=0.0,
        low_engagement=0.05,
        quiet_hour_cost=0.0,
        confirmed_risk=0.50,
        domain_mismatch=True,
        sender_is_group_admin=True,
    ),
    # Mention + admin would override, but score ceiling forbids notify.
    "mention_admin_with_domain_mismatch": PriorityTerms(
        urgency=0.7,
        direct_mention=1.0,
        sender_trust=0.9,
        personal_relevance=0.5,
        positive_history=0.8,
        repetition=0.0,
        low_engagement=0.2,
        quiet_hour_cost=0.0,
        confirmed_risk=0.50,
        domain_mismatch=True,
        sender_is_group_admin=True,
    ),
    # Injection-only risk (no domain mismatch, no credential/OTP keyword).
    # confirmed_risk mirrors a single override-phrase hit (~0.55). Mentions +
    # urgency + history must still be capped at digest by the score ceiling.
    "injection_only_risk_with_mention": PriorityTerms(
        urgency=0.8,
        direct_mention=1.0,
        sender_trust=0.7,
        personal_relevance=0.6,
        positive_history=0.95,
        repetition=0.0,
        low_engagement=0.05,
        quiet_hour_cost=0.0,
        confirmed_risk=0.55,
        domain_mismatch=False,
        credential_otp_or_payment_request=False,
        sender_is_group_admin=False,
    ),
}


def compare_profiles(scenario_name: str) -> dict[str, PriorityDecision]:
    terms = SCENARIOS[scenario_name]
    return {
        "default": decide_action(terms, DEFAULT_WEIGHTS),
        "mention_and_risk": decide_action(terms, MENTION_AND_RISK_WEIGHTS),
        "history_heavy": decide_action(terms, HISTORY_HEAVY_WEIGHTS),
    }


if __name__ == "__main__":
    print("Updated interrupt formula:")
    print("  0) if hard_blocked_by_safety (from safety.py): action := mute")
    print("  1) if confirmed_risk >= TECHNICAL_RISK_FLOOR:")
    print("         max_action := digest   # score alone; no indicator allowlist")
    print("  2) priority := Σ(w_pos * term) − Σ(w_pen * term)")
    print("  3) raw_action := notify/digest/mute from thresholds")
    print("  4) action := min(raw_action, max_action)")
    print("  5) compound mention override (NOT a global weight raise):")
    print("        if direct_mention AND (urgency>=floor OR sender_is_group_admin)")
    print("           AND not hard_blocked_by_safety")
    print("           AND not technical_ceiling_active:")
    print("             action := notify")
    print("        bare non-urgent mentions stay on the linear path (often digest)")
    print()
    print(f"TECHNICAL_RISK_FLOOR={TECHNICAL_RISK_FLOOR}")
    print(f"URGENCY_MENTION_FLOOR={URGENCY_MENTION_FLOOR}")
    print(f"direct_mention weight (all profiles)={DEFAULT_WEIGHTS['direct_mention']}")
    print(f"thresholds: notify>={NOTIFY_THRESHOLD}, digest>={DIGEST_THRESHOLD}, else mute\n")

    profiles = (
        ("DEFAULT", DEFAULT_WEIGHTS),
        ("MENTION_AND_RISK (risk penalty up; mention weight unchanged)", MENTION_AND_RISK_WEIGHTS),
        ("HISTORY_HEAVY", HISTORY_HEAVY_WEIGHTS),
    )

    for label, weights in profiles:
        print(f"=== {label} ===")
        for name, weight in weights.items():
            sign = "+" if name in POSITIVE_TERMS else "−"
            print(f"  {sign}{name:20s} w={weight:.2f}  # {WEIGHT_REASONS[name]}")
        print()

    print(
        f"{'scenario':<40} {'default':>16} {'mention+risk':>16} {'history':>16}"
    )
    print("-" * 92)
    for name in SCENARIOS:
        rows = compare_profiles(name)

        def fmt(key: str) -> str:
            d = rows[key]
            if d.mention_override_applied:
                mark = "!"
            elif d.technical_ceiling_applied:
                mark = "*"
            elif d.technical_ceiling_active:
                mark = "~"
            else:
                mark = " "
            return f"{d.priority:+.2f}/{d.action}{mark}"

        print(
            f"{name:<40} {fmt('default'):>16} "
            f"{fmt('mention_and_risk'):>16} {fmt('history_heavy'):>16}"
        )
    print(
        "legend: ! = compound mention forced notify; "
        "* = ceiling forced notify→digest; "
        "~ = ceiling armed, raw already ≤digest"
    )

    print()
    for scenario_name, terms in SCENARIOS.items():
        print("=" * 92)
        print(f"SCENARIO: {scenario_name}")
        print("plugged-in term values (already clipped to [0, 1]):")
        for term_name in list(POSITIVE_TERMS) + list(PENALTY_TERMS):
            print(f"  {term_name:20s} = {getattr(terms, term_name):.2f}")
        print(f"  sender_is_group_admin = {terms.sender_is_group_admin}")
        indicators = technical_indicators(terms)
        print(f"  technical_indicators  = {indicators if indicators else '()'}")
        print(
            f"  ceiling_active        = {technical_notify_ceiling_active(terms)}"
            f"  (confirmed_risk>={TECHNICAL_RISK_FLOOR}; indicators not required)"
        )
        print(f"  mention_override_armed= {compound_mention_override_active(terms)}")
        print()
        for label, weights in profiles:
            decision = decide_action(terms, weights)
            notes: list[str] = []
            if decision.mention_override_applied:
                notes.append(
                    f"MENTION OVERRIDE applied: raw={decision.raw_action} -> notify"
                )
            if decision.technical_ceiling_applied:
                notes.append(
                    f"CEILING applied: raw={decision.raw_action} -> digest"
                )
            elif decision.technical_ceiling_active:
                notes.append(f"ceiling armed; raw already {decision.raw_action}")
            note = f"  [{'; '.join(notes)}]" if notes else ""
            print(
                f"-- under {label}: priority={decision.priority:+.4f} "
                f"-> {decision.action}{note}"
            )
            print(f"  {'term':<20} {'value':>6} {'weight':>7} {'contrib':>9}")
            for term_name, value, weight, contrib in explain(terms, weights):
                print(
                    f"  {term_name:<20} {value:>6.2f} {weight:>7.2f} {contrib:>+9.4f}"
                )
            print()
