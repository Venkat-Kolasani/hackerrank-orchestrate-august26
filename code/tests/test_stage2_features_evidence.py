"""Stage 2 tests: history joins, features, evidence, quiet hours."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

_CODE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _CODE_ROOT.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from router.baseline import route_dataset, route_message
from router.data import load_dataset
from router.evidence import retrieve_evidence
from router.features import compute_features, in_quiet_hours, recipient_history, same_entity
from router.output import validate_predictions


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(_REPO_ROOT / "dataset")


def test_history_joins_to_events(dataset):
    assert dataset.message_history
    assert dataset.message_events
    missing = 0
    for message in dataset.message_history:
        if dataset.get_event(message.user_id, message.message_id) is None:
            missing += 1
    assert missing == 0

    # Every loaded recipient history item can resolve an event.
    sample_user = dataset.messages[0].user_id
    items = recipient_history(dataset, sample_user)
    assert items
    assert all(item.event is not None for item in items)


def test_quiet_hours_cross_midnight():
    assert in_quiet_hours("22:00-07:00", datetime(2026, 7, 30, 23, 30)) is True
    assert in_quiet_hours("22:00-07:00", datetime(2026, 7, 30, 6, 30)) is True
    assert in_quiet_hours("22:00-07:00", datetime(2026, 7, 30, 7, 0)) is False
    assert in_quiet_hours("22:00-07:00", datetime(2026, 7, 30, 21, 59)) is False
    assert in_quiet_hours("09:00-17:00", datetime(2026, 7, 30, 12, 0)) is True
    assert in_quiet_hours("09:00-17:00", datetime(2026, 7, 30, 18, 0)) is False


def _first_of_type(dataset, conversation_type: str):
    for message in dataset.messages:
        if message.conversation_type == conversation_type:
            return message
    raise AssertionError(f"no {conversation_type} message found")


def test_group_business_personal_features(dataset):
    group_msg = _first_of_type(dataset, "group")
    business_msg = _first_of_type(dataset, "business")
    personal_msg = _first_of_type(dataset, "personal")

    group_features = compute_features(dataset, group_msg)
    business_features = compute_features(dataset, business_msg)
    personal_features = compute_features(dataset, personal_msg)

    assert 0.0 <= group_features.sender_trust <= 1.0
    assert 0.0 <= business_features.personal_relevance <= 1.0
    assert 0.0 <= personal_features.positive_history <= 1.0

    # Group features see membership mute/admin context when available.
    membership = dataset.get_group_member(group_msg.group_id, group_msg.user_id)
    if membership and membership.group_muted_by_user:
        assert group_features.muted_group is True
        assert group_features.low_engagement >= 0.7

    business = dataset.get_business(business_msg.business_id)
    if business and business.domain_mismatch:
        assert business_features.domain_mismatch is True
        assert business_features.sender_trust <= 0.15

    # Same-entity helper distinguishes conversation scopes.
    for other in dataset.history_by_user.get(group_msg.user_id, []):
        if other.group_id == group_msg.group_id:
            assert same_entity(group_msg, other) is True
            break


def test_evidence_ids_exist_in_history_and_support_decision(dataset):
    history_ids = {item.message_id for item in dataset.message_history}
    predictions = route_dataset(dataset)
    with_evidence = 0
    for prediction, message in zip(predictions, dataset.messages, strict=True):
        evidence = prediction.evidence_message_ids
        if evidence == "none":
            continue
        with_evidence += 1
        parts = evidence.split(";")
        assert 1 <= len(parts) <= 3
        for message_id in parts:
            assert message_id in history_ids
            # Evidence must belong to the same recipient's history.
            assert any(
                item.message_id == message_id and item.user_id == message.user_id
                for item in dataset.message_history
            )
    # Stage 2 should find supporting evidence for a non-trivial share of rows.
    assert with_evidence >= 20


def test_evidence_none_when_no_support(dataset):
    # Force an action with empty history by using a synthetic unknown user copy.
    message = dataset.messages[0]
    empty = load_dataset(_REPO_ROOT / "dataset")
    empty.history_by_user[message.user_id] = []
    features = compute_features(empty, message)
    features.history_for_user = []
    assert retrieve_evidence(empty, message, "mute", features) == "none"


def test_router_output_still_valid(dataset):
    predictions = route_dataset(dataset)
    problems = validate_predictions(
        predictions,
        required_message_ids=[m.message_id for m in dataset.messages],
        history_ids={item.message_id for item in dataset.message_history},
    )
    assert problems == []
    # Spot-check one routed message keeps enums.
    sample = route_message(dataset, dataset.messages[0])
    assert sample.action in {"notify", "digest", "mute"}
