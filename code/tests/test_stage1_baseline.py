"""Schema, loader, and baseline self-checks for Stage 1."""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import pytest

_CODE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _CODE_ROOT.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from router.baseline import in_quiet_hours, route_dataset
from router.data import load_dataset
from router.output import (
    OutputValidationError,
    read_output_csv,
    validate_predictions,
    write_output_csv,
)
from router.types import OUTPUT_COLUMNS, Prediction


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(_REPO_ROOT / "dataset")


def test_output_column_contract():
    assert list(OUTPUT_COLUMNS) == [
        "message_id",
        "action",
        "message_type",
        "reason",
        "confidence",
        "evidence_message_ids",
    ]


def test_loader_counts_and_media(dataset):
    assert len(dataset.messages) == 110
    assert len(dataset.users) == 54
    assert len(dataset.groups) == 23
    assert len(dataset.businesses) == 110
    assert len(dataset.images) == 20
    assert len(dataset.voice_notes) == 13
    assert dataset.media_warnings == []
    # Every referenced media path resolves under dataset/
    for media in list(dataset.images.values()) + list(dataset.voice_notes.values()):
        assert media.available
        assert media.absolute_path is not None
        assert Path(media.absolute_path).is_file()


def test_loader_never_requires_sample_messages(dataset):
    # Production loader API has no sample path; file may exist on disk unused.
    assert not hasattr(dataset, "sample_messages")


def test_quiet_hours_cross_midnight():
    when = datetime(2026, 7, 30, 23, 30)
    assert in_quiet_hours("22:00-07:00", when) is True
    assert in_quiet_hours("22:00-07:00", datetime(2026, 7, 30, 6, 59)) is True
    assert in_quiet_hours("22:00-07:00", datetime(2026, 7, 30, 7, 0)) is False
    assert in_quiet_hours("09:00-17:00", datetime(2026, 7, 30, 12, 0)) is True
    assert in_quiet_hours("09:00-17:00", datetime(2026, 7, 30, 18, 0)) is False


def test_baseline_cardinality_and_enums(dataset, tmp_path):
    predictions = route_dataset(dataset)
    required_ids = [message.message_id for message in dataset.messages]
    history_ids = {item.message_id for item in dataset.message_history}

    problems = validate_predictions(
        predictions,
        required_message_ids=required_ids,
        history_ids=history_ids,
    )
    assert problems == []

    out = tmp_path / "output.csv"
    write_output_csv(
        out,
        predictions,
        required_message_ids=required_ids,
        history_ids=history_ids,
    )
    rows = read_output_csv(out)
    assert len(rows) == len(required_ids)
    assert [row["message_id"] for row in rows] == required_ids
    for row in rows:
        assert row["action"] in {"notify", "digest", "mute"}
        assert 0.0 <= float(row["confidence"]) <= 1.0
        assert row["reason"].strip()
        assert row["evidence_message_ids"] == "none" or ";" in row["evidence_message_ids"] or row[
            "evidence_message_ids"
        ].startswith("message_")


def test_write_rejects_bad_enum(tmp_path):
    bad = [
        Prediction(
            message_id="msg_x",
            action="ping",
            message_type="urgent",
            reason="bad action",
            confidence=0.5,
            evidence_message_ids="none",
        )
    ]
    with pytest.raises(OutputValidationError):
        write_output_csv(
            tmp_path / "bad.csv",
            bad,
            required_message_ids=["msg_x"],
            history_ids=set(),
        )


def test_production_modules_do_not_load_sample_labels():
    """Production path may mention the sample file only to refuse loading it."""
    code_root = _CODE_ROOT
    scanned = []
    for path in code_root.rglob("*.py"):
        if "evaluation" in path.parts:
            continue
        if path.name.startswith("test_"):
            continue
        text = path.read_text(encoding="utf-8")
        scanned.append(path)
        lowered = text.lower()
        # Allow explicit refusal / documentation, but no CSV open/load of labels.
        if "sample_messages" not in lowered:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if "sample_messages" not in stripped.lower():
                continue
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if "refus" in stripped.lower() or "never" in stripped.lower() or "evaluation" in stripped.lower():
                continue
            if "open(" in stripped or "read_csv" in stripped or "DictReader" in stripped:
                raise AssertionError(f"{path} appears to load sample labels: {stripped}")
            if "load_dataset" in stripped and "sample_messages" in stripped:
                raise AssertionError(f"{path} passes sample_messages into loader: {stripped}")
    assert scanned
    # main.py must hard-refuse sample input.
    main_text = (code_root / "main.py").read_text(encoding="utf-8")
    assert "sample_messages.csv" in main_text
    assert "Refusing" in main_text
