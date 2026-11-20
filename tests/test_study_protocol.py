"""Protocol contracts and SQLite transitions use synthetic software fixtures."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from myosuite_shadow.study_protocol import StudyDefinition, TrialJournal


def _spec():
    return {
        "schema_version": 1,
        "study_id": "synthetic_protocol",
        "protocol_version": "1.0",
        "conditions": {"light": {"mass_kg": 1}, "heavy": {"mass_kg": 2}},
        "trials": [
            {"trial_id": "t1", "condition": "light"},
            {"trial_id": "t2", "condition": "heavy"},
        ],
        "outcomes": [
            {
                "name": "elbow_torque",
                "selector": "tau_0",
                "semantic_id": "synthetic.elbow.required_torque",
                "unit": "Nm",
                "coordinate_frame": "model_joint",
                "normalization": "none",
                "method_version": "1",
                "statistic": "time_mean",
            }
        ],
    }


def _journal(path):
    return TrialJournal.create(
        path,
        StudyDefinition.from_dict(_spec()),
        participant_id="SYNTHETIC_P01",
        participant_link_key="synthetic_link",
        run_id="synthetic_run",
        session_id="synthetic_session",
    )


def test_definition_normalizes_and_copies_without_changing_identity(tmp_path):
    source = _spec()
    definition = StudyDefinition.from_dict(source)
    assert definition.data["quality"] == {
        "max_gap_s": 0.1,
        "min_coverage": 0.9,
        "max_rejected_fraction": 0.1,
        "reject_flags": 511,
        "require_tracking": True,
    }
    assert definition.data["requirements"]["required_calibrations"] == []
    digest = definition.sha256
    source["conditions"]["light"]["mass_kg"] = 99
    result = definition.data
    result["outcomes"][0]["quantity_class"] = "required_joint_torque"
    result["trials"].reverse()
    assert definition.sha256 == digest
    assert "quantity_class" not in definition.record()["outcomes"][0]
    path = tmp_path / "definition.json"
    path.write_text(json.dumps(definition.record()), encoding="utf-8-sig")
    assert StudyDefinition.load(path).sha256 == digest
    assert StudyDefinition.from_dict(definition.record()).sha256 == digest


@pytest.mark.parametrize(
    "damage",
    [
        lambda d: d.update(unknown=True),
        lambda d: d.update(schema_version=True),
        lambda d: d.update(study_id=" "),
        lambda d: d["conditions"]["light"].update(value=float("nan")),
        lambda d: d["trials"].append(copy.deepcopy(d["trials"][0])),
        lambda d: d["trials"][0].update(condition="unknown"),
        lambda d: d.update(quality={"max_gap_s": 0}),
        lambda d: d.update(quality={"min_coverage": float("inf")}),
        lambda d: d.update(quality={"max_rejected_fraction": 1.1}),
        lambda d: d.update(quality={"reject_flags": True}),
        lambda d: d.update(quality={"require_tracking": 1}),
        lambda d: d.update(quality={"typo": 1}),
        lambda d: d["outcomes"][0].pop("normalization"),
        lambda d: d["outcomes"][0].update(selector="activation: "),
        lambda d: d["outcomes"][0].update(selector="contact:hand:q"),
        lambda d: d["outcomes"][0].update(statistic="median"),
        lambda d: d["outcomes"].append(copy.deepcopy(d["outcomes"][0])),
        lambda d: d.update(requirements={"model_sha256": "bad"}),
        lambda d: d.update(requirements={"unknown": True}),
        lambda d: d.update(requirements={"required_calibrations": ["MVC", "MVC"]}),
    ],
)
def test_invalid_definitions_fail(damage):
    value = _spec()
    damage(value)
    with pytest.raises(ValueError):
        StudyDefinition.from_dict(value)


def test_requirements_are_explicit_and_hashes_normalized():
    value = _spec()
    value["requirements"] = {
        "model_sha256": "A" * 64,
        "tracking_config_sha256": "b" * 64,
        "required_calibrations": ["geometry", "MVC"],
    }
    requirements = StudyDefinition.from_dict(value).data["requirements"]
    assert requirements["model_sha256"] == "a" * 64
    assert requirements["required_calibrations"] == ["geometry", "MVC"]


def test_duplicate_json_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":2}')
    with pytest.raises(ValueError, match="duplicate JSON"):
        StudyDefinition.load(path)


def test_intervals_pause_reopen_identity_and_binding(tmp_path):
    path = tmp_path / "run.sqlite"
    journal = _journal(path)
    context = journal.context
    assert journal.path == path.resolve()
    assert journal.snapshot()["state"] == "ready"
    session = tmp_path / "session"
    journal.bind_session(session)
    journal.bind_session(session / ".." / "session")
    with pytest.raises(ValueError, match="different session"):
        journal.bind_session(tmp_path / "other")
    journal.start("t1", 1)
    assert journal.snapshot()["trials"][0]["incomplete"]
    assert journal.snapshot()["trials"][0]["intervals"] == []
    journal.pause(2)
    journal = TrialJournal.open(path)
    assert journal.context == context
    assert journal.session_path == session.resolve()
    journal.resume(5)
    assert journal.snapshot()["trials"][0]["intervals"] == [[1, 2]]
    journal.end(7)
    assert journal.snapshot()["trials"][0]["intervals"] == [[1, 2], [5, 7]]
    assert not journal.snapshot()["trials"][0]["incomplete"]
    journal.start("t2", 7)
    journal.abort(8)
    final = TrialJournal.open(path).snapshot()
    assert final["state"] == "completed"
    assert final["trials"][1]["state"] == "aborted"
    assert all(isinstance(e["recorded_unix"], float) for e in final["events"])


def test_wrong_transitions_and_times_do_not_mutate(tmp_path):
    journal = _journal(tmp_path / "run.sqlite")
    before = journal.snapshot()
    for action in (
        lambda: journal.pause(1),
        lambda: journal.start("t2", 1),
        lambda: journal.start("missing", 1),
        lambda: journal.start("t1", float("nan")),
        lambda: journal.start("t1", True),
    ):
        with pytest.raises(ValueError):
            action()
        assert journal.snapshot() == before
    journal.start("t1", 2)
    before = journal.snapshot()
    for action in (
        lambda: journal.resume(3),
        lambda: journal.pause(1),
        lambda: journal.start("t2", 3),
    ):
        with pytest.raises(ValueError):
            action()
        assert journal.snapshot() == before
    journal.end(3)
    before = journal.snapshot()
    with pytest.raises(ValueError):
        journal.start("t1", 4)
    with pytest.raises(FileExistsError):
        _journal(journal.path)
    assert journal.snapshot() == before


def test_concurrent_start_has_one_atomic_winner(tmp_path):
    path = tmp_path / "run.sqlite"
    _journal(path)
    barrier = threading.Barrier(2)

    def start():
        journal = TrialJournal.open(path)
        barrier.wait(timeout=5)
        try:
            journal.start("t1", 1)
            return "started"
        except ValueError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: start(), range(2)))
    assert sorted(results) == ["rejected", "started"]
    assert len(TrialJournal.open(path).snapshot()["events"]) == 1


def test_committed_transition_survives_immediate_process_exit(tmp_path):
    path = tmp_path / "run.sqlite"
    _journal(path).start("t1", 1)
    script = (
        "import os,sys;from myosuite_shadow.study_protocol import TrialJournal;"
        "TrialJournal.open(sys.argv[1]).pause(2);os._exit(0)"
    )
    subprocess.run([sys.executable, "-c", script, str(path)], check=True)
    snapshot = TrialJournal.open(path).snapshot()
    assert snapshot["state"] == "paused"
    assert snapshot["trials"][0]["intervals"] == [[1, 2]]


def test_process_exit_before_commit_preserves_previous_state(tmp_path):
    path = tmp_path / "run.sqlite"
    journal = _journal(path)
    journal.start("t1", 1)
    before = journal.snapshot()
    script = (
        "import os,sqlite3,sys;c=sqlite3.connect(sys.argv[1]);"
        "c.execute('PRAGMA synchronous=FULL');c.execute('BEGIN IMMEDIATE');"
        "c.execute(\"INSERT INTO events VALUES (2,'pause','t1',2,1000)\");os._exit(0)"
    )
    subprocess.run([sys.executable, "-c", script, str(path)], check=True)
    assert TrialJournal.open(path).snapshot() == before
