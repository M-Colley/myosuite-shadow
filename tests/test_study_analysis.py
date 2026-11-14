"""Analytical synthetic signals; no participant measurements are implied."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from myosuite_shadow import study_analysis
from myosuite_shadow.study_protocol import StudyDefinition, TrialJournal


@pytest.fixture
def run(tmp_path, monkeypatch):
    spec = {
        "schema_version": 1,
        "study_id": "synthetic",
        "protocol_version": "1",
        "conditions": {"a": {"load_kg": 0}, "b": {"load_kg": 1}},
        "trials": [
            {"trial_id": "a1", "condition": "a"},
            {"trial_id": "b1", "condition": "b"},
        ],
        "quality": {"max_gap_s": 0.15, "min_coverage": 0.8},
        "outcomes": [
            {
                "name": "loading",
                "selector": "tau_0",
                "semantic_id": "right.elbow.required_torque",
                "unit": "Nm",
                "coordinate_frame": "model_joint",
                "normalization": "none",
                "method_version": "synthetic-v1",
                "statistic": "time_mean",
            }
        ],
    }
    journal = TrialJournal.create(
        tmp_path / "run.sqlite",
        StudyDefinition.from_dict(spec),
        participant_id="SYNTHETIC_P01",
        run_id="r1",
    )
    root = tmp_path / "session"
    root.mkdir()
    journal.bind_session(root)
    times = np.arange(20) / 10
    session = SimpleNamespace(
        time=times,
        flags=np.zeros(20, dtype=int),
        status=np.array(["solved"] * 20),
        columns={"tau_0": np.where(times < 1, 2.0, 5.0)},
        record={
            "metadata": {"study": journal.context},
            "model_sha256": "0" * 64,
            "model": {"joints": [{"name": "elbow", "type": 3, "dofadr": 0}]},
        },
        checkpoint={"closed": True},
    )
    quality = root / "tracking_quality.jsonl"
    quality.write_text(
        "\n".join(
            json.dumps(
                {
                    "time_s": t / 20,
                    "accepted": t != 3,
                    "reason": "occlusion" if t == 3 else None,
                }
            )
            for t in range(40)
        )
        + "\n"
    )
    monkeypatch.setattr(study_analysis, "read_session", lambda _: session)
    monkeypatch.setattr(study_analysis, "package_provenance", lambda _: {})
    return journal, spec, root, session


def complete(journal):
    journal.start("a1", 0.0)
    journal.end(1.0)
    journal.start("b1", 1.0)
    journal.end(2.0)


def test_trial_boundaries_denominators_and_typed_estimates(run):
    journal, _, _, _ = run
    complete(journal)
    report = study_analysis.study_report(journal.path)
    a, b = report["trials"]
    assert a["outcomes"]["loading"]["value"] == pytest.approx(2)
    assert b["outcomes"]["loading"]["value"] == pytest.approx(5)
    assert a["quality"]["coverage"] == pytest.approx(0.8)
    assert a["quality"]["raw_frames"] == 20
    assert a["quality"]["rejected_frames"] == 1
    assert a["quality"]["rejection_reasons"] == {"occlusion": 1}
    assert a["quality"]["status"] == "usable"
    assert (
        a["outcomes"]["loading"]["contract"]["quantity_class"]
        == "required_joint_torque"
    )
    json.dumps(report, allow_nan=False)


def test_pauses_cannot_be_bridged_even_with_large_allowed_gap(run, tmp_path):
    old, spec, root, session = run
    accept_all(root)
    spec["quality"].update(max_gap_s=0.4, min_coverage=0.7)
    journal = TrialJournal.create(
        tmp_path / "paused.sqlite",
        StudyDefinition.from_dict(spec),
        participant_id="SYNTHETIC_P01",
        run_id="paused",
    )
    journal.bind_session(root)
    session.record["metadata"]["study"] = journal.context
    journal.start("a1", 0.0)
    journal.pause(0.4)
    journal.resume(0.6)
    journal.end(1.0)
    session.columns["tau_0"] = np.where(session.time < 0.6, 2.0, 10.0)
    trial = study_analysis.study_report(journal.path)["trials"][0]
    assert trial["quality"]["active_duration_s"] == pytest.approx(0.8)
    assert trial["quality"]["coverage"] == pytest.approx(0.75)
    assert trial["outcomes"]["loading"]["value"] == pytest.approx(6.0)


def test_missing_samples_and_flags_do_not_become_zero_values(run):
    journal, _, root, session = run
    complete(journal)
    session.columns["tau_0"][3:8] = np.nan
    session.flags[12:18] = 256
    (root / "tracking_quality.jsonl").unlink()
    a, b = study_analysis.study_report(journal.path)["trials"]
    assert a["outcomes"]["loading"]["value"] is None
    assert b["outcomes"]["loading"]["value"] is None
    assert "tracking_evidence_missing" in a["quality"]["issues"]
    assert b["quality"]["status"] != "usable"
    assert b["quality"]["flag_counts"]["256"] == 6


@pytest.mark.parametrize(
    "statistic,expected", [("time_mean", 0.45), ("rms", np.sqrt(0.27)), ("peak", 0.9)]
)
def test_time_weighted_statistics_match_linear_signal(
    run, tmp_path, statistic, expected
):
    _, spec, root, session = run
    accept_all(root)
    spec["outcomes"][0]["statistic"] = statistic
    journal = TrialJournal.create(
        tmp_path / "linear.sqlite",
        StudyDefinition.from_dict(spec),
        participant_id="SYNTHETIC_P01",
        run_id="linear",
    )
    journal.bind_session(root)
    session.record["metadata"]["study"] = journal.context
    session.columns["tau_0"] = session.time.copy()
    complete(journal)
    trial = study_analysis.study_report(journal.path)["trials"][0]
    assert trial["outcomes"]["loading"]["value"] == pytest.approx(expected)


def accept_all(root):
    path = root / "tracking_quality.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text(
        "\n".join(json.dumps(row | {"accepted": True, "reason": None}) for row in rows)
        + "\n"
    )


def test_reset_and_gap_boundaries_are_mandatory_even_if_flags_allowed(run, tmp_path):
    _, spec, root, session = run
    accept_all(root)
    spec["quality"].update(reject_flags=0, min_coverage=0.1)
    journal = TrialJournal.create(
        tmp_path / "resets.sqlite",
        StudyDefinition.from_dict(spec),
        participant_id="SYNTHETIC_P01",
        run_id="resets",
    )
    journal.bind_session(root)
    session.record["metadata"]["study"] = journal.context
    complete(journal)
    session.input_time = session.time.copy()
    session.operations = [{"kind": "reset", "input": 5}]
    session.flags[3] = 32
    trial = study_analysis.study_report(journal.path)["trials"][0]
    assert trial["quality"]["supported_duration_s"] == pytest.approx(0.7)


def test_unimplemented_normalization_is_rejected(run, tmp_path):
    _, spec, root, session = run
    spec["outcomes"][0]["normalization"] = "MVC"
    journal = TrialJournal.create(
        tmp_path / "normalized.sqlite",
        StudyDefinition.from_dict(spec),
        participant_id="SYNTHETIC_P01",
        run_id="normalized",
    )
    journal.bind_session(root)
    session.record["metadata"]["study"] = journal.context
    with pytest.raises(ValueError, match="normalization"):
        study_analysis.study_report(journal.path)


def test_wrong_context_and_mutating_evidence_cannot_report(run, monkeypatch):
    journal, _, root, session = run
    complete(journal)
    session.record["metadata"]["study"] = {}
    with pytest.raises(ValueError, match="context"):
        study_analysis.study_report(journal.path)
    session.record["metadata"]["study"] = journal.context

    def changed(_):
        (root / "new.txt").write_text("unrelated")
        with (root / "tracking_quality.jsonl").open("a") as stream:
            stream.write("{}\n")
        return session

    monkeypatch.setattr(study_analysis, "read_session", changed)
    with pytest.raises(ValueError, match="changed during"):
        study_analysis.study_report(journal.path)


def test_wrong_units_and_open_session_are_not_silently_usable(run, tmp_path):
    journal, spec, root, session = run
    complete(journal)
    session.checkpoint["closed"] = False
    assert (
        "session_not_closed"
        in study_analysis.study_report(journal.path)["trials"][0]["quality"]["issues"]
    )
    spec["outcomes"][0]["unit"] = "N"
    wrong = TrialJournal.create(
        tmp_path / "wrong.sqlite",
        StudyDefinition.from_dict(spec),
        participant_id="SYNTHETIC_P01",
        run_id="wrong",
    )
    wrong.bind_session(root)
    session.record["metadata"]["study"] = wrong.context
    with pytest.raises(ValueError, match="requires unit Nm"):
        study_analysis.study_report(wrong.path)
