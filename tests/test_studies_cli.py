"""Exercise the installed study interface without participant data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from myosuite_shadow.studies_cli import main
from myosuite_shadow.study_protocol import TrialJournal


def test_cli_initializes_and_resumes_a_trial_journal(tmp_path, capsys):
    definition = (
        Path(__file__).parents[1] / "examples" / "studies" / "elbow_protocol.json"
    )
    journal = tmp_path / "run.sqlite"
    assert (
        main(
            [
                "init",
                str(definition),
                str(journal),
                "--participant",
                "SYNTHETIC_P01",
                "--run-id",
                "A",
            ]
        )
        == 0
    )
    first = json.loads(capsys.readouterr().out)
    assert first["context"]["study_id"] == "elbow_hold_example"
    assert (
        main(
            ["trial", str(journal), "start", "--trial-id", "unloaded_1", "--time", "0"]
        )
        == 0
    )
    assert main(["trial", str(journal), "pause", "--time", "1"]) == 0
    assert main(["trial", str(journal), "resume", "--time", "2"]) == 0
    assert main(["trial", str(journal), "end", "--time", "3"]) == 0
    capsys.readouterr()
    assert main(["status", str(journal)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["trials"][0]["intervals"] == [[0.0, 1.0], [2.0, 3.0]]
    assert result["trials"][0]["state"] == "completed"


def test_cli_invalid_transition_does_not_append_events(tmp_path):
    definition = (
        Path(__file__).parents[1] / "examples" / "studies" / "elbow_protocol.json"
    )
    path = tmp_path / "run.sqlite"
    main(
        [
            "init",
            str(definition),
            str(path),
            "--participant",
            "SYNTHETIC_P01",
            "--run-id",
            "A",
        ]
    )
    before = TrialJournal.open(path).snapshot()
    with pytest.raises(SystemExit) as error:
        main(["trial", str(path), "pause", "--time", "0"])
    assert error.value.code == 2
    assert TrialJournal.open(path).snapshot() == before


def test_learning_cli_marks_latent_targets_ineligible(tmp_path, capsys):
    manifest = tmp_path / "learning.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "id": "fatigue",
                        "sha256": "0" * 64,
                        "parents": [],
                        "kind": "latent",
                    }
                ],
                "targets": ["fatigue"],
                "model_inputs": [],
                "fit_inputs": [],
                "splits": {"fatigue": "test"},
                "participant_keys": {"fatigue": "project:SYNTHETIC_P01"},
            }
        )
    )
    assert main(["learning-check", str(manifest)]) == 1
    assert json.loads(capsys.readouterr().out)["eligible"] is False


def test_empty_local_catalog_can_be_listed(tmp_path, capsys):
    assert main(["catalog-list", str(tmp_path / "catalog.sqlite")]) == 0
    assert json.loads(capsys.readouterr().out) == {"runs": []}
