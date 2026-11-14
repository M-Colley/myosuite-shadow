"""Raw acquisition replay is deterministic and resume preflight is non-destructive."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite_shadow import ShadowModel, ShadowModelConfig
from myosuite_shadow.log import RECOVERY, SessionWriter, read_session
from myosuite_shadow.study_protocol import StudyDefinition, TrialJournal
from myosuite_shadow.track_cli import main
from myosuite_shadow.tracking import RawFrameWriter, TrackerFrame


def _fixture(model: mujoco.MjModel, root: Path) -> tuple[Path, Path]:
    model_path = root / "subject.mjb"
    mujoco.mj_saveModel(model, str(model_path), None)
    settings = {
        "model": "subject.mjb",
        "keypoints": [
            {"name": "elbow", "frame": "r_ulna_radius_hand"},
            {"name": "wrist", "frame": "wrist", "frame_type": "site"},
        ],
        "shadow": {"window": 5, "fatigue": False, "checkpoint_every": 0},
    }
    config = root / "config.json"
    config.write_text(json.dumps(settings), encoding="utf-8")
    data = mujoco.MjData(model)
    data.qpos[:] = 1.2
    mujoco.mj_forward(model, data)
    raw = root / "frames.jsonl"
    with RawFrameWriter(raw) as writer:
        for k in range(20):
            writer.write(
                TrackerFrame(
                    time_s=k / 20,
                    source_id="recorded_camera",
                    sequence=k,
                    person_id="P07",
                    coordinate_frame="model_world",
                    positions={
                        "elbow": data.xpos[model.body("r_ulna_radius_hand").id],
                        "wrist": data.site_xpos[model.site("wrist").id],
                    },
                    position_tracked={"elbow": True, "wrist": True},
                )
            )
    return config, raw


def test_offline_raw_frames_have_no_machine_dependent_backlog(
    elbow: mujoco.MjModel,
    tmp_path: Path,
) -> None:
    config, raw = _fixture(elbow, tmp_path)
    for name in ("a", "b"):
        main(
            [
                "--config",
                str(config),
                "--frames",
                str(raw),
                "--log-dir",
                str(tmp_path / name),
            ]
        )
    a, b = read_session(tmp_path / "a"), read_session(tmp_path / "b")
    assert len(a.input_time) == 20 and len(a.time) == 16
    assert a.columns["skipped"].sum() == 0
    np.testing.assert_array_equal(a.activation, b.activation)
    np.testing.assert_array_equal(a.input_qpos, b.input_qpos)


def test_study_capture_context_and_automatic_trial_report(elbow, tmp_path):
    from myosuite_shadow.study_analysis import study_report
    from myosuite_shadow.study_catalog import StudyCatalog

    config, raw = _fixture(elbow, tmp_path)
    definition = StudyDefinition.from_dict(
        {
            "schema_version": 1,
            "study_id": "synthetic_capture",
            "protocol_version": "1",
            "conditions": {"hold": {"mass_kg": 0}},
            "trials": [{"trial_id": "hold1", "condition": "hold"}],
            "quality": {"min_coverage": 0.5},
            "outcomes": [
                {
                    "name": "torque",
                    "selector": "tau_0",
                    "semantic_id": "synthetic.elbow.required_torque",
                    "unit": "Nm",
                    "coordinate_frame": "model_joint",
                    "normalization": "none",
                    "method_version": "synthetic_v1",
                    "statistic": "time_mean",
                }
            ],
        }
    )
    journal = TrialJournal.create(
        tmp_path / "study.sqlite",
        definition,
        participant_id="SYNTHETIC_P01",
        run_id="r1",
    )
    journal.start("hold1", 0.0)
    journal.end(1.0)
    log = tmp_path / "session"
    main(
        [
            "--config",
            str(config),
            "--frames",
            str(raw),
            "--log-dir",
            str(log),
            "--study-run",
            str(journal.path),
        ]
    )
    report = study_report(journal.path)
    assert read_session(log).record["metadata"]["study"] == journal.context
    assert report["trials"][0]["quality"]["raw_frames"] == 20
    assert report["trials"][0]["quality"]["status"] == "usable"
    assert "study_preflight" in read_session(log).record["metadata"]
    assert np.isfinite(report["trials"][0]["outcomes"]["torque"]["value"])
    with StudyCatalog(tmp_path / "catalog.sqlite") as catalog:
        assert len(catalog.add(report)) == 64
        assert catalog.list()[0]["definition_sha256"] == definition.sha256


def test_study_model_requirement_fails_before_capture_or_binding(elbow, tmp_path):
    config, raw = _fixture(elbow, tmp_path)
    definition = StudyDefinition.from_dict(
        {
            "schema_version": 1,
            "study_id": "wrong_model",
            "protocol_version": "1",
            "conditions": {"a": {}},
            "trials": [{"trial_id": "a1", "condition": "a"}],
            "outcomes": [],
            "requirements": {"model_sha256": "0" * 64},
        }
    )
    journal = TrialJournal.create(
        tmp_path / "study.sqlite",
        definition,
        participant_id="SYNTHETIC_P01",
        run_id="r1",
    )
    log = tmp_path / "never_opened"
    with pytest.raises(SystemExit):
        main(
            [
                "--config",
                str(config),
                "--frames",
                str(raw),
                "--log-dir",
                str(log),
                "--study-run",
                str(journal.path),
            ]
        )
    assert not log.exists()
    assert journal.session_path is None


def test_study_resume_cannot_omit_journal_or_change_frozen_config(elbow, tmp_path):
    config, raw = _fixture(elbow, tmp_path)
    definition = StudyDefinition.from_dict(
        {
            "schema_version": 1,
            "study_id": "frozen",
            "protocol_version": "1",
            "conditions": {"a": {}},
            "trials": [{"trial_id": "a1", "condition": "a"}],
            "outcomes": [],
            "requirements": {
                "tracking_config_sha256": hashlib.sha256(
                    config.read_bytes()
                ).hexdigest()
            },
        }
    )
    journal = TrialJournal.create(
        tmp_path / "study.sqlite",
        definition,
        participant_id="SYNTHETIC_P01",
        run_id="r1",
    )
    log = tmp_path / "session"
    args = ["--config", str(config), "--frames", str(raw), "--log-dir", str(log)]
    main(args + ["--study-run", str(journal.path)])
    before = {p.name: p.read_bytes() for p in log.iterdir() if p.is_file()}
    with pytest.raises(SystemExit):
        main(args + ["--resume"])
    settings = json.loads(config.read_text())
    settings["acquisition"] = {"max_age_s": 0.5}
    config.write_text(json.dumps(settings))
    with pytest.raises(SystemExit):
        main(args + ["--resume", "--study-run", str(journal.path)])
    assert before == {p.name: p.read_bytes() for p in log.iterdir() if p.is_file()}


@pytest.mark.parametrize("evidence", [None, {}, [], "", False, 0])
def test_empty_required_calibration_is_rejected_before_binding(
    elbow, tmp_path, evidence
):
    config, raw = _fixture(elbow, tmp_path)
    settings = json.loads(config.read_text())
    settings["calibration"] = {"strength": evidence}
    config.write_text(json.dumps(settings))
    definition = StudyDefinition.from_dict(
        {
            "schema_version": 1,
            "study_id": "calibration",
            "protocol_version": "1",
            "conditions": {"a": {}},
            "trials": [{"trial_id": "a1", "condition": "a"}],
            "outcomes": [],
            "requirements": {"required_calibrations": ["strength"]},
        }
    )
    journal = TrialJournal.create(
        tmp_path / "study.sqlite",
        definition,
        participant_id="SYNTHETIC_P01",
        run_id="r1",
    )
    log = tmp_path / "never_opened"
    with pytest.raises(SystemExit):
        main(
            [
                "--config",
                str(config),
                "--frames",
                str(raw),
                "--log-dir",
                str(log),
                "--study-run",
                str(journal.path),
            ]
        )
    assert not log.exists()
    assert journal.session_path is None


def test_resume_config_is_checked_before_modifying_session_files(
    elbow: mujoco.MjModel,
    tmp_path: Path,
) -> None:
    config, raw = _fixture(elbow, tmp_path)
    log = tmp_path / "session"
    main(["--config", str(config), "--frames", str(raw), "--log-dir", str(log)])
    before = {p.name: p.read_bytes() for p in log.iterdir() if p.is_file()}
    settings = json.loads(config.read_text())
    settings["shadow"]["window"] = 9
    config.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(
            [
                "--config",
                str(config),
                "--frames",
                str(raw),
                "--log-dir",
                str(log),
                "--resume",
            ]
        )
    assert before == {p.name: p.read_bytes() for p in log.iterdir() if p.is_file()}
    settings["shadow"]["window"] = 5
    config.write_text(json.dumps(settings), encoding="utf-8")
    continuation = tmp_path / "continuation.jsonl"
    from myosuite_shadow.tracking import read_frames

    with RawFrameWriter(continuation) as writer:
        for frame in read_frames(raw):
            writer.write(
                dataclasses.replace(
                    frame, time_s=frame.time_s + 1, sequence=frame.sequence + 20
                )
            )
    main(
        [
            "--config",
            str(config),
            "--frames",
            str(continuation),
            "--log-dir",
            str(log),
            "--resume",
        ]
    )
    assert len(read_session(log).input_time) == 40


@pytest.mark.parametrize("alias_kind", ["same_path", "relative_path", "hardlink"])
def test_resume_refuses_raw_input_output_alias_before_changing_session(
    elbow: mujoco.MjModel, tmp_path: Path, monkeypatch, alias_kind: str
) -> None:
    config, _ = _fixture(elbow, tmp_path)
    missing = tmp_path / "missing.jsonl"
    with RawFrameWriter(missing) as writer:
        for index in range(2):
            writer.write(
                TrackerFrame(
                    index, "camera", index, "P07", {}, coordinate_frame="model_world"
                )
            )
    log = tmp_path / "session"
    main(["--config", str(config), "--frames", str(missing), "--log-dir", str(log)])
    tracked = log / "tracking.jsonl"
    if alias_kind == "hardlink":
        source = tmp_path / "tracking-alias.jsonl"
        source.hardlink_to(tracked)
        assert source.resolve() != tracked.resolve()
    elif alias_kind == "relative_path":
        monkeypatch.chdir(tmp_path)
        source = Path("session") / "tracking.jsonl"
    else:
        source = tracked
    before = {p.name: p.read_bytes() for p in log.iterdir() if p.is_file()}
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--config",
                str(config),
                "--frames",
                str(source),
                "--log-dir",
                str(log),
                "--resume",
                "--max-frames",
                "5",
            ]
        )
    assert error.value.code == 2
    assert before == {p.name: p.read_bytes() for p in log.iterdir() if p.is_file()}
    assert len(tracked.read_text().splitlines()) == 2


def test_pending_recovery_cli_preflight_is_readonly_then_valid_resume_keeps_inputs(
    elbow: mujoco.MjModel, tmp_path: Path
) -> None:
    config, _ = _fixture(elbow, tmp_path)
    settings = json.loads(config.read_text())
    settings["shadow"]["checkpoint_every"] = 1000
    config.write_text(json.dumps(settings))
    log = tmp_path / "session"
    shadow = ShadowModel(
        elbow, ShadowModelConfig.from_dict(settings["shadow"]), log_dir=log
    )
    for index in range(12):
        shadow.push(index / 20, [1.2])
    shadow.log.close_files()
    writer, _, _, _ = SessionWriter.reopen(log, elbow.nv)
    writer.close_files()
    assert (log / RECOVERY).exists()
    frames = tmp_path / "continuation.jsonl"
    frames.write_text("")
    settings["shadow"]["window"] = 9
    config.write_text(json.dumps(settings))
    before = {p.name: p.read_bytes() for p in log.iterdir() if p.is_file()}
    args = [
        "--config",
        str(config),
        "--frames",
        str(frames),
        "--log-dir",
        str(log),
        "--resume",
    ]
    with pytest.raises(SystemExit):
        main(args)
    assert before == {p.name: p.read_bytes() for p in log.iterdir() if p.is_file()}
    settings["shadow"]["window"] = 5
    config.write_text(json.dumps(settings))
    main(args)
    assert len(read_session(log).input_time) == 12
    assert not (log / RECOVERY).exists()
