"""Synthetic software fixtures; these tests are not real-participant validation."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite_shadow.log import read_session
from myosuite_shadow.pipeline import ShadowModel
from myosuite_shadow.recordings import (
    audit_recording,
    freeze_recording,
    run_recording_regression,
)
from myosuite_shadow.tracking import RawFrameWriter, Registration, TrackerFrame
from myosuite_shadow.tracking_config import load_tracking_setup


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def recording(elbow, tmp_path):
    """Known generated poses plus explicit software fault annotations."""
    mujoco.mj_saveModel(elbow, str(tmp_path / "subject.mjb"), None)
    registration = Registration(np.eye(3), np.zeros(3)).record()
    _write(tmp_path / "registration1.json", registration | {"revision": 1})
    _write(
        tmp_path / "config.json",
        {
            "model": "subject.mjb",
            "keypoints": [
                {"name": "elbow", "frame": "r_ulna_radius_hand"},
                {"name": "wrist", "frame": "wrist", "frame_type": "site"},
            ],
            "ik": {"required_keypoints": ["elbow", "wrist"]},
            "registration": registration,
            "calibration": {"evidence": "synthetic software fixture"},
            "shadow": {"window": 5, "fatigue": True, "checkpoint_every": 0},
        },
    )
    data = mujoco.MjData(elbow)
    data.qpos[:] = 1.2
    mujoco.mj_forward(elbow, data)
    points = {
        "elbow": data.xpos[elbow.body("r_ulna_radius_hand").id].copy(),
        "wrist": data.site_xpos[elbow.site("wrist").id].copy(),
    }
    with RawFrameWriter(tmp_path / "frames.jsonl") as writer:
        for index in range(40):
            positions = points if index not in {5, 6} else {"elbow": points["elbow"]}
            writer.write(
                TrackerFrame(
                    time_s=index / 20,
                    sequence=index,
                    source_id="synthetic_camera",
                    person_id="SYNTHETIC_P07",
                    positions=positions,
                    position_tracked={name: True for name in positions},
                    transform_revision=int(index >= 10),
                    time_kind="capture",
                    capture_unix=1000 + index / 20,
                    receive_unix=1000.01 + index / 20,
                    metadata={
                        "clock_mapping": {
                            "source_anchor_s": int(index >= 15),
                            "uncertainty_s": 0.001,
                        },
                        "missing_cameras": ["side"] if index == 6 else [],
                    },
                )
            )
    draft = {
        "schema_version": 1,
        "origin": "synthetic",
        "recording_id": "software_fixture_v1",
        "source_id": "synthetic_camera",
        "person_id": "SYNTHETIC_P07",
        "source_versions": {"generator": "test_recordings.py/v1"},
        "files": {
            "frames": {"path": "frames.jsonl", "role": "recording"},
            "config": {"path": "config.json", "role": "configuration"},
            "model": {"path": "subject.mjb", "role": "configuration"},
            "registration1": {
                "path": "registration1.json",
                "role": "calibration",
            },
        },
        "controls": [
            {"before_row": 11, "kind": "registration", "file": "registration1"},
            {"before_row": 25, "kind": "resume"},
        ],
        "expected_quality": [
            {"event": "occlusion", "rows": [5, 6], "reason": "required_landmarks"},
            {
                "event": "missing view",
                "rows": [6, 7],
                "reason": "required_landmarks",
                "raw_field": "missing_cameras",
            },
            {
                "event": "recenter needs registration",
                "rows": [10, 11],
                "reason": "registration_required",
            },
        ],
        "expected_events": [
            {"event": "measured registration", "label": "registration"},
            {"event": "clock refresh", "label": "clock_mapping"},
            {"event": "resume", "label": "resumed"},
        ],
    }
    path = _write(tmp_path / "draft.json", draft)
    return path


def test_raw_ik_pipeline_regression_and_original_schedule(recording, tmp_path):
    manifest = tmp_path / "frozen.json"
    freeze_recording(recording, manifest)
    audit = audit_recording(manifest)
    assert audit["frames"] == 40 and audit["clock_changes"] == 1
    first = run_recording_regression(manifest, tmp_path / "first")
    assert first["passed"], first
    assert first["scientific_validation"] == "not established by software regression"
    session = read_session(tmp_path / "first" / "session")
    assert len(session.input_time) == 37
    assert session.record["metadata"]["origin_declared"] == "synthetic"
    draft = json.loads(recording.read_text())
    draft["baseline_session"] = {"path": str(tmp_path / "first" / "session")}
    draft["files"]["baseline_quality"] = {
        "path": str(tmp_path / "first" / "session" / "tracking_quality.jsonl"),
        "role": "baseline",
    }
    _write(recording, draft)
    freeze_recording(recording, tmp_path / "baseline.json")
    second = run_recording_regression(tmp_path / "baseline.json", tmp_path / "second")
    assert second["passed"], second
    assert second["comparisons"]["input_qpos"]["max_abs_difference"] == 0
    assert second["comparisons"]["fatigue"]["max_abs_difference"] < 1e-12
    assert second["checks"][-1]["event"] == "quality_baseline"
    scheduled = read_session(tmp_path / "second" / "scheduled")
    assert scheduled.record["calibration"] == session.record["calibration"]
    assert scheduled.record["metadata"]["acquisition_mode"] == "offline_reprocess"


def test_tampered_files_and_identity_fail_before_output(recording, tmp_path):
    frozen = tmp_path / "frozen.json"
    freeze_recording(recording, frozen)
    raw = tmp_path / "frames.jsonl"
    raw.write_text(raw.read_text().replace("SYNTHETIC_P07", "OTHER_PERSON"))
    with pytest.raises(ValueError, match="hash mismatch"):
        run_recording_regression(frozen, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    freeze_recording(recording, tmp_path / "newhash.json")
    with pytest.raises(ValueError, match="identity mismatch"):
        audit_recording(tmp_path / "newhash.json")


def test_external_xml_change_is_detected_without_baseline(recording, tmp_path):
    model = tmp_path / "subject.xml"
    model.write_text('<mujoco><include file="body.xml"/></mujoco>')
    included = tmp_path / "body.xml"
    included.write_text(
        '<mujocoinclude><worldbody><body name="link"><joint name="j"/>'
        '<geom type="sphere" size=".01"/><site name="wrist" pos=".1 0 0"/>'
        "</body></worldbody></mujocoinclude>"
    )
    settings = json.loads((tmp_path / "config.json").read_text())
    settings["model"] = model.name
    settings["keypoints"][0]["frame"] = "link"
    _write(tmp_path / "config.json", settings)
    draft = json.loads(recording.read_text())
    draft["files"]["model"]["path"] = model.name
    _write(recording, draft)
    frozen = tmp_path / "frozen.json"
    freeze_recording(recording, frozen)
    assert audit_recording(frozen)["frames"] == 40
    top_level = model.read_bytes()
    included.write_text(included.read_text().replace('size=".01"', 'size=".02"'))
    assert model.read_bytes() == top_level
    with pytest.raises(ValueError, match="compiled model hash mismatch"):
        run_recording_regression(frozen, tmp_path / "result")
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("clock_key", ["clock_mapping", "clock"])
def test_clock_audit_counts_missing_to_declared_mapping(recording, tmp_path, clock_key):
    raw = tmp_path / "frames.jsonl"
    rows = [json.loads(row) for row in raw.read_text().splitlines()]
    for row in rows:
        mapping = row["metadata"].pop("clock_mapping")
        row["metadata"][clock_key] = mapping
    rows[0]["metadata"].pop(clock_key)
    raw.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    freeze_recording(recording, tmp_path / "frozen.json")
    assert audit_recording(tmp_path / "frozen.json")["clock_changes"] == 2


def test_annotations_are_assertions_not_assumed_truth(recording, tmp_path):
    draft = json.loads(recording.read_text())
    draft["expected_quality"][1]["raw_field"] = "unobserved_occlusion"
    _write(recording, draft)
    freeze_recording(recording, tmp_path / "frozen.json")
    report = run_recording_regression(tmp_path / "frozen.json", tmp_path / "result")
    assert report["status"] == "complete" and report["passed"] is False
    assert report["checks"][1]["observed"] == 0


def test_recording_changed_during_reprocessing_cannot_pass(
    recording, tmp_path, monkeypatch
):
    from myosuite_shadow import recordings

    freeze_recording(recording, tmp_path / "frozen.json")
    original, calls = recordings.read_frames, [0]

    def changing_input(path):
        calls[0] += 1
        yield from original(path)
        if calls[0] == 2:  # raw replay, after the initial identity audit
            with Path(path).open("a") as stream:
                stream.write("\n")

    monkeypatch.setattr(recordings, "read_frames", changing_input)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        run_recording_regression(tmp_path / "frozen.json", tmp_path / "result")
    report = json.loads((tmp_path / "result" / "regression.json").read_text())
    assert report["status"] == "failed"


def test_missing_receive_clock_requires_explicit_offline(recording, tmp_path):
    raw = tmp_path / "frames.jsonl"
    rows = [json.loads(row) for row in raw.read_text().splitlines()]
    rows[0]["receive_unix"] = None
    raw.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    freeze_recording(recording, tmp_path / "frozen.json")
    with pytest.raises(ValueError, match="receive_unix"):
        audit_recording(tmp_path / "frozen.json")
    draft = json.loads(recording.read_text()) | {"timing": "offline"}
    _write(recording, draft)
    freeze_recording(recording, tmp_path / "offline.json")
    assert audit_recording(tmp_path / "offline.json")["timing"] == "offline"


def test_explicit_baseline_loads_reach_raw_and_scheduled_pipeline(recording, tmp_path):
    freeze_recording(recording, tmp_path / "unloaded.json")
    run_recording_regression(tmp_path / "unloaded.json", tmp_path / "first")
    source = read_session(tmp_path / "first" / "session")
    # A generated time-varying load is software evidence, not a measured force.
    loaded = dataclasses.replace(
        source, input_qfrc=np.linspace(0.1, 0.8, len(source.input_time))[:, None]
    )
    setup = load_tracking_setup(tmp_path / "config.json")
    with ShadowModel(
        setup.model, setup.shadow_config, log_dir=tmp_path / "loaded_baseline"
    ) as shadow:
        list(loaded.replay(shadow))
    draft = json.loads(recording.read_text())
    draft["baseline_session"] = {"path": str(tmp_path / "loaded_baseline")}
    _write(recording, draft)
    freeze_recording(recording, tmp_path / "undeclared.json")
    with pytest.raises(ValueError, match="explicitly select baseline_inputs"):
        run_recording_regression(tmp_path / "undeclared.json", tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()
    draft["loads"] = "baseline_inputs"
    _write(recording, draft)
    freeze_recording(recording, tmp_path / "loaded.json")
    report = run_recording_regression(tmp_path / "loaded.json", tmp_path / "result")
    assert report["passed"], report
    actual = read_session(tmp_path / "result" / "session")
    np.testing.assert_array_equal(actual.input_qfrc, loaded.input_qfrc)
    assert report["load_provenance"]["independent_validation"] is False


def test_reference_and_probe_integration_plus_runnable_example(recording, tmp_path):
    # Generated measurements exercise software; their magnitude has no human meaning.
    (tmp_path / "torque.csv").write_text(
        "time,torque,role\n0.2,2,calibration\n0.9,3,validation\n" "1.0,4,validation\n"
    )
    (tmp_path / "probes.csv").write_text(
        "time,joint,measured_torque,role,qpos_0\n"
        "0.2,r_elbow_flex,2,calibration,1.2\n"
        "0.9,r_elbow_flex,3,validation,1.2\n"
    )
    draft = json.loads(recording.read_text())
    draft["files"].update(
        torque={"path": "torque.csv", "role": "validation"},
        probes={"path": "probes.csv", "role": "validation"},
    )
    draft["references"] = [
        {
            "file": "torque",
            "format": "csv",
            "kind": "torque",
            "role": "validation",
            "instrument": "synthetic_software_fixture",
            "clock": {
                "source_anchor_s": 0,
                "session_anchor_s": 0,
                "uncertainty_s": 0,
                "clock_id": "synthetic",
            },
            "channels": [{"name": "elbow", "column": "torque", "unit": "Nm"}],
            "max_gap_s": 0.1,
            "comparison": {
                "channel": "elbow",
                "prediction": "tau_0",
                "independent": True,
            },
        },
        {"file": "probes", "kind": "mvc_probes", "independent": True},
    ]
    _write(recording, draft)
    freeze_recording(recording, tmp_path / "reference_manifest.json")
    script = Path(__file__).parents[1] / "examples/recordings/check_recording.py"
    command = subprocess.run(
        [
            sys.executable,
            str(script),
            "run",
            str(tmp_path / "reference_manifest.json"),
            str(tmp_path / "example_output"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(command.stdout)
    assert report["passed"]
    assert report["references"][0]["provenance"]["excluded_role_rows"] == 1
    assert report["references"][0]["paired"] >= 2
    probes = report["references"][1]
    assert probes["probes"][0]["exclusion"] == "used_for_calibration"
    assert probes["included"] == 1
    draft["files"]["probes"]["role"] = "calibration"
    _write(recording, draft)
    freeze_recording(recording, tmp_path / "wrong_probe_role.json")
    with pytest.raises(ValueError, match="independent validation artifacts"):
        audit_recording(tmp_path / "wrong_probe_role.json")
