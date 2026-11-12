"""Real IK acquisition gates preserve evidence and prevent unsupported submissions."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import pytest

from myosuite_shadow.acquisition import AcquisitionConfig, TrackedSession
from myosuite_shadow.flags import Flag
from myosuite_shadow.ik import IKConfig, Keypoint, KeypointIK
from myosuite_shadow.tracking import Registration, TrackerFrame, read_frames

_PLANAR = """
<mujoco><compiler angle="radian"/><worldbody>
  <body name="upper"><joint name="shoulder" axis="0 1 0"/>
    <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/>
    <body name="fore" pos=".3 0 0"><joint name="elbow" axis="0 1 0" range="0 2.5"/>
      <geom type="capsule" fromto="0 0 0 .25 0 0" size=".02"/>
      <site name="hand" pos=".25 0 0"/></body></body>
</worldbody></mujoco>"""
_POSE = np.array([0.3, 1.0])


class _LiveSpy:
    """Record the acquisition boundary while using the actual model and IK."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.shadow = SimpleNamespace(log=None, model=model)
        self.submissions: list[tuple] = []
        self.events: list[tuple] = []
        self.resets = 0
        self.health = {"status": "running", "error": None}

    def submit(self, *args: Any, **kwargs: Any) -> None:
        self.submissions.append((args, kwargs))

    def mark(self, label: str, **kwargs: Any) -> None:
        self.events.append((label, kwargs))

    def reset_tracking(self) -> None:
        self.resets += 1


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr("myosuite_shadow.acquisition.now", lambda: 1000.0)
    model = mujoco.MjModel.from_xml_string(_PLANAR)
    solver = KeypointIK(
        model,
        [Keypoint("elbow", "fore"), Keypoint("hand", "hand", "site")],
        IKConfig(required_keypoints=("elbow", "hand")),
        qpos0=_POSE,
    )
    data = mujoco.MjData(model)
    data.qpos[:] = _POSE
    mujoco.mj_forward(model, data)
    points = {
        "elbow": data.xpos[model.body("fore").id].copy(),
        "hand": data.site_xpos[model.site("hand").id].copy(),
    }
    return model, solver, _LiveSpy(model), points


def _frame(points: dict, **changes: Any) -> TrackerFrame:
    positions = changes.get("positions", points)
    return TrackerFrame(
        **(
            {
                "time_s": 1.0,
                "sequence": 1,
                "source_id": "measured_tracker",
                "person_id": "P07",
                "positions": positions,
                "position_tracked": {name: True for name in positions},
                "time_kind": "capture",
                "capture_unix": 1000.0,
                "receive_unix": 1000.0,
                "coordinate_frame": "model_world",
            }
            | changes
        )
    )


def _decisions(directory: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (directory / "tracking_quality.jsonl").read_text().splitlines()
    ]


def test_real_ik_submits_physical_held_load_and_retains_raw_evidence(setup, tmp_path):
    _, solver, live, points = setup
    config = AcquisitionConfig(held_mass_kg=2.0, load_site="hand")
    with TrackedSession(live, solver, config=config, raw_dir=tmp_path) as session:
        assert session.submit(_frame(points))
        assert session.accepted == 1 and session.rejected == 0
    ((args, kwargs),) = live.submissions
    np.testing.assert_allclose(args[1], _POSE, atol=1e-5)
    shoulder = 2 * 9.81 * (0.3 * np.cos(0.3) + 0.25 * np.cos(1.3))
    elbow = 2 * 9.81 * 0.25 * np.cos(1.3)
    np.testing.assert_allclose(args[2], [shoulder, elbow], atol=1e-5)
    assert kwargs["capture_unix"] == 1000.0
    (raw,) = list(read_frames(tmp_path / "tracking.jsonl"))
    assert raw.source_id == "measured_tracker" and raw.person_id == "P07"
    assert _decisions(tmp_path)[0]["accepted"] is True
    assert live.events[0][0] == "tracking_setup"


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"confidence": {"elbow": 1.0, "hand": 0.2}}, "required_landmarks"),
        ({"position_tracked": {"elbow": True, "hand": False}}, "inferred_landmarks"),
        ({"coordinate_frame": "camera_pixels"}, "registration_required"),
        (
            {"prediction_s": 0.02, "time_kind": "predicted", "capture_unix": None},
            "prediction_horizon",
        ),
        ({"metadata": {"clock_mapping": {"uncertainty_s": 0.1}}}, "clock_uncertainty"),
    ],
)
def test_bad_observations_never_reach_live_but_remain_auditable(
    setup, tmp_path, change, reason
):
    _, solver, live, points = setup
    with TrackedSession(live, solver, raw_dir=tmp_path) as session:
        assert not session.submit(_frame(points, **change))
        assert session.rejected == 1
    assert live.submissions == []
    assert len(list(read_frames(tmp_path / "tracking.jsonl"))) == 1
    assert _decisions(tmp_path)[0]["reason"] == reason


def test_missing_required_or_nonmetric_input_cannot_enter_ik(setup, tmp_path):
    _, solver, live, points = setup
    with TrackedSession(live, solver, raw_dir=tmp_path) as session:
        assert not session.submit(_frame(points, positions={"elbow": points["elbow"]}))
        with pytest.raises(TypeError, match="metric 3D"):
            session.submit({"positions_px": {"hand": [300, 250]}})
        with pytest.raises(ValueError, match="3D"):
            _frame(points, positions={"hand": [300, 250]})
    assert not live.submissions
    assert _decisions(tmp_path)[0]["reason"] == "required_landmarks"


def test_reordering_identity_switch_and_bad_ik_cannot_poison_valid_followup(
    setup, tmp_path
):
    _, solver, live, points = setup
    with TrackedSession(live, solver, raw_dir=tmp_path) as session:
        assert session.submit(_frame(points))
        assert not session.submit(_frame(points))
        assert not session.submit(
            _frame(points, sequence=2, time_s=1.1, person_id="P08")
        )
        assert not session.submit(
            _frame(points, sequence=2, time_s=1.1, source_id="other_tracker")
        )
        bad = {"elbow": np.array([3.0, 0, 0]), "hand": np.array([5.0, 0, 0])}
        assert not session.submit(_frame(points, sequence=2, time_s=1.1, positions=bad))
        assert session.submit(_frame(points, sequence=3, time_s=1.2))
    assert len(live.submissions) == 2
    np.testing.assert_allclose(live.submissions[-1][0][1], _POSE, atol=1e-5)
    assert [d["reason"] for d in _decisions(tmp_path)] == [
        None,
        "out_of_order",
        "source_or_person_changed",
        "source_or_person_changed",
        "ik_quality",
        None,
    ]
    assert len(list(read_frames(tmp_path / "tracking.jsonl"))) == 6


def test_reference_and_skeleton_changes_require_renewed_registration(setup, tmp_path):
    _, solver, live, points = setup
    registration = Registration(np.eye(3), np.zeros(3))
    with TrackedSession(
        live, solver, registration=registration, raw_dir=tmp_path
    ) as session:
        assert session.submit(_frame(points, coordinate_frame="tracker_world"))
        new = _frame(
            points,
            coordinate_frame="tracker_world",
            sequence=2,
            time_s=1.1,
            transform_revision=1,
        )
        assert not session.submit(new)
        session.set_registration(dataclasses.replace(registration, revision=1))
        assert session.submit(dataclasses.replace(new, sequence=3, time_s=1.2))
        assert not session.submit(
            dataclasses.replace(new, sequence=4, time_s=1.3, skeleton_revision=7)
        )
        session.set_registration(dataclasses.replace(registration, revision=1))
        assert session.submit(
            dataclasses.replace(new, sequence=5, time_s=1.4, skeleton_revision=7)
        )
    assert live.resets == 2 and len(live.submissions) == 3
    assert [d["reason"] for d in _decisions(tmp_path)] == [
        None,
        "registration_required",
        None,
        "registration_required",
        None,
    ]


def test_opted_in_inference_retains_its_quality_flag(setup):
    _, solver, live, points = setup
    with TrackedSession(
        live, solver, config=AcquisitionConfig(allow_inferred=True)
    ) as session:
        assert session.submit(
            _frame(points, position_tracked={"elbow": True, "hand": False})
        )
    assert live.submissions[0][1]["flags"] & Flag.INFERRED


def test_stale_source_estimate_is_rejected_despite_fresh_receive_time(setup, tmp_path):
    _, solver, live, points = setup
    with TrackedSession(live, solver, raw_dir=tmp_path) as session:
        stale = _frame(
            points,
            time_kind="estimate",
            capture_unix=None,
            evaluation_unix=990.0,
            receive_unix=1000.0,
        )
        assert not session.submit(stale)
    assert not live.submissions
    assert _decisions(tmp_path)[0]["reason"] == "stale_or_future_timestamp"


@pytest.mark.parametrize(
    "change,reason",
    [
        (
            {"capture_unix": 1030.0, "transform_revision": 1, "skeleton_revision": 2},
            "stale_or_future_timestamp",
        ),
        ({"capture_unix": None, "time_kind": "estimate"}, "unmapped_pose_clock"),
        (
            {"metadata": {"clock_mapping": {"uncertainty_s": 0.1}}},
            "clock_uncertainty",
        ),
        ({"metadata": {"clock_mapping": []}}, "invalid_clock_mapping"),
    ],
)
def test_invalid_time_or_clock_does_not_poison_valid_followup(
    setup, tmp_path, change, reason
):
    _, solver, live, points = setup
    with TrackedSession(live, solver, raw_dir=tmp_path) as session:
        assert session.submit(_frame(points))
        assert not session.submit(_frame(points, sequence=2, time_s=31.0, **change))
        assert session.submit(_frame(points, sequence=3, time_s=1.1))
        assert not session.health["registration_required"]
    assert len(live.submissions) == 2
    assert [d["reason"] for d in _decisions(tmp_path)] == [None, reason, None]
    assert len(list(read_frames(tmp_path / "tracking.jsonl"))) == 3


@pytest.mark.parametrize("uncertainty", ["oops", None, -0.1, True, 10**400])
def test_invalid_clock_uncertainty_is_rejected_without_poisoning_order(
    setup, uncertainty
):
    _, solver, live, points = setup
    with TrackedSession(live, solver) as session:
        assert session.submit(_frame(points))
        assert not session.submit(
            _frame(
                points,
                sequence=2,
                time_s=31.0,
                metadata={"clock_mapping": {"uncertainty_s": uncertainty}},
            )
        )
        assert session.health["last_rejection"] == "invalid_clock_mapping"
        assert session.submit(_frame(points, sequence=3, time_s=1.1))
    assert len(live.submissions) == 2


def test_first_future_frame_cannot_bind_the_participant_identity(setup):
    _, solver, live, points = setup
    with TrackedSession(live, solver) as session:
        assert not session.submit(
            _frame(
                points, person_id="P08", sequence=100, time_s=31.0, capture_unix=1030.0
            )
        )
        assert session.submit(_frame(points))
    assert len(live.submissions) == 1


def test_normal_quality_rejection_still_consumes_its_sequence(setup, tmp_path):
    _, solver, live, points = setup
    with TrackedSession(live, solver, raw_dir=tmp_path) as session:
        assert session.submit(_frame(points))
        bad = _frame(
            points, sequence=2, time_s=1.1, confidence={"elbow": 1.0, "hand": 0.2}
        )
        assert not session.submit(bad)
        assert not session.submit(_frame(points, sequence=2, time_s=1.1))
        assert session.submit(_frame(points, sequence=3, time_s=1.2))
    assert len(live.submissions) == 2
    assert [d["reason"] for d in _decisions(tmp_path)] == [
        None,
        "required_landmarks",
        "out_of_order",
        None,
    ]


def test_failed_worker_health_is_not_hidden_by_stale_tracking(setup, monkeypatch):
    _, solver, live, points = setup
    with TrackedSession(live, solver) as session:
        assert session.submit(_frame(points))
        monkeypatch.setattr("myosuite_shadow.acquisition.now", lambda: 1002.0)
        assert session.health["status"] == "tracking_stale"
        live.health = {"status": "failed", "error": "solver stopped"}
        assert session.health["status"] == "failed"
        assert session.health["error"] == "solver stopped"


def test_constructor_refuses_registration_to_a_different_target_world(setup):
    _, solver, live, _ = setup
    wrong = Registration(np.eye(3), np.zeros(3), target_frame="camera_world")
    with pytest.raises(ValueError, match="model_world"):
        TrackedSession(live, solver, registration=wrong)
    assert not live.submissions


def test_invalid_renewed_registration_cannot_replace_the_valid_transform(setup):
    _, solver, live, points = setup
    correct = Registration(np.eye(3), np.zeros(3))
    wrong = dataclasses.replace(correct, target_frame="camera_world")
    with TrackedSession(live, solver, registration=correct) as session:
        with pytest.raises(ValueError, match="model_world"):
            session.set_registration(wrong)
        assert session.registration is correct
        assert live.resets == 0
        assert session.submit(_frame(points, coordinate_frame="tracker_world"))
    assert len(live.submissions) == 1


def test_live_pose_without_mapped_capture_or_evaluation_time_is_rejected(
    setup, tmp_path
):
    _, solver, live, points = setup
    with TrackedSession(live, solver, raw_dir=tmp_path) as session:
        unknown_time = _frame(
            points,
            time_kind="estimate",
            capture_unix=None,
            evaluation_unix=None,
            receive_unix=1000.0,
        )
        assert not session.submit(unknown_time)
    assert not live.submissions
    assert _decisions(tmp_path)[0]["reason"] == "unmapped_pose_clock"
    assert len(list(read_frames(tmp_path / "tracking.jsonl"))) == 1


def test_same_dimension_different_subject_geometry_is_rejected(setup):
    model, solver, _, _ = setup
    other_model = mujoco.MjModel.from_xml_string(_PLANAR.replace(".25", ".40"))
    assert other_model.nq == model.nq and other_model.nv == model.nv
    live = _LiveSpy(other_model)
    with pytest.raises(ValueError, match="calibrated model"):
        TrackedSession(live, solver)
    assert not live.submissions


def test_identical_compiled_model_copy_is_accepted(setup):
    model, solver, _, points = setup
    model_copy = copy.deepcopy(model)
    assert model_copy is not model
    live = _LiveSpy(model_copy)
    with TrackedSession(live, solver) as session:
        assert session.submit(_frame(points))
    np.testing.assert_allclose(live.submissions[0][0][1], _POSE, atol=1e-5)


def test_clock_refresh_resets_derivatives_without_changing_spatial_calibration(setup):
    _, solver, live, points = setup
    with TrackedSession(live, solver) as session:
        first = _frame(
            points,
            metadata={
                "clock_mapping": {
                    "clock_id": "quest",
                    "source_anchor_s": 0,
                    "uncertainty_s": 0.001,
                }
            },
        )
        assert session.submit(first)
        refreshed = dataclasses.replace(
            first,
            sequence=2,
            time_s=1.03,
            metadata={
                "clock_mapping": {
                    "clock_id": "quest",
                    "source_anchor_s": 1,
                    "uncertainty_s": 0.001,
                },
            },
        )
        assert session.submit(refreshed)
        assert live.resets == 1
        assert live.events[-1][0] == "clock_mapping"


def test_fidelity_fallback_cannot_enter_a_high_fidelity_protocol(setup):
    _, solver, live, points = setup
    with TrackedSession(
        live, solver, config=AcquisitionConfig(required_fidelity="high")
    ) as session:
        assert not session.submit(_frame(points, metadata={"actual_fidelity": "low"}))
        assert session.submit(
            _frame(
                points, sequence=2, time_s=1.03, metadata={"actual_fidelity": "high"}
            )
        )
    assert len(live.submissions) == 1
