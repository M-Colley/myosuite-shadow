"""Packaged anatomy and synthetic observations, without participant claims."""

import asyncio
import copy

import mujoco
import numpy as np
import pytest

from myosuite_shadow.body_cli import initialize, synthetic_stream
from myosuite_shadow.body_profiles import load_body_profile, profile_for_model
from myosuite_shadow.body_server import BodyServer
from myosuite_shadow.ik import IKConfig, KeypointIK
from myosuite_shadow.preflight import preflight
from myosuite_shadow.tracking.quest import QUEST_FULL_JOINTS
from myosuite_shadow.tracking_config import load_tracking_setup


@pytest.fixture(scope="module")
def profile():
    return load_body_profile()


def positions(profile, qpos):
    data = mujoco.MjData(profile.model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(profile.model, data)
    return {
        p.name: data.xpos[profile.model.body(p.frame).id].copy()
        for p in profile.keypoints
    }


def test_reported_and_excluded_partition_real_muscles(profile):
    model = profile.model
    assert model.nu > 250 and model.nv > 40
    assert {r.id for r in profile.regions} == {
        "trunk",
        "left_arm",
        "right_arm",
        "left_leg",
        "right_leg",
    }
    members = [n for r in profile.regions for n in r.actuators]
    muscles = {
        model.actuator(int(i)).name
        for i in np.flatnonzero(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
    }
    assert len(members) == len(set(members))
    excluded = profile.excluded_actuators
    assert excluded and len(excluded) == len(set(excluded))
    assert not set(members).intersection(excluded)
    assert set(members).union(excluded) == muscles
    mapped = {p.name for p in profile.keypoints}
    assert mapped <= set(QUEST_FULL_JOINTS)
    for region in profile.regions:
        assert region.actuators and region.anchor in mapped
        assert set(region.required_landmarks) <= mapped


def test_finger_and_toe_muscles_do_not_drive_region_summaries(profile):
    reported = {n for r in profile.regions for n in r.actuators}
    excluded = set(profile.excluded_actuators)
    # Distal flexors/extensors depend on nuisance finger/toe configurations.
    assert {
        "FDP2",
        "FDP2_l",
        "FDS5",
        "FDS5_l",
        "EPL",
        "EPL_l",
        "fhl_l",
        "fhl_r",
        "edl_l",
        "edl_r",
    } <= excluded
    assert {"DELT1", "DELT1_l", "soleus_l", "soleus_r"} <= reported


def test_profile_retains_calibrated_parameters(profile):
    model = copy.deepcopy(profile.model)
    model.body_mass[model.body("humerus_l").id] *= 1.05
    before = (model.body_mass.copy(), model.actuator_gainprm.copy(), model.qpos0.copy())
    result = profile_for_model(model)
    assert result.model is model and result.target_dofs
    for actual, expected in zip(
        (model.body_mass, model.actuator_gainprm, model.qpos0),
        before,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert len(result.target_dofs) < model.nv


def test_metric_hand_geometry_and_synthetic_root_motion(profile):
    model = profile.model
    pose = model.key_qpos[0].copy() if model.nkey else model.qpos0.copy()
    points = positions(profile, pose)
    for side in ("left", "right"):
        assert np.linalg.norm(points[f"{side}_wrist"] - points[f"{side}_elbow"]) > 0.1
        assert (
            np.linalg.norm(
                points[f"{side}_index_metacarpal"] - points[f"{side}_little_metacarpal"]
            )
            > 0.01
        )
    ik = KeypointIK(
        model,
        profile.keypoints,
        IKConfig(
            required_keypoints=tuple(p.name for p in profile.keypoints),
            max_residual=0.01,
        ),
        qpos0=pose,
    )
    joint = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
    address = int(model.jnt_qposadr[joint])
    for index in range(3):
        target = pose.copy()
        target[address] += index * 0.001
        result = ik.solve(index / 30, positions(profile, target))
        assert result.usable and max(result.residual.values()) < 1e-3


def test_generated_reference_has_explicit_observable_target_scope(tmp_path):
    path = initialize(tmp_path / "synthetic", source="synthetic")
    report = preflight(path)
    assert report["status"] != "blocked", report["checks"]
    assert report["observability"][0]["rank"] == len(
        report["observability"][0]["target_dofs"]
    )
    assert report["warnings"] > 0


def test_unsupported_anatomy_fails():
    with pytest.raises(ValueError, match="available body profile"):
        load_body_profile("upper_body")
    with pytest.raises(ValueError, match="expected body frames"):
        profile_for_model(mujoco.MjModel.from_xml_string("<mujoco/>"))


def test_fullbody_fk_through_real_clock_ik_physics_and_region_feedback(tmp_path):
    async def run():
        path = initialize(tmp_path / "demo", source="synthetic")
        server = BodyServer(load_tracking_setup(path), log_dir=tmp_path / "run")
        await server.start(port=0)
        states = []
        try:
            producer = asyncio.create_task(synthetic_stream(path, server.uri, 4.5))
            while not producer.done():
                states.append(server.state())
                await asyncio.sleep(0.04)
            await producer
            good = [s for s in states if s["quality"]["status"] == "running"]
            assert good, [s["quality"] for s in states[-5:]]
            assert any(
                all(
                    r["activation"] is not None and r["fatigue"] is not None
                    for r in s["regions"]
                )
                for s in good
            )
            assert good[-1]["quality"]["preflight_status"] == "needs_attention"
        finally:
            await server.close()

    asyncio.run(run())
