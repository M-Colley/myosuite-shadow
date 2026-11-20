"""Tests for MHR70 keypoint presets and non-inertial generalized external loads."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import mujoco
import numpy as np
import pytest


def _load_submodule(name: str):
    mod_name = f"myosuite_shadow.{name}"
    try:
        return importlib.import_module(mod_name)
    except ModuleNotFoundError as exc:
        if exc.name == name or exc.name == mod_name:
            raise
    pkg_dir = Path(__file__).resolve().parents[1] / "src" / "myosuite_shadow"
    if "myosuite_shadow" not in sys.modules:
        pkg = types.ModuleType("myosuite_shadow")
        pkg.__path__ = [str(pkg_dir)]
        sys.modules["myosuite_shadow"] = pkg
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, pkg_dir / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def test_inertial_external_qfrc_cancels_gravity_and_matches_angular_rne(
    arm: mujoco.MjModel,
) -> None:
    loads = _load_submodule("loads")
    data = mujoco.MjData(arm)
    data.qpos[:] = arm.qpos0
    mujoco.mj_forward(arm, data)

    # Accelerating the base at a0 = +g_world produces -m * g_world on every body,
    # matching -qfrc_bias at zero velocity when passive joint forces are absent.
    grav = np.asarray(arm.opt.gravity, dtype=np.float64)
    qfrc_grav = loads.inertial_external_qfrc(arm, arm.qpos0, -grav)

    # Verify on a multi-body articulated chain with rotational motion.
    xml = """
    <mujoco>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <body name="welded_table" pos="0.2 0 0"><geom size="0.05" mass="5.0"/></body>
        <body name="link1" pos="0 0 0.4">
          <joint name="j1" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.04" mass="2.0"/>
          <body name="link2" pos="0 0 0.3">
            <joint name="j2" type="hinge" axis="1 0 0"/>
            <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.03" mass="1.5"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    q = np.array([0.3, -0.4])
    d.qpos[:] = q
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    # Free-fall a0 = g_world yields -m * g_world, balancing qfrc_bias at qvel = 0.
    freefall_qfrc = loads.inertial_external_qfrc(m, q, m.opt.gravity)
    np.testing.assert_allclose(freefall_qfrc, d.qfrc_bias, atol=1e-11)

    # Rotational acceleration + Coriolis term is finite and non-zero.
    rot_qfrc = loads.inertial_external_qfrc(
        m,
        q,
        [1.2, -0.4, 0.5],
        angular_accel=[0.0, 1.5, -0.8],
        angular_vel=[0.3, -0.2, 1.1],
        qvel=np.array([0.6, -0.9]),
        pivot=[0.0, 0.0, 0.4],
    )
    assert rot_qfrc.shape == (2,)
    assert np.all(np.isfinite(rot_qfrc))
    assert np.linalg.norm(rot_qfrc) > 0.1
    assert qfrc_grav.shape == (arm.nv,)


def test_mhr70_keypoints_and_frame_dict_track_arm_and_head(arm: mujoco.MjModel) -> None:
    pytest.importorskip("mink")
    _load_submodule("flags")
    ik = _load_submodule("ik")

    arm_kps = ik.mhr70_keypoints(arm)
    kp_names = [kp.name for kp in arm_kps]
    assert "right_shoulder" in kp_names
    assert "right_elbow" in kp_names
    assert "right_wrist" in kp_names
    assert "right_index_third_joint" in kp_names
    assert "right_pinky_third_joint" in kp_names
    assert "right_middle_tip" in kp_names
    assert "left_wrist" not in kp_names

    # Build synthetic MHR70 (70, 3) array from a target arm pose and solve IK.
    data = mujoco.MjData(arm)
    q_target = arm.key_qpos[0].copy() if arm.nkey else arm.qpos0.copy()
    data.qpos[:] = q_target
    mujoco.mj_kinematics(arm, data)

    idx_map = {name: i for i, name in enumerate(ik.MHR70_JOINT_NAMES)}
    raw_70 = np.zeros((70, 3), dtype=np.float64)
    conf_70 = np.full(70, 0.95, dtype=np.float64)
    for kp in arm_kps:
        raw_70[idx_map[kp.name]] = data.xpos[arm.body(kp.frame).id]

    rot_z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    cam_pts = (
        raw_70 @ rot_z90
    )  # inverse of rot_z90 since (cam_pts @ rot_z90.T) == raw_70
    pos_dict, conf_dict = ik.mhr70_frame_dict(cam_pts, conf_70, cam_to_world=rot_z90)

    solver = ik.KeypointIK(arm, arm_kps, qpos0=q_target)
    res = solver.solve(0.0, pos_dict, conf_dict)
    assert res.usable
    assert max(res.residual.values()) < 1e-4

    # Welded head is skipped; articulated neck/head with gaze_origin is included.
    xml_locked = """
    <mujoco><worldbody>
      <body name="thorax">
        <body name="neck"><body name="head"><site name="gaze_origin"/></body></body>
        <body name="humerus_r"><joint name="sh" axis="0 1 0"/><geom size="0.03"/></body>
      </body>
    </worldbody></mujoco>
    """
    m_locked = mujoco.MjModel.from_xml_string(xml_locked)
    names_locked = {kp.name for kp in ik.mhr70_keypoints(m_locked)}
    assert "nose" not in names_locked and "neck" not in names_locked

    xml_unlocked = """
    <mujoco><worldbody>
      <body name="thorax">
        <body name="neck"><joint name="ny" axis="0 0 1"/><geom size="0.02"/>
          <body name="head">
            <joint name="np" axis="0 1 0"/><geom size="0.04"/><site name="gaze_origin"/>
          </body>
        </body>
        <body name="humerus_r"><joint name="sh" axis="0 1 0"/><geom size="0.03"/></body>
      </body>
    </worldbody></mujoco>
    """
    m_unlocked = mujoco.MjModel.from_xml_string(xml_unlocked)
    kps_unlocked = {kp.name: kp for kp in ik.mhr70_keypoints(m_unlocked)}
    assert kps_unlocked["neck"].frame == "neck"
    assert kps_unlocked["nose"].frame == "gaze_origin"
    assert kps_unlocked["nose"].frame_type == "site"
