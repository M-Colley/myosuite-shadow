"""Keypoint IK: tracks myoArm on its couplings and flags bad frames."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

from myosuite_shadow import ShadowModel, ShadowModelConfig, reaching_motion
from myosuite_shadow.flags import Flag
from myosuite_shadow.motions import independent_hinges

ik = pytest.importorskip("myosuite_shadow.ik")

_ARM_POINTS = [
    "humerus_r",
    "ulna_r",
    "lunate_r",
    "secondmc_r",
    "fifthmc_r",
    "distph3_r",
]

# Shoulder and elbow hinges in a plane; the elbow stops at 0 and 2.5 rad.
_PLANAR = """
<mujoco><compiler angle="radian"/><worldbody>
  <body name="upper"><joint name="shoulder" axis="0 1 0"/>
    <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/>
    <body name="fore" pos=".3 0 0"><joint name="elbow" axis="0 1 0" range="0 2.5"/>
      <geom type="capsule" fromto="0 0 0 .25 0 0" size=".02"/>
      <site name="hand" pos=".25 0 0"/></body></body>
</worldbody></mujoco>"""


def _points(model: mujoco.MjModel, qpos: np.ndarray, names: list[str]) -> list[dict]:
    data, out = mujoco.MjData(model), []
    for q in np.atleast_2d(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        out.append({n: data.xpos[model.body(n).id].copy() for n in names})
    return out


def test_tracks_myoarm_reaches_on_its_couplings(arm: mujoco.MjModel) -> None:
    qpos = reaching_motion(arm, 30.0, 4.0, seed=1)
    solver = ik.KeypointIK(arm, [ik.Keypoint(n, n) for n in _ARM_POINTS])
    results = [
        solver.solve(k / 30, p) for k, p in enumerate(_points(arm, qpos, _ARM_POINTS))
    ]
    assert all(r.usable and r.flags == Flag.NONE for r in results)
    assert max(max(r.residual.values()) for r in results) < 5e-4
    solved = np.array([r.qpos for r in results])
    adr = arm.jnt_qposadr[independent_hinges(arm)]
    assert np.degrees(np.abs(solved[5:, adr] - qpos[5:, adr])).max() < 0.5
    projected = solved.copy()
    project_joint_couplings(JointCouplings.from_model(arm), projected)
    np.testing.assert_allclose(projected, solved, atol=1e-12)


def test_ik_feeds_the_shadow_model(arm: mujoco.MjModel) -> None:
    qpos = reaching_motion(arm, 30.0, 2.0, seed=2)
    solver = ik.KeypointIK(arm, [ik.Keypoint(n, n) for n in _ARM_POINTS])
    shadow = ShadowModel(arm, ShadowModelConfig(window=7, fatigue=False))
    frames = []
    for k, p in enumerate(_points(arm, qpos, _ARM_POINTS)):
        r = solver.solve(k / 30, p)
        frames.append(shadow.push(k / 30, r.qpos, flags=r.flags))
    assert sum(f is not None for f in frames) == 60 - 6


def test_missing_and_unconfident_keypoints() -> None:
    model = mujoco.MjModel.from_xml_string(_PLANAR)
    solver = ik.KeypointIK(
        model, [ik.Keypoint("elbow", "fore"), ik.Keypoint("hand", "hand", "site")]
    )
    data = mujoco.MjData(model)
    data.qpos[:] = [0.3, 1.0]
    mujoco.mj_kinematics(model, data)
    hand = data.site_xpos[0].copy()
    r = solver.solve(0.0, {"hand": hand}, {"hand": 0.9})
    assert not r.usable and Flag.MISSING in r.flags  # one point is not enough
    elbow = data.xpos[model.body("fore").id].copy()
    r = solver.solve(0.03, {"elbow": elbow, "hand": hand}, {"elbow": 0.2, "hand": 0.9})
    assert not r.usable and Flag.LOW_CONFIDENCE in r.flags
    r = solver.solve(0.06, {"elbow": elbow, "hand": hand})
    assert r.usable and r.flags == Flag.NONE
    np.testing.assert_allclose(r.qpos, [0.3, 1.0], atol=1e-4)


def test_unreachable_targets_flag_limits_and_residual() -> None:
    model = mujoco.MjModel.from_xml_string(_PLANAR)
    solver = ik.KeypointIK(
        model,
        [ik.Keypoint("elbow", "fore"), ik.Keypoint("hand", "hand", "site")],
        qpos0=np.array([0.0, 0.5]),
    )
    # The hand bent backwards: the elbow would have to go below 0.
    r = solver.solve(
        0.0, {"elbow": np.array([0.3, 0, 0]), "hand": np.array([0.5, 0, 0.1])}
    )
    assert Flag.JOINT_LIMIT in r.flags and Flag.IK_RESIDUAL in r.flags
    assert not r.usable
    assert r.qpos[1] >= 0.0


def test_rejected_pose_cannot_poison_the_next_ik_seed() -> None:
    model = mujoco.MjModel.from_xml_string(_PLANAR)
    solver = ik.KeypointIK(
        model,
        [ik.Keypoint("elbow", "fore"), ik.Keypoint("hand", "hand", "site")],
        qpos0=np.array([0.3, 1.0]),
    )
    before = solver.configuration.q.copy()
    bad = solver.solve(
        0, {"elbow": np.array([3.0, 0, 0]), "hand": np.array([5.0, 0, 0])}
    )
    assert not bad.usable
    np.testing.assert_array_equal(solver.configuration.q, before)
    data = mujoco.MjData(model)
    data.qpos[:] = before
    mujoco.mj_kinematics(model, data)
    good = {
        "elbow": data.xpos[model.body("fore").id].copy(),
        "hand": data.site_xpos[model.site("hand").id].copy(),
    }
    assert solver.solve(0.1, good).usable
    assert not solver.solve(0.2, good, {"elbow": np.nan, "hand": 1.0}).usable


def test_required_landmark_cannot_be_replaced_by_other_points() -> None:
    model = mujoco.MjModel.from_xml_string(_PLANAR)
    solver = ik.KeypointIK(
        model,
        [ik.Keypoint("elbow", "fore"), ik.Keypoint("hand", "hand", "site")],
        ik.IKConfig(min_keypoints=1, required_keypoints=("hand",)),
    )
    assert not solver.solve(0, {"elbow": np.array([0.3, 0, 0])}).usable


def test_a_jump_between_frames_is_a_glitch() -> None:
    model = mujoco.MjModel.from_xml_string(_PLANAR)
    names = ["fore"]
    solver = ik.KeypointIK(
        model,
        [ik.Keypoint("elbow", "fore"), ik.Keypoint("hand", "hand", "site")],
        qpos0=np.array([0.0, 1.0]),
    )
    data = mujoco.MjData(model)

    def frame(q: list[float]) -> dict:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        return {
            "elbow": data.xpos[model.body(names[0]).id].copy(),
            "hand": data.site_xpos[0].copy(),
        }

    assert solver.solve(0.0, frame([0.0, 1.0])).flags == Flag.NONE
    assert solver.solve(1 / 30, frame([0.02, 1.02])).flags == Flag.NONE
    assert Flag.GLITCH in solver.solve(2 / 30, frame([1.5, 2.0])).flags


def test_noisy_keypoints_stop_early(arm: mujoco.MjModel) -> None:
    qpos = reaching_motion(arm, 30.0, 1.0, seed=4)
    rng = np.random.default_rng(0)
    frames = [
        {n: p + 0.003 * rng.standard_normal(3) for n, p in f.items()}
        for f in _points(arm, qpos, _ARM_POINTS)
    ]
    solver = ik.KeypointIK(arm, [ik.Keypoint(n, n) for n in _ARM_POINTS])
    results = [solver.solve(k / 30, f) for k, f in enumerate(frames)]
    # 3 mm noise never reaches the 0.1 mm tolerance; the step size ends the solve.
    assert np.mean([r.iterations for r in results[1:]]) < 10
    assert max(max(r.residual.values()) for r in results) < 0.02


def test_smoothing_prevents_flips_of_poorly_observed_joints(
    arm: mujoco.MjModel,
) -> None:
    # Pronation is the least observed joint; noise flips it between frames.
    qpos = reaching_motion(arm, 90.0, 6.0, seed=5)
    rng = np.random.default_rng(0)
    frames = [
        {n: p + 0.003 * rng.standard_normal(3) for n, p in f.items()}
        for f in _points(arm, qpos, _ARM_POINTS)
    ]
    glitches = {}
    for smoothing in (0.0, 1e-4):
        solver = ik.KeypointIK(
            arm,
            [ik.Keypoint(n, n) for n in _ARM_POINTS],
            ik.IKConfig(smoothing=smoothing),
        )
        results = [solver.solve(k / 90, f) for k, f in enumerate(frames)]
        glitches[smoothing] = sum(Flag.GLITCH in r.flags for r in results)
    assert glitches[1e-4] <= 0.5 * glitches[0.0] and glitches[0.0] >= 10
