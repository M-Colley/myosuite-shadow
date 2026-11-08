"""Synthetic reaches and tracking noise."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

from myosuite_shadow import add_tracking_noise, reaching_motion
from myosuite_shadow.motions import independent_hinges


def test_reaches_stay_in_range_and_on_the_couplings(arm: mujoco.MjModel) -> None:
    qpos = reaching_motion(arm, 90.0, 5.0, seed=3)
    assert qpos.shape == (450, arm.nq)
    joints = independent_hinges(arm)
    lo, hi = arm.jnt_range[joints].T
    q = qpos[:, arm.jnt_qposadr[joints]]
    assert (q >= lo + 0.2 * (hi - lo) - 1e-12).all()
    assert (q <= hi - 0.2 * (hi - lo) + 1e-12).all()
    projected = qpos.copy()
    project_joint_couplings(JointCouplings.from_model(arm), projected)
    np.testing.assert_allclose(projected, qpos, atol=1e-12)
    # Smooth: no step larger than a fast reach covers in one frame.
    assert np.abs(np.diff(q, axis=0)).max() < 0.1
    np.testing.assert_array_equal(qpos, reaching_motion(arm, 90.0, 5.0, seed=3))


def test_reaches_need_a_fixed_base() -> None:
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint/><geom size='.1'/></body>"
        "</worldbody></mujoco>"
    )
    with pytest.raises(ValueError, match="fixed-base"):
        reaching_motion(model, 90.0, 1.0, seed=0)


def test_noise_is_on_hinges_only() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody><body><joint type="slide"/><geom size=".1"/>
          <body><joint type="ball"/><geom size=".1"/>
          <body><joint axis="0 1 0"/><geom size=".1"/></body></body></body>
        </worldbody></mujoco>"""
    )
    qpos = np.tile(model.qpos0, (4000, 1))
    noisy = add_tracking_noise(model, qpos, 0.5, seed=0)
    std = np.degrees((noisy - qpos).std(axis=0))
    np.testing.assert_allclose(std[5], 0.5, rtol=0.1)  # the hinge after slide + ball
    np.testing.assert_array_equal(noisy[:, :5], qpos[:, :5])
