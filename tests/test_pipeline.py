"""Shadow model: composition, fatigue feedback, tracking gaps and external loads."""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np
import pytest
from myosuite.core.muscle_conditions import CumulativeFatigue
from myosuite.physics.static_optimization import (
    StaticOptimizationConfig,
    StaticOptimizer,
)
from myosuite.physics.streaming_kinematics import StreamingKinematics

from myosuite_shadow import ShadowModel, ShadowModelConfig
from myosuite_shadow.flags import Flag


def _hold(model, load: float, limits: bool, seconds: float = 120.0, rate: float = 20.0):
    """Hold the forearm at 69 degrees against a hand load [N m]."""
    shadow = ShadowModel(
        model, ShadowModelConfig(window=5, fatigue_limits_activation=limits)
    )
    frames = [
        shadow.push(k / rate, np.array([1.2]), np.array([load]))
        for k in range(int(seconds * rate))
    ]
    frames = [f for f in frames if f is not None]
    act = np.array([f.optimization.activation for f in frames])
    mf = np.array([f.fatigue for f in frames])
    return act, mf, np.array([f.optimization.reserve[0] for f in frames])


def test_shadow_model_equals_its_parts(elbow: mujoco.MjModel) -> None:
    t = np.arange(0.0, 2.0, 1 / 90)
    qpos = (1.0 + 0.6 * np.sin(2.0 * t))[:, None]
    shadow = ShadowModel(elbow, ShadowModelConfig(fatigue=False))
    stream, so = StreamingKinematics(elbow, window=15), StaticOptimizer(elbow)
    for ti, q in zip(t, qpos):
        frame, state = shadow.push(ti, q), stream.push(ti, q)
        assert (frame is None) == (state is None)
        if state is not None:
            assert frame.state.time == state.time and frame.fatigue is None
            assert frame.delay == pytest.approx(7 / 90)
            np.testing.assert_array_equal(
                frame.optimization.activation,
                so.solve(state.qpos, state.qvel, state.qacc).activation,
            )


def test_fatigue_caps_activation_and_shifts_the_load(elbow: mujoco.MjModel) -> None:
    act, mf, reserve = _hold(elbow, -20.0, limits=True)
    act0, mf0, reserve0 = _hold(elbow, -20.0, limits=False)
    assert mf[-1].max() > 0.5  # two minutes at about half of maximum
    # A fatigued muscle never exceeds what it has left; without the cap it does.
    assert (act[1:] - (1.0 - mf[:-1])).max() < 1e-6
    assert (act0[1:] - (1.0 - mf0[:-1])).max() > 0.15
    bic_long, bra = elbow.actuator("BIClong").id, elbow.actuator("BRA").id
    assert act[-1, bic_long] < act0[-1, bic_long] - 0.1
    assert act[-1, bra] > act0[-1, bra] + 0.05
    assert reserve[-1] > 1.0 > reserve0[-1]  # the fatigued arm starts to fall short


def test_fatigue_integrates_the_time_between_estimates(elbow: mujoco.MjModel) -> None:
    # Jittered 30-70 Hz poses: fatigue must step by each estimate's own interval.
    rng = np.random.default_rng(0)
    times = np.cumsum(rng.uniform(1 / 70, 1 / 30, 400))
    shadow = ShadowModel(elbow, ShadowModelConfig(window=5))
    frames = [shadow.push(t, np.array([1.2]), np.array([-20.0])) for t in times]
    frames = [f for f in frames if f is not None]
    ref = CumulativeFatigue(elbow)
    ids = ref.muscle_ids
    for prev, frame in zip(frames, frames[1:]):
        ref.compute_act(
            frame.optimization.activation[ids], dt=frame.state.time - prev.state.time
        )
        np.testing.assert_allclose(frame.fatigue[ids], ref.MF, rtol=0, atol=1e-12)
    assert frames[-1].fatigue.max() > 0.05  # ten seconds of holding


def _reference_fatigue(model, frames) -> CumulativeFatigue:
    """3CC-r driven by each estimate's activations over the interval before it."""
    ref = CumulativeFatigue(model)
    for prev, f in zip(frames, frames[1:]):
        ref.compute_act(
            f.optimization.activation[ref.muscle_ids], dt=f.state.time - prev.state.time
        )
    return ref


@pytest.mark.parametrize("gap_load", ["rest", "hold"])
def test_gap_integrates_the_untracked_time(
    elbow: mujoco.MjModel, gap_load: str
) -> None:
    shadow = ShadowModel(elbow, ShadowModelConfig(window=5, gap_load=gap_load))
    pose, load = np.array([1.2]), np.array([-20.0])
    before = [shadow.push(k / 20, pose, load) for k in range(40)]
    after = [shadow.push(3.0 + k / 20, pose, load) for k in range(5)]  # 1.05 s gap
    before = [f for f in before if f is not None]
    assert all(f is None for f in after[:4])
    first = after[4]
    assert first.gap == pytest.approx(1.05) and Flag.AFTER_GAP in first.flags
    assert Flag.AFTER_GAP not in before[-1].flags
    ref = _reference_fatigue(elbow, before)
    last = before[-1].optimization.activation[ref.muscle_ids]
    ref.compute_act(last, dt=1.95 - before[-1].state.time)  # tracked until 1.95 s
    ref.compute_act(np.zeros_like(last) if gap_load == "rest" else last, dt=1.05)
    ref.compute_act(
        first.optimization.activation[ref.muscle_ids], dt=first.state.time - 3.0
    )
    np.testing.assert_allclose(first.fatigue[ref.muscle_ids], ref.MF, atol=1e-12)


def test_muscles_recover_during_a_rest(elbow: mujoco.MjModel) -> None:
    pose, load = np.array([1.2]), np.array([-20.0])
    out = {}
    for gap_load in ("rest", "hold"):
        shadow = ShadowModel(elbow, ShadowModelConfig(window=5, gap_load=gap_load))
        held = [shadow.push(k / 20, pose, load) for k in range(1200)][-1]  # 60 s
        # Two minutes without tracking, then the same hold again.
        out[gap_load] = [shadow.push(180.0 + k / 20, pose, load) for k in range(5)][-1]
    assert out["rest"].fatigue.max() < 0.6 * held.fatigue.max()
    assert out["hold"].fatigue.max() > held.fatigue.max() + 0.1


def test_gap_before_the_first_estimate_leaves_fatigue_alone(
    elbow: mujoco.MjModel,
) -> None:
    shadow = ShadowModel(elbow, ShadowModelConfig(window=5))
    [shadow.push(k / 20, np.array([1.2])) for k in range(3)]
    frames = [shadow.push(2.0 + k / 20, np.array([1.2])) for k in range(5)]
    assert frames[4].gap == pytest.approx(1.9) and not frames[4].fatigue.any()


def test_flags_follow_the_poses_in_the_fit(elbow: mujoco.MjModel) -> None:
    shadow = ShadowModel(elbow, ShadowModelConfig(window=5, fatigue=False))
    frames = [
        shadow.push(k / 20, np.array([1.2]), flags=Flag.GLITCH if k == 10 else 0)
        for k in range(20)
    ]
    glitch = [k for k, f in enumerate(frames) if f and Flag.GLITCH in f.flags]
    assert glitch == list(range(10, 15))  # the five fits that include pose 10
    first = next(f for f in frames if f is not None)
    assert first.flags == Flag.NONE  # a session start is not a gap


def test_capacity_and_solver_flags(elbow: mujoco.MjModel) -> None:
    shadow = ShadowModel(elbow, ShadowModelConfig(window=5, fatigue=False))
    light = [shadow.push(k / 20, np.array([1.2]), np.array([-5.0])) for k in range(5)]
    heavy = [
        shadow.push(1 + k / 20, np.array([1.2]), np.array([-200.0])) for k in range(5)
    ]
    assert Flag.CAPACITY_EXCEEDED not in light[-1].flags
    assert Flag.CAPACITY_EXCEEDED in heavy[-1].flags
    assert Flag.NOT_SOLVED not in heavy[-1].flags


def test_decimation_skips_optimizations_not_fatigue(elbow: mujoco.MjModel) -> None:
    shadow = ShadowModel(elbow, ShadowModelConfig(window=5, estimate_every=3))
    pushed = [
        shadow.push(k / 20, np.array([1.2]), np.array([-20.0])) for k in range(64)
    ]
    frames = [f for f in pushed if f is not None]
    assert [k for k, f in enumerate(pushed) if f is not None][:3] == [4, 7, 10]
    assert frames[0].skipped == 0 and all(f.skipped == 2 for f in frames[1:])
    ref = _reference_fatigue(elbow, frames)
    np.testing.assert_allclose(frames[-1].fatigue[ref.muscle_ids], ref.MF, atol=1e-12)
    # A runner behind schedule passes estimate=False: kinematics only.
    assert shadow.push(64 / 20, np.array([1.2]), estimate=False) is None
    assert shadow.push(65 / 20, np.array([1.2]), estimate=False) is None
    assert shadow.push(66 / 20, np.array([1.2])).skipped == 4  # 62, 63, 64, 65


def test_config_rejects_unknown_gap_load() -> None:
    with pytest.raises(ValueError, match="gap_load"):
        ShadowModelConfig(gap_load="freeze")


def test_external_load_belongs_to_the_estimated_frame(elbow: mujoco.MjModel) -> None:
    loaded, free = (
        ShadowModel(elbow, ShadowModelConfig(fatigue=False)),
        ShadowModel(elbow, ShadowModelConfig(fatigue=False)),
    )
    for k in range(40):
        q, ext = np.array([1.0 + 0.01 * k]), np.array([-0.5 * k])
        a, b = loaded.push(k / 50, q, ext), free.push(k / 50, q)
        if a is not None:
            k_est = round(a.state.time * 50)
            np.testing.assert_allclose(
                a.optimization.joint_torque - b.optimization.joint_torque, [0.5 * k_est]
            )


def test_zero_lag_matches_the_newest_external_load(elbow: mujoco.MjModel) -> None:
    shadow = ShadowModel(elbow, ShadowModelConfig(window=5, lag=0, fatigue=False))
    frames = [
        shadow.push(k / 20, np.array([1.2]), np.array([-float(k)])) for k in range(8)
    ]
    assert frames[3] is None and frames[4].delay == 0.0
    ref = ShadowModel(elbow, ShadowModelConfig(window=5, lag=0, fatigue=False))
    plain = [ref.push(k / 20, np.array([1.2])) for k in range(8)]
    np.testing.assert_allclose(
        frames[7].optimization.joint_torque - plain[7].optimization.joint_torque, [7.0]
    )


def test_config_round_trips_through_a_dict() -> None:
    config = ShadowModelConfig(
        window=21,
        lag=3,
        optimization=StaticOptimizationConfig(contact_forces=True),
        log_contact_bodies=("calcn_r",),
    )
    assert ShadowModelConfig.from_dict(dataclasses.asdict(config)) == config
