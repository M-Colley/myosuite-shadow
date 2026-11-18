"""Free-root support policy and mixed actuator loads at the pipeline boundary."""

import dataclasses
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from myosuite.biomechanics.static_optimization import StaticOptimizationResult
from myosuite.core.muscle_conditions import CumulativeFatigue

from myosuite_shadow import ShadowModel, ShadowModelConfig, pipeline
from myosuite_shadow.flags import Flag


@pytest.fixture(scope="module")
def mixed_model():
    return mujoco.MjModel.from_xml_string(
        """<mujoco>
        <worldbody>
          <site name="origin" pos="0 0 1"/>
          <body name="root" pos="0 0 1">
            <freejoint/>
            <geom type="sphere" size=".1" mass="1"/>
            <body name="limb">
              <joint name="hinge" axis="0 1 0"/>
              <geom type="capsule" fromto="0 0 0 .3 0 0" size=".03" mass="1"/>
              <site name="insertion" pos=".3 0 0"/>
            </body>
          </body>
        </worldbody>
        <tendon><spatial name="tendon">
          <site site="origin"/><site site="insertion"/>
        </spatial></tendon>
        <actuator>
          <muscle name="muscle" tendon="tendon" force="100" lengthrange=".05 2"/>
          <motor name="assist" joint="hinge" ctrllimited="true" ctrlrange="-3 3"/>
        </actuator>
        </mujoco>"""
    )


def candidate(model, *, muscle=0.25, motor=0.0, root=(0.0,) * 6):
    reserve = np.zeros(model.nv)
    reserve[:6] = root
    return StaticOptimizationResult(
        activation=np.array([muscle, motor]),
        actuator_force=np.zeros(model.nu),
        reserve=reserve,
        constraint_force=np.zeros(model.nv),
        joint_torque=reserve.copy(),
        status=np.asarray("solved"),
        contact_force=np.zeros((model.nbody, 3)),
        contact_qfrc=np.zeros(model.nv),
    )


def first_frame(shadow, model):
    for index in range(5):
        frame = shadow.push(index / 30, model.qpos0)
    assert frame is not None
    return frame


def test_free_root_estimate_reports_push_compute_time(mixed_model, monkeypatch):
    clock = iter([50.0] * 5 + [50.012])
    with ShadowModel(mixed_model, ShadowModelConfig(window=5, fatigue=False)) as shadow:
        monkeypatch.setattr(
            pipeline, "time", SimpleNamespace(perf_counter=lambda: next(clock))
        )
        monkeypatch.setattr(
            shadow.optimizer, "solve", lambda *args, **kwargs: candidate(mixed_model)
        )
        assert first_frame(shadow, mixed_model).compute_ms == pytest.approx(12.0)


@pytest.mark.parametrize(
    "field,offset",
    [("root_residual_force_limit", 0), ("root_residual_moment_limit", 3)],
)
@pytest.mark.parametrize("limit,exceeded", [(4.99, True), (5.0, False), (5.01, False)])
def test_root_support_uses_vector_norm_and_inclusive_threshold(
    mixed_model, monkeypatch, field, offset, limit, exceeded
):
    root = np.zeros(6)
    root[offset : offset + 3] = [3.0, 4.0, 0.0]
    config = ShadowModelConfig(window=5, fatigue=False, **{field: limit})
    with ShadowModel(mixed_model, config) as shadow:
        monkeypatch.setattr(
            shadow.optimizer, "solve", lambda *a, **k: candidate(mixed_model, root=root)
        )
        frame = first_frame(shadow, mixed_model)
    assert bool(frame.flags & Flag.ROOT_RESIDUAL) == exceeded
    assert not frame.flags & Flag.NOT_SOLVED
    assert frame.optimization.status == "solved"


def test_force_and_moment_are_separate_norms(mixed_model, monkeypatch):
    config = ShadowModelConfig(
        window=5,
        fatigue=False,
        root_residual_force_limit=5.0,
        root_residual_moment_limit=5.0,
    )
    with ShadowModel(mixed_model, config) as shadow:
        monkeypatch.setattr(
            shadow.optimizer,
            "solve",
            lambda *a, **k: candidate(mixed_model, root=(3, 4, 0, 3, 4, 0)),
        )
        assert first_frame(shadow, mixed_model).flags == Flag.NONE


def test_unconfigured_root_residual_limits_remain_diagnostic(mixed_model, monkeypatch):
    config = ShadowModelConfig(window=5, fatigue=False)
    assert config.root_residual_force_limit is None
    assert config.root_residual_moment_limit is None
    with ShadowModel(mixed_model, config) as shadow:
        monkeypatch.setattr(
            shadow.optimizer,
            "solve",
            lambda *a, **k: candidate(mixed_model, root=(1000,) * 6),
        )
        assert first_frame(shadow, mixed_model).flags == Flag.NONE


@pytest.mark.parametrize(
    "field,offset",
    [("root_residual_force_limit", 0), ("root_residual_moment_limit", 3)],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_root_residual_is_rejected(
    mixed_model, monkeypatch, field, offset, value
):
    root = np.zeros(6)
    root[offset] = value
    config = ShadowModelConfig(window=5, fatigue=False, **{field: 5.0})
    with ShadowModel(mixed_model, config) as shadow:
        monkeypatch.setattr(
            shadow.optimizer, "solve", lambda *a, **k: candidate(mixed_model, root=root)
        )
        frame = first_frame(shadow, mixed_model)
    assert frame.flags & Flag.ROOT_RESIDUAL
    assert not frame.flags & Flag.NOT_SOLVED


@pytest.mark.parametrize("policy", ["rest", "hold"])
@pytest.mark.parametrize(
    "field,offset",
    [("root_residual_force_limit", 0), ("root_residual_moment_limit", 3)],
)
def test_unsupported_load_uses_gap_assumption_and_preserves_trusted_load(
    mixed_model, monkeypatch, policy, field, offset
):
    config = ShadowModelConfig(window=5, gap_load=policy, **{field: 4.99})
    current = [candidate(mixed_model)]
    with ShadowModel(mixed_model, config) as shadow:
        monkeypatch.setattr(shadow.optimizer, "solve", lambda *a, **k: current[0])
        first_frame(shadow, mixed_model)
        previous = shadow.push(5 / 30, mixed_model.qpos0)
        reference = CumulativeFatigue(mixed_model)
        reference.load_state_dict(shadow.fatigue.state_dict())
        trusted = shadow._load.copy()
        root = np.zeros(6)
        root[offset : offset + 3] = [3.0, 4.0, 0.0]
        current[0] = candidate(mixed_model, muscle=0.95, root=root)
        rejected = shadow.push(6 / 30, mixed_model.qpos0)
        dt = rejected.state.time - previous.state.time
        reference.compute_act(
            trusted if policy == "hold" else np.zeros_like(trusted), dt=dt
        )
        np.testing.assert_allclose(
            rejected.fatigue[reference.muscle_ids], reference.MF, atol=1e-12, rtol=0
        )
        np.testing.assert_array_equal(shadow._load, trusted)
        assert rejected.flags & Flag.ROOT_RESIDUAL
        assert rejected.flags & Flag.AFTER_GAP
        assert not rejected.flags & Flag.NOT_SOLVED
        assert rejected.gap == pytest.approx(dt)
        current[0] = candidate(mixed_model, muscle=0.4)
        recovered = shadow.push(7 / 30, mixed_model.qpos0)
        reference.compute_act(
            np.full_like(trusted, 0.4), dt=recovered.state.time - rejected.state.time
        )
        np.testing.assert_allclose(
            recovered.fatigue[reference.muscle_ids], reference.MF, atol=1e-12, rtol=0
        )
        assert not recovered.flags & (Flag.ROOT_RESIDUAL | Flag.NOT_SOLVED)


@pytest.mark.parametrize(
    "field", ["root_residual_force_limit", "root_residual_moment_limit"]
)
@pytest.mark.parametrize("value", [-0.01, float("nan"), float("inf"), -float("inf")])
def test_invalid_root_residual_limits(field, value):
    with pytest.raises(ValueError, match="root residual limits"):
        ShadowModelConfig(**{field: value})


def test_zero_root_limits_survive_config_round_trip():
    config = ShadowModelConfig(
        root_residual_force_limit=0.0, root_residual_moment_limit=0.0
    )
    restored = ShadowModelConfig.from_dict(dataclasses.asdict(config))
    assert restored.root_residual_force_limit == 0.0
    assert restored.root_residual_moment_limit == 0.0


@pytest.mark.parametrize("motor", [-2.0, 2.0])
def test_valid_motor_controls_do_not_invalidate_muscle_loading(
    mixed_model, monkeypatch, motor
):
    with ShadowModel(mixed_model, ShadowModelConfig(window=5)) as shadow:
        monkeypatch.setattr(
            shadow.optimizer,
            "solve",
            lambda *a, **k: candidate(mixed_model, muscle=0.6, motor=motor),
        )
        previous = first_frame(shadow, mixed_model)
        frame = shadow.push(5 / 30, mixed_model.qpos0)
        reference = CumulativeFatigue(mixed_model)
        reference.compute_act(
            np.array([0.6]), dt=frame.state.time - previous.state.time
        )
        assert frame.flags == Flag.NONE
        np.testing.assert_array_equal(shadow._load, [0.6])
        np.testing.assert_allclose(
            frame.fatigue[reference.muscle_ids], reference.MF, atol=1e-12, rtol=0
        )


@pytest.mark.parametrize("muscle", [-0.1, 1.1])
def test_muscle_bounds_still_apply_with_valid_motor_controls(
    mixed_model, monkeypatch, muscle
):
    with ShadowModel(mixed_model, ShadowModelConfig(window=5)) as shadow:
        monkeypatch.setattr(
            shadow.optimizer,
            "solve",
            lambda *a, **k: candidate(mixed_model, muscle=muscle, motor=-2.0),
        )
        assert first_frame(shadow, mixed_model).flags & Flag.NOT_SOLVED
