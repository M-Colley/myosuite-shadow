"""Calibration: MVC strength, body mass and 3CC-r rates from endurance times."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from myosuite.core.muscle_conditions import CumulativeFatigue

from myosuite_shadow import ShadowModel, ShadowModelConfig, read_session
from myosuite_shadow.calibration import (
    endurance_time,
    fatigue_params,
    fit_fatigue_parameters,
    max_joint_torque,
    posture_at,
    scale_body_mass,
    scale_strength,
    strength_from_mvc,
)

_ELBOW_3CCR = (0.01086, 0.00225, 4.93)  # MyoSuite's "Elbow" row


def test_strength_from_mvc_matches_the_measurement(elbow: mujoco.MjModel) -> None:
    qpos = posture_at(elbow, {"r_elbow_flex": np.radians(90)})
    model_flexion, flexors = max_joint_torque(elbow, qpos, "r_elbow_flex", 1.0)
    assert set(flexors) == {"BIClong", "BICshort", "BRA"}
    cal = strength_from_mvc(elbow, qpos, "r_elbow_flex", 1.3 * model_flexion)
    original_force = elbow.actuator_gainprm[0, 2]
    scaled = scale_strength(elbow, cal.scale, cal.agonists)
    assert max_joint_torque(scaled, qpos, "r_elbow_flex", 1.0)[0] == pytest.approx(
        cal.measured_torque, rel=1e-9
    )
    # Extension MVCs calibrate against the triceps.
    ext = strength_from_mvc(elbow, qpos, "r_elbow_flex", -30.0)
    assert set(ext.agonists) == {"TRIlong", "TRIlat", "TRImed"} and ext.scale > 0
    assert elbow.actuator_gainprm[0, 2] == original_force


def test_a_stronger_model_needs_less_activation(elbow: mujoco.MjModel) -> None:
    def hold(model: mujoco.MjModel) -> np.ndarray:
        shadow = ShadowModel(model, ShadowModelConfig(window=5, fatigue=False))
        frames = [
            shadow.push(k / 20, np.array([1.2]), np.array([-20.0])) for k in range(5)
        ]
        return frames[-1].optimization.activation

    weak, strong = hold(elbow), hold(scale_strength(elbow, 2.0))
    assert strong.max() < 0.6 * weak.max()


def test_scale_body_mass_keeps_proportions(arm: mujoco.MjModel) -> None:
    heavy = scale_body_mass(arm, 2.0 * arm.body_mass[1:].sum())
    np.testing.assert_allclose(heavy.body_mass[1:], 2 * arm.body_mass[1:])
    np.testing.assert_allclose(heavy.body_inertia[1:], 2 * arm.body_inertia[1:])


def test_endurance_time_follows_3ccr() -> None:
    loads = [0.3, 0.5, 0.7]
    et = endurance_time(loads, *_ELBOW_3CCR)
    assert np.all(np.diff(et) < 0) and 250 < et[0] < 350 and 30 < et[2] < 50
    # The interpolated crossing does not depend on the check step.
    np.testing.assert_allclose(
        endurance_time(loads, *_ELBOW_3CCR, dt=0.05), et, atol=0.01
    )
    assert np.isinf(endurance_time([0.01], *_ELBOW_3CCR, max_time=60.0)[0])


def test_fit_recovers_the_fatigue_rate() -> None:
    loads = [0.3, 0.5, 0.7]
    observed = endurance_time(loads, *_ELBOW_3CCR)
    fit = fit_fatigue_parameters(loads, observed, F=0.02, R=0.00225, r=4.93)
    assert fit.F == pytest.approx(_ELBOW_3CCR[0], rel=1e-3)
    assert fit.rmse_log < 1e-3 and fit.fitted == ("F",)


def test_fitted_rates_reach_the_shadow_model(
    elbow: mujoco.MjModel, tmp_path: Path
) -> None:
    fit = fit_fatigue_parameters(
        [0.3, 0.6],
        endurance_time([0.3, 0.6], 0.02, 0.002, 4.93),
        F=0.01,
        R=0.002,
        r=4.93,
    )
    config = ShadowModelConfig(
        fatigue_params=fatigue_params(elbow, ["BIClong", "BRA"], fit)
    )
    with ShadowModel(
        elbow, config, log_dir=tmp_path, calibration={"fatigue": fit.record()}
    ) as shadow:
        f = shadow.fatigue
        ids = [
            list(f.muscle_ids).index(elbow.actuator(n).id) for n in ("BIClong", "BRA")
        ]
        np.testing.assert_allclose(np.asarray(f.F)[ids], fit.F)
        default = CumulativeFatigue(elbow)
        other = list(f.muscle_ids).index(elbow.actuator("TRIlong").id)
        assert f.F[other] == default.F[other]
    record = read_session(tmp_path).record
    assert record["calibration"]["fatigue"]["F"] == fit.F
    assert ShadowModelConfig.from_dict(record["config"]) == shadow.config
