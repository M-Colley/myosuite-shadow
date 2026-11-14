"""Unsolved/nonfinite candidates must never become fatigue targets."""

import dataclasses

import numpy as np
import pytest
from myosuite.core.muscle_conditions import CumulativeFatigue

from myosuite_shadow import ShadowModel, ShadowModelConfig
from myosuite_shadow.flags import Flag


@pytest.mark.parametrize("policy", ["rest", "hold"])
@pytest.mark.parametrize("failure", ["unsolved", "nonfinite", "bounds"])
def test_failed_candidate_uses_gap_assumption(elbow, policy, failure):
    with ShadowModel(elbow, ShadowModelConfig(window=5, gap_load=policy)) as shadow:
        previous = None
        for k in range(10):
            previous = shadow.push(k / 30, np.array([1.2]), np.array([-20.0]))
        ref = CumulativeFatigue(elbow)
        ref.load_state_dict(shadow.fatigue.state_dict())
        trusted = shadow._load.copy()
        solve = shadow.optimizer.solve

        def failed(*args, **kwargs):
            return dataclasses.replace(
                solve(*args, **kwargs),
                status="iteration_limit" if failure == "unsolved" else "solved",
                activation=np.full(elbow.nu, np.nan if failure == "nonfinite" else 2.0),
            )

        shadow.optimizer.solve = failed
        frame = shadow.push(10 / 30, np.array([1.2]), np.array([-20.0]))
        dt = frame.state.time - previous.state.time
        ref.compute_act(trusted if policy == "hold" else np.zeros_like(trusted), dt=dt)
        np.testing.assert_allclose(frame.fatigue[ref.muscle_ids], ref.MF, atol=1e-12)
        np.testing.assert_array_equal(shadow._load, trusted)
        assert frame.flags & Flag.NOT_SOLVED and frame.flags & Flag.AFTER_GAP
        assert frame.gap == pytest.approx(dt)
        shadow.optimizer.solve = solve
        recovered = shadow.push(11 / 30, np.array([1.2]), np.array([-20.0]))
        assert not recovered.flags & Flag.NOT_SOLVED
        assert np.isfinite(recovered.fatigue).all()


def test_initial_failure_has_no_fabricated_load(elbow):
    with ShadowModel(elbow, ShadowModelConfig(window=5, gap_load="hold")) as shadow:
        solve = shadow.optimizer.solve

        def failed(*args, **kwargs):
            return dataclasses.replace(
                solve(*args, **kwargs),
                status="not_solved",
                activation=np.full(elbow.nu, np.nan),
            )

        shadow.optimizer.solve = failed
        for k in range(6):
            frame = shadow.push(k / 30, np.array([1.2]))
        assert frame is not None and frame.flags & Flag.NOT_SOLVED
        assert np.isfinite(frame.fatigue).all()
        np.testing.assert_array_equal(shadow._load, 0.0)
