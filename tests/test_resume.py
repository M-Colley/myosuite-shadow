"""A crashed session resumes from its checkpoint as if uninterrupted."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite_shadow import ShadowModel, ShadowModelConfig, read_session
from myosuite_shadow.log import INPUTS

_N, _CRASH = 600, 437  # 30 s at 20 Hz; the crash comes 1.85 s after a checkpoint


def _pose(k: int) -> tuple[float, np.ndarray, np.ndarray]:
    t = k / 20
    return t, np.array([1.2 + 0.3 * np.sin(0.5 * t)]), np.array([-15.0 - 5 * np.sin(t)])


def _crash(shadow: ShadowModel) -> None:
    """Stop without a final checkpoint and leave a torn line, like a killed process."""
    for f in shadow.log._files.values():
        f.close()
    with open(shadow.log.log_dir / INPUTS, "ab") as f:
        f.write(b"21.9,17000")


@pytest.fixture()
def config() -> ShadowModelConfig:
    return ShadowModelConfig(window=5, checkpoint_every=5.0)


def test_resume_matches_an_uninterrupted_session(
    elbow: mujoco.MjModel, tmp_path: Path, config: ShadowModelConfig
) -> None:
    with ShadowModel(elbow, config, log_dir=tmp_path / "a") as shadow:
        for k in range(_N):
            shadow.push(*_pose(k))
    crashed = ShadowModel(elbow, config, log_dir=tmp_path / "b")
    for k in range(_CRASH):
        crashed.push(*_pose(k))
    _crash(crashed)
    with ShadowModel.resume(tmp_path / "b") as shadow:
        for k in range(_CRASH, _N):
            shadow.push(*_pose(k))
    a, b = read_session(tmp_path / "a"), read_session(tmp_path / "b")
    np.testing.assert_array_equal(b.input_time, a.input_time)
    np.testing.assert_array_equal(b.input_qpos, a.input_qpos)
    np.testing.assert_array_equal(b.time, a.time)
    # Only the optimizer's warm start differs after the restart.
    np.testing.assert_allclose(b.activation, a.activation, atol=1e-6)
    np.testing.assert_allclose(b.fatigue, a.fatigue, atol=1e-8)
    assert b.fatigue[-1].max() > 0.1
    assert len(b.record["resumes"]) == 1
    assert b.record["resumes"][0]["replayed_inputs"] == _CRASH - 401
    assert [e["label"] for e in b.events] == ["resumed"]


def test_downtime_after_a_crash_is_a_rest(
    elbow: mujoco.MjModel, tmp_path: Path, config: ShadowModelConfig
) -> None:
    crashed = ShadowModel(elbow, config, log_dir=tmp_path)
    frames = [crashed.push(*_pose(k)) for k in range(_CRASH)]
    _crash(crashed)
    with ShadowModel.resume(tmp_path) as shadow:
        # The participant rested for a minute while the software restarted.
        later = [shadow.push(60.0 + k / 20, *_pose(k)[1:]) for k in range(5)][-1]
    assert later.gap == pytest.approx(60.0 - (_CRASH - 1) / 20)
    assert later.fatigue.max() < frames[-1].fatigue.max()


def test_resume_rejects_another_model(
    elbow: mujoco.MjModel, arm: mujoco.MjModel, tmp_path: Path
) -> None:
    ShadowModel(elbow, log_dir=tmp_path).close()
    with pytest.raises(ValueError, match="model_sha256"):
        ShadowModel.resume(tmp_path, arm)


def test_a_session_can_resume_twice(
    elbow: mujoco.MjModel, tmp_path: Path, config: ShadowModelConfig
) -> None:
    shadow = ShadowModel(elbow, config, log_dir=tmp_path)
    for k in range(200):
        shadow.push(*_pose(k))
    _crash(shadow)
    shadow = ShadowModel.resume(tmp_path)
    for k in range(200, 300):
        shadow.push(*_pose(k))
    _crash(shadow)
    with ShadowModel.resume(tmp_path) as shadow:
        for k in range(300, 320):
            shadow.push(*_pose(k))
    s = read_session(tmp_path)
    assert len(s.record["resumes"]) == 2 and len(s.input_time) == 320
    np.testing.assert_array_equal(s.input_time, np.arange(320) / 20)
    assert np.all(np.diff(s.time) > 0)
