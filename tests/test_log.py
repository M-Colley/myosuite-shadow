"""Session logs: every frame and pose recorded, read back exactly, with provenance."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite_shadow import ShadowModel, ShadowModelConfig, read_session
from myosuite_shadow.log import model_sha256, package_provenance


def _record_session(model: mujoco.MjModel, log_dir: Path, **kwargs):
    body = model.body(model.nbody - 1).name
    config = ShadowModelConfig(window=5, log_contact_bodies=(body,))
    rng = np.random.default_rng(1)
    with ShadowModel(model, config, log_dir=log_dir, **kwargs) as shadow:
        frames = [
            shadow.push(
                k / 20, 1.2 + 0.01 * rng.standard_normal(1), rng.uniform(-5, 0, 1)
            )
            for k in range(30)
        ]
    return body, frames


def test_log_records_every_frame(elbow: mujoco.MjModel, tmp_path: Path) -> None:
    body, frames = _record_session(elbow, tmp_path, metadata={"participant": "P00"})
    with open(tmp_path / "estimates.csv", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    assert len(rows) == sum(f is not None for f in frames) == 26
    names = [elbow.actuator(i).name for i in range(elbow.nu)]
    assert header[5:] == [f"a_{n}" for n in names] + [f"mf_{n}" for n in names] + [
        f"contact_{body}_{x}" for x in "xyz"
    ]
    assert all(len(r) == len(header) for r in rows)
    record = json.loads((tmp_path / "record.json").read_text(encoding="utf-8"))
    assert record["model_sha256"] == model_sha256(elbow)
    assert record["metadata"] == {"participant": "P00"}
    assert record["config"]["window"] == 5
    assert record["model"]["actuators"] == names
    assert record["packages"]["myosuite"]["version"] is not None


def test_session_reads_back_exactly(elbow: mujoco.MjModel, tmp_path: Path) -> None:
    body, frames = _record_session(elbow, tmp_path)
    frames = [f for f in frames if f is not None]
    s = read_session(tmp_path)
    np.testing.assert_array_equal(s.time, [f.state.time for f in frames])
    np.testing.assert_array_equal(s.latest_time, [f.latest_time for f in frames])
    np.testing.assert_array_equal(
        s.activation, [f.optimization.activation for f in frames]
    )
    np.testing.assert_array_equal(s.fatigue, [f.fatigue for f in frames])
    np.testing.assert_array_equal(
        s.contact_force[body],
        [f.optimization.contact_force[elbow.nbody - 1] for f in frames],
    )
    assert (s.status == "solved").all()
    # Inputs: every pose, including the 4 that only filled the window.
    assert s.input_qpos.shape == (30, elbow.nq) and s.input_qfrc.shape == (30, elbow.nv)
    np.testing.assert_array_equal(s.input_time, np.arange(30) / 20)
    model = s.load_model(tmp_path)
    assert model_sha256(model) == model_sha256(elbow)


def test_replaying_the_inputs_reproduces_the_estimates(
    elbow: mujoco.MjModel, tmp_path: Path
) -> None:
    _record_session(elbow, tmp_path / "live")
    live = read_session(tmp_path / "live")
    config = ShadowModelConfig.from_dict(live.record["config"])
    with ShadowModel(
        live.load_model(tmp_path / "live"), config, log_dir=tmp_path / "replay"
    ) as shadow:
        for t, q, f in zip(live.input_time, live.input_qpos, live.input_qfrc):
            shadow.push(t, q, f)
    replay = read_session(tmp_path / "replay")
    np.testing.assert_array_equal(replay.activation, live.activation)
    np.testing.assert_array_equal(replay.fatigue, live.fatigue)


def test_tampered_model_is_rejected(elbow: mujoco.MjModel, tmp_path: Path) -> None:
    _record_session(elbow, tmp_path)
    other = mujoco.MjModel.from_binary_path(str(tmp_path / "model.mjb"))
    other.body_mass[-1] *= 1.1
    mujoco.mj_saveModel(other, str(tmp_path / "model.mjb"), None)
    with pytest.raises(ValueError, match="model_sha256"):
        read_session(tmp_path).load_model(tmp_path)


def test_rejected_pose_is_not_logged(elbow: mujoco.MjModel, tmp_path: Path) -> None:
    with ShadowModel(elbow, log_dir=tmp_path, save_model=False) as shadow:
        shadow.push(0.0, np.array([1.0]))
        with pytest.raises(ValueError, match="increase"):
            shadow.push(0.0, np.array([1.0]))
        with pytest.raises(ValueError, match="finite"):
            shadow.push(0.1, np.array([np.nan]))
    s = read_session(tmp_path)
    assert s.input_qpos.shape == (1, 1) and not (tmp_path / "model.mjb").exists()


def test_provenance_records_versions() -> None:
    p = package_provenance(("numpy", "not-a-package-xyz"))
    assert p["numpy"]["version"] == np.__version__
    assert p["not-a-package-xyz"] == {"version": None}
