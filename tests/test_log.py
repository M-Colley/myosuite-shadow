"""Session logs: every frame and pose recorded, read back exactly, with provenance."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite_shadow import ShadowModel, ShadowModelConfig, read_session
from myosuite_shadow.flags import Flag
from myosuite_shadow.log import ESTIMATE_COLUMNS, model_sha256, package_provenance


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
    assert header[: len(ESTIMATE_COLUMNS)] == ESTIMATE_COLUMNS
    assert header[len(ESTIMATE_COLUMNS) :] == [f"a_{n}" for n in names] + [
        f"mf_{n}" for n in names
    ] + [f"contact_{body}_{x}" for x in "xyz"]
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
    np.testing.assert_array_equal(
        s.columns["latest_time"], [f.latest_time for f in frames]
    )
    np.testing.assert_array_equal(
        s.columns["frame_unix"], [f.frame_unix for f in frames]
    )
    assert (s.columns["done_unix"] >= s.columns["frame_unix"]).all()
    np.testing.assert_array_equal(s.flags, [int(f.flags) for f in frames])
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


def test_events_and_wall_clock_times(elbow: mujoco.MjModel, tmp_path: Path) -> None:
    with ShadowModel(elbow, ShadowModelConfig(window=5), log_dir=tmp_path) as shadow:
        shadow.mark("session_start", condition="B")
        for k in range(10):
            flags = Flag.LOW_CONFIDENCE if k == 3 else Flag.NONE
            shadow.push(
                k / 20, np.array([1.2]), flags=flags, capture_unix=1000.0 + k / 20
            )
        shadow.mark("set_end", set=1, unix=2000.0)
    s = read_session(tmp_path)
    assert [e["label"] for e in s.events] == ["session_start", "set_end"]
    assert s.events[0]["session_time"] is None and s.events[1]["session_time"] == 0.45
    assert s.events[0]["data"] == {"condition": "B"} and s.events[1]["unix"] == 2000.0
    np.testing.assert_array_equal(
        s.input_columns["capture_unix"], 1000 + np.arange(10) / 20
    )
    assert s.input_columns["flags"][3] == Flag.LOW_CONFIDENCE
    # Each estimate is stamped with its own frame's capture time (lag 2 frames).
    np.testing.assert_allclose(s.columns["frame_unix"], 1000 + s.time)
    assert s.checkpoint["closed"]


def test_a_session_directory_is_not_overwritten(
    elbow: mujoco.MjModel, tmp_path: Path
) -> None:
    ShadowModel(elbow, log_dir=tmp_path).close()
    with pytest.raises(FileExistsError, match="resume"):
        ShadowModel(elbow, log_dir=tmp_path)
