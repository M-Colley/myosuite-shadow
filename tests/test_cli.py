"""``myosuite-shadow replay`` from reaches, model files and recorded sessions."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite_shadow import ShadowModel, ShadowModelConfig, read_session
from myosuite_shadow.cli import main

_ELBOW = ["--env", "myoElbowPose1D6MRandom-v0", "--duration", "2"]


def test_replay_reaches(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    main(["replay", *_ELBOW, "--log-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "166 frames at 90 Hz, delay 78 ms" in out and "solved 166/166" in out
    assert read_session(tmp_path).record["metadata"]["replay"]["noise_deg"] == 0.5


def test_model_file_matches_the_env(
    elbow: mujoco.MjModel, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    mujoco.mj_saveModel(elbow, str(tmp_path / "elbow.mjb"), None)
    main(["replay", *_ELBOW, "--log-dir", str(tmp_path / "env")])
    main(
        [
            "replay",
            "--model",
            str(tmp_path / "elbow.mjb"),
            "--duration",
            "2",
            "--log-dir",
            str(tmp_path / "file"),
        ]
    )
    a, b = read_session(tmp_path / "env"), read_session(tmp_path / "file")
    np.testing.assert_array_equal(a.activation, b.activation)


def test_session_replay_reproduces_and_overrides(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    main(["replay", *_ELBOW, "--window", "9", "--log-dir", str(tmp_path / "live")])
    capsys.readouterr()
    # The recorded settings, model and poses are reused: no new noise, same window.
    main(
        [
            "replay",
            "--session",
            str(tmp_path / "live"),
            "--log-dir",
            str(tmp_path / "same"),
        ]
    )
    assert "delay 44 ms" in capsys.readouterr().out
    live, same = read_session(tmp_path / "live"), read_session(tmp_path / "same")
    np.testing.assert_array_equal(same.input_qpos, live.input_qpos)
    np.testing.assert_array_equal(same.activation, live.activation)
    np.testing.assert_array_equal(same.fatigue, live.fatigue)
    main(["replay", "--session", str(tmp_path / "live"), "--window", "21"])
    assert "delay 111 ms" in capsys.readouterr().out


def test_session_replay_keeps_external_loads(
    elbow: mujoco.MjModel, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    with ShadowModel(
        elbow, ShadowModelConfig(window=5), log_dir=tmp_path / "live"
    ) as s:
        for k in range(60):
            s.push(k / 20, np.array([1.2]), np.array([-0.5 * k]))
    main(
        [
            "replay",
            "--session",
            str(tmp_path / "live"),
            "--log-dir",
            str(tmp_path / "re"),
        ]
    )
    live, again = read_session(tmp_path / "live"), read_session(tmp_path / "re")
    np.testing.assert_array_equal(again.activation, live.activation)
    np.testing.assert_array_equal(again.input_qfrc, live.input_qfrc)


def test_session_without_model_needs_one(tmp_path: Path) -> None:
    main(["replay", *_ELBOW, "--log-dir", str(tmp_path)])
    (tmp_path / "model.mjb").unlink()
    with pytest.raises(SystemExit):
        main(["replay", "--session", str(tmp_path)])
