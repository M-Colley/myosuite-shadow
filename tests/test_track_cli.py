"""Raw acquisition replay is deterministic and resume preflight is non-destructive."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite_shadow.log import read_session
from myosuite_shadow.track_cli import main
from myosuite_shadow.tracking import RawFrameWriter, TrackerFrame


def _fixture(model: mujoco.MjModel, root: Path) -> tuple[Path, Path]:
    model_path = root / "subject.mjb"
    mujoco.mj_saveModel(model, str(model_path), None)
    settings = {
        "model": "subject.mjb",
        "keypoints": [
            {"name": "elbow", "frame": "r_ulna_radius_hand"},
            {"name": "wrist", "frame": "wrist", "frame_type": "site"},
        ],
        "shadow": {"window": 5, "fatigue": False, "checkpoint_every": 0},
    }
    config = root / "config.json"
    config.write_text(json.dumps(settings), encoding="utf-8")
    data = mujoco.MjData(model)
    data.qpos[:] = 1.2
    mujoco.mj_forward(model, data)
    raw = root / "frames.jsonl"
    with RawFrameWriter(raw) as writer:
        for k in range(20):
            writer.write(
                TrackerFrame(
                    time_s=k / 20,
                    source_id="recorded_camera",
                    sequence=k,
                    person_id="P07",
                    coordinate_frame="model_world",
                    positions={
                        "elbow": data.xpos[model.body("r_ulna_radius_hand").id],
                        "wrist": data.site_xpos[model.site("wrist").id],
                    },
                    position_tracked={"elbow": True, "wrist": True},
                )
            )
    return config, raw


def test_offline_raw_frames_have_no_machine_dependent_backlog(
    elbow: mujoco.MjModel,
    tmp_path: Path,
) -> None:
    config, raw = _fixture(elbow, tmp_path)
    for name in ("a", "b"):
        main(
            [
                "--config",
                str(config),
                "--frames",
                str(raw),
                "--log-dir",
                str(tmp_path / name),
            ]
        )
    a, b = read_session(tmp_path / "a"), read_session(tmp_path / "b")
    assert len(a.input_time) == 20 and len(a.time) == 16
    assert a.columns["skipped"].sum() == 0
    np.testing.assert_array_equal(a.activation, b.activation)
    np.testing.assert_array_equal(a.input_qpos, b.input_qpos)


def test_resume_config_is_checked_before_modifying_session_files(
    elbow: mujoco.MjModel,
    tmp_path: Path,
) -> None:
    config, raw = _fixture(elbow, tmp_path)
    log = tmp_path / "session"
    main(["--config", str(config), "--frames", str(raw), "--log-dir", str(log)])
    before = {p.name: p.read_bytes() for p in log.iterdir() if p.is_file()}
    settings = json.loads(config.read_text())
    settings["shadow"]["window"] = 9
    config.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(
            [
                "--config",
                str(config),
                "--frames",
                str(raw),
                "--log-dir",
                str(log),
                "--resume",
            ]
        )
    assert before == {p.name: p.read_bytes() for p in log.iterdir() if p.is_file()}
    settings["shadow"]["window"] = 5
    config.write_text(json.dumps(settings), encoding="utf-8")
    continuation = tmp_path / "continuation.jsonl"
    from myosuite_shadow.tracking import read_frames

    with RawFrameWriter(continuation) as writer:
        for frame in read_frames(raw):
            writer.write(
                dataclasses.replace(
                    frame, time_s=frame.time_s + 1, sequence=frame.sequence + 20
                )
            )
    main(
        [
            "--config",
            str(config),
            "--frames",
            str(continuation),
            "--log-dir",
            str(log),
            "--resume",
        ]
    )
    assert len(read_session(log).input_time) == 40


@pytest.mark.parametrize("alias_kind", ["same_path", "relative_path", "hardlink"])
def test_resume_refuses_raw_input_output_alias_before_changing_session(
    elbow: mujoco.MjModel, tmp_path: Path, monkeypatch, alias_kind: str
) -> None:
    config, _ = _fixture(elbow, tmp_path)
    missing = tmp_path / "missing.jsonl"
    with RawFrameWriter(missing) as writer:
        for index in range(2):
            writer.write(
                TrackerFrame(
                    index, "camera", index, "P07", {}, coordinate_frame="model_world"
                )
            )
    log = tmp_path / "session"
    main(["--config", str(config), "--frames", str(missing), "--log-dir", str(log)])
    tracked = log / "tracking.jsonl"
    if alias_kind == "hardlink":
        source = tmp_path / "tracking-alias.jsonl"
        source.hardlink_to(tracked)
        assert source.resolve() != tracked.resolve()
    elif alias_kind == "relative_path":
        monkeypatch.chdir(tmp_path)
        source = Path("session") / "tracking.jsonl"
    else:
        source = tracked
    before = {p.name: p.read_bytes() for p in log.iterdir() if p.is_file()}
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--config",
                str(config),
                "--frames",
                str(source),
                "--log-dir",
                str(log),
                "--resume",
                "--max-frames",
                "5",
            ]
        )
    assert error.value.code == 2
    assert before == {p.name: p.read_bytes() for p in log.iterdir() if p.is_file()}
    assert len(tracked.read_text().splitlines()) == 2
