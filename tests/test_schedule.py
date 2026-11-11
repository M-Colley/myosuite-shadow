"""Recorded live decisions survive replay, resets and a real checkpoint rollback."""

from __future__ import annotations

import csv
import dataclasses
import io
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from myosuite.biomechanics.static_optimization import StaticOptimizer

from myosuite_shadow import ShadowModel, ShadowModelConfig, read_session
from myosuite_shadow.calibration import scale_strength
from myosuite_shadow.cli import main
from myosuite_shadow.flags import Flag
from myosuite_shadow.log import ESTIMATES, OPERATIONS
from myosuite_shadow.sensitivity import Variant, run_sensitivity


def _push(shadow: ShadowModel, k: int) -> None:
    if k in (11, 29):
        shadow.reset_tracking()
    shadow.push(
        k / 20,
        [1.2 + 0.1 * np.sin(k / 10)],
        [-15.0],
        estimate=k % 5 == 0 or k == 59,
        flags=Flag.LOW_CONFIDENCE if k == 43 else Flag.NONE,
        receive_unix=1000 + k / 20 + 0.02,
        capture_unix=1000 + k / 20 if k % 2 == 0 else None,
        evaluation_unix=1000 + k / 20 + 0.005,
    )


def _new(model: mujoco.MjModel, path: Path) -> ShadowModel:
    shadow = ShadowModel(
        model, ShadowModelConfig(window=5, checkpoint_every=0.6), log_dir=path
    )
    # Initial fatigue deliberately differs from the constructor checkpoint.
    shadow.fatigue.compute_act(np.full(len(shadow.fatigue.muscle_ids), 0.4), dt=20)
    return shadow


def _assert_estimates_equal(a, b) -> None:
    np.testing.assert_array_equal(a.time, b.time)
    np.testing.assert_array_equal(a.flags, b.flags)
    np.testing.assert_array_equal(a.columns["skipped"], b.columns["skipped"])
    np.testing.assert_array_equal(a.columns["frame_unix"], b.columns["frame_unix"])
    np.testing.assert_allclose(a.activation, b.activation, atol=1e-6)
    np.testing.assert_allclose(a.fatigue, b.fatigue, atol=1e-8)


@pytest.mark.parametrize("crash_at", [8, 32])
def test_checkpoint_resume_preserves_skips_resets_and_initial_fatigue(
    elbow: mujoco.MjModel, tmp_path: Path, crash_at: int
) -> None:
    with _new(elbow, tmp_path / "reference") as shadow:
        for k in range(60):
            _push(shadow, k)
    crashed = _new(elbow, tmp_path / "crashed")
    for k in range(crash_at):
        _push(crashed, k)
    for f in crashed.log._files.values():
        f.close()
    # A written decision with no complete input must not become a phantom pose.
    with open(tmp_path / "crashed" / OPERATIONS, "ab") as f:
        f.write(
            (
                json.dumps({"kind": "pose", "input": crash_at, "estimate": False})
                + "\n"
            ).encode()
        )
        f.write(b'{"kind": "reset"')
    with ShadowModel.resume(tmp_path / "crashed") as resumed:
        for k in range(crash_at, 60):
            _push(resumed, k)
    a, b = read_session(tmp_path / "reference"), read_session(tmp_path / "crashed")
    _assert_estimates_equal(a, b)
    assert sum(op["kind"] == "reset" for op in b.operations) == 2
    assert sum(op["kind"] == "pose" for op in b.operations) == 60
    with ShadowModel(elbow, ShadowModelConfig.from_dict(b.record["config"])) as replay:
        frames = list(b.replay(replay))
    np.testing.assert_array_equal([f.state.time for f in frames], a.time)
    np.testing.assert_allclose([f.fatigue for f in frames], a.fatigue, atol=1e-8)


def test_cli_and_sensitivity_share_the_recorded_schedule(
    elbow: mujoco.MjModel, tmp_path: Path
) -> None:
    with _new(elbow, tmp_path / "original") as shadow:
        for k in range(60):
            _push(shadow, k)
    original = read_session(tmp_path / "original")
    main(
        [
            "replay",
            "--session",
            str(tmp_path / "original"),
            "--log-dir",
            str(tmp_path / "replay"),
        ]
    )
    replay = read_session(tmp_path / "replay")
    _assert_estimates_equal(original, replay)
    np.testing.assert_array_equal(
        original.input_columns["capture_unix"], replay.input_columns["capture_unix"]
    )
    result = run_sensitivity(tmp_path / "original", [Variant("baseline")])
    np.testing.assert_array_equal(result.time, original.time)
    np.testing.assert_array_equal(result.activation[0], original.activation)
    np.testing.assert_array_equal(result.fatigue[0], original.fatigue)


def test_logs_without_a_schedule_remain_readable_and_resumable(
    elbow: mujoco.MjModel, tmp_path: Path
) -> None:
    with ShadowModel(elbow, ShadowModelConfig(window=5), log_dir=tmp_path) as shadow:
        for k in range(10):
            shadow.push(k / 20, [1.2])
    (tmp_path / OPERATIONS).unlink()
    record_path = tmp_path / "record.json"
    record = json.loads(record_path.read_text())
    record.pop("schedule_version")
    record_path.write_text(json.dumps(record))
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["offsets"].pop(OPERATIONS)
    inputs_path = tmp_path / "inputs.csv"
    inputs = list(csv.reader(io.StringIO(inputs_path.read_text())))
    assert inputs[0][-1] == "evaluation_unix"
    with inputs_path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream, lineterminator="\n").writerows(row[:-1] for row in inputs)
    checkpoint["offsets"]["inputs.csv"] = inputs_path.stat().st_size
    # Legacy 0.2 estimate rows have no per-DOF reserve/torque detail.
    estimate_path = tmp_path / ESTIMATES
    rows = list(csv.reader(io.StringIO(estimate_path.read_text())))
    width = next(i for i, name in enumerate(rows[0]) if name.startswith("reserve_"))
    with estimate_path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream, lineterminator="\n").writerows(row[:width] for row in rows)
    checkpoint["offsets"][ESTIMATES] = estimate_path.stat().st_size
    checkpoint_path.write_text(json.dumps(checkpoint))
    original = read_session(tmp_path)
    assert original.operations is None
    assert "reserve_0" not in original.columns
    assert np.isnan(original.input_columns["evaluation_unix"]).all()
    with ShadowModel(elbow, ShadowModelConfig(window=5)) as replay:
        frames = list(original.replay(replay))
    np.testing.assert_array_equal([f.state.time for f in frames], original.time)
    with ShadowModel.resume(tmp_path) as resumed:
        resumed.push(0.5, [1.2], evaluation_unix=1234.5)
    assert read_session(tmp_path).record["schedule_reconstructed"]
    with estimate_path.open(newline="", encoding="utf-8") as stream:
        assert all(len(row) == width for row in csv.reader(stream))
    with inputs_path.open(newline="", encoding="utf-8") as stream:
        assert all(len(row) == len(inputs[0]) - 1 for row in csv.reader(stream))
    updated = read_session(tmp_path)
    with ShadowModel(
        elbow, ShadowModelConfig(window=5), log_dir=tmp_path / "new"
    ) as replay:
        list(updated.replay(replay))
    assert read_session(tmp_path / "new").input_columns["evaluation_unix"][-1] == 1234.5


@pytest.mark.parametrize("fatigue", [True, False])
def test_empty_and_fatigue_disabled_sensitivity_sessions(
    elbow: mujoco.MjModel, tmp_path: Path, fatigue: bool
) -> None:
    for count in (0, 10):
        path = tmp_path / str(count)
        with ShadowModel(
            elbow, ShadowModelConfig(window=5, fatigue=fatigue), log_dir=path
        ) as shadow:
            for k in range(count):
                shadow.push(k / 20, [1.2])
        result = run_sensitivity(
            path, [Variant("baseline")], out_dir=path / "sensitivity"
        )
        assert result.activation.shape == (1, max(0, count - 4), elbow.nu)
        assert (result.fatigue is not None) == fatigue
        main(["sensitivity", "--session", str(path), "--out", str(path / "cli")])


def test_window_variants_do_not_fill_unobserved_endpoints_or_gaps(
    elbow: mujoco.MjModel, tmp_path: Path
) -> None:
    with ShadowModel(elbow, ShadowModelConfig(window=5), log_dir=tmp_path) as shadow:
        for start in (0, 5):
            for k in range(12):
                shadow.push(start + k / 20, [1.2])
    result = run_sensitivity(
        tmp_path, [Variant("baseline"), Variant("longer", window=9)]
    )
    assert np.isnan(result.activation[1, 0]).all()
    assert np.isnan(result.activation[1, -1]).all()
    assert np.isfinite(result.activation[1]).any()
    assert np.isnan(result.band("activation")[0][0]).all()


def test_sensitivity_keeps_nominal_reserve_penalty_when_strength_changes(
    elbow: mujoco.MjModel, tmp_path: Path
) -> None:
    with ShadowModel(elbow, ShadowModelConfig(window=5), log_dir=tmp_path) as shadow:
        for k in range(20):
            shadow.push(k / 20, [1.2], [-30.0])
    # Represent a legacy/dynamic config that has not stored resolved weights.
    record_path = tmp_path / "record.json"
    record = json.loads(record_path.read_text())
    record["config"]["optimization"]["reserve_optimal_force"] = None
    record_path.write_text(json.dumps(record))
    session = read_session(tmp_path)
    base = ShadowModelConfig.from_dict(record["config"])
    nominal = StaticOptimizer(elbow, base.optimization).config
    weak_model = scale_strength(elbow, 0.5)
    with ShadowModel(
        weak_model, dataclasses.replace(base, optimization=nominal)
    ) as expected:
        frames = list(session.replay(expected))
    result = run_sensitivity(tmp_path, [Variant("weak", strength=0.5)])
    np.testing.assert_array_equal(
        result.activation[0], [f.optimization.activation for f in frames]
    )


def test_interpolation_does_not_bridge_an_explicit_short_reset(
    elbow: mujoco.MjModel, tmp_path: Path
) -> None:
    with ShadowModel(
        elbow, ShadowModelConfig(window=5, max_gap=1.0), log_dir=tmp_path
    ) as shadow:
        for k in range(24):
            if k == 12:
                shadow.reset_tracking()
            shadow.push(k / 20, [1.2])
    result = run_sensitivity(
        tmp_path, [Variant("baseline"), Variant("longer", window=9)]
    )
    # The longer fit ends its first segment at .35 and starts again at .80.
    # Baseline estimates .40/.45/.70/.75 must not span that short reset.
    for t in (0.4, 0.45, 0.7, 0.75):
        index = int(np.flatnonzero(np.isclose(result.time, t))[0])
        assert np.isnan(result.activation[1, index]).all()


@pytest.mark.parametrize(
    "damage",
    ["unknown", "missing", "duplicate", "order", "not_boolean", "missing_file"],
)
def test_corrupt_schedule_is_rejected_before_resume_changes_files(
    elbow: mujoco.MjModel, tmp_path: Path, damage: str
) -> None:
    shadow = _new(elbow, tmp_path)
    for k in range(22):
        _push(shadow, k)
    for f in shadow.log._files.values():
        f.close()
    path = tmp_path / OPERATIONS
    operations = [json.loads(line) for line in path.read_text().splitlines()]
    index = next(
        i
        for i, op in enumerate(operations)
        if op["kind"] == "pose" and op["input"] == 20
    )
    if damage == "unknown":
        operations[index]["kind"] = "unrecognized"
    elif damage == "missing":
        del operations[index]
    elif damage == "duplicate":
        operations.insert(index, operations[index].copy())
    elif damage == "order":
        operations[index]["input"] = 1000
    elif damage == "not_boolean":
        operations[index]["estimate"] = 1
    if damage == "missing_file":
        path.unlink()
    else:
        path.write_text("".join(json.dumps(op) + "\n" for op in operations))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="replay|checkpoint"):
        ShadowModel.resume(tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="replay"):
        read_session(tmp_path)
