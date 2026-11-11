"""Probe independence and tracking discontinuities survive validation analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite_shadow import ShadowModel, ShadowModelConfig
from myosuite_shadow.log import read_session
from myosuite_shadow.study import validate_probes
from myosuite_shadow.validation import emg_median_frequency, mvc_capacity, sample_at


def test_sample_at_keeps_exact_samples_and_respects_explicit_resets() -> None:
    out = sample_at(
        np.array([0.0, 1.0, 4.0, 5.0]),
        np.array([2.0, 3.0, 6.0, 7.0]),
        np.array([1.0, 3.0, 4.0, 4.5]),
        max_gap=1.5,
    )
    np.testing.assert_allclose(out, [3, np.nan, 6, 6.5], equal_nan=True)
    out = sample_at(
        np.array([0.0, 0.1, 0.2]),
        np.array([0.0, 1.0, 2.0]),
        np.array([0.05, 0.1, 0.15]),
        break_before=np.array([False, True, False]),
    )
    np.testing.assert_allclose(out, [np.nan, 1, 1.5], equal_nan=True)
    assert np.isnan(sample_at(np.array([]), np.array([]), np.array([1.0]))[0])


def test_flat_or_missing_emg_cannot_be_reported_as_fatigue_frequency() -> None:
    _, frequencies = emg_median_frequency(np.zeros(1000), 1000)
    assert np.isnan(frequencies).all()
    with pytest.raises(ValueError, match="resolve"):
        emg_median_frequency(np.zeros(1000), 1000, step_s=0.0001)


def test_independent_probes_exclude_calibration_and_tracking_gaps(
    elbow: mujoco.MjModel,
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "session"
    with ShadowModel(elbow, ShadowModelConfig(window=5), log_dir=log_dir) as shadow:
        for k in range(10):
            shadow.push(k / 20, np.array([1.2]), [-15.0])
        for k in range(10):
            shadow.push(2 + k / 20, np.array([1.2]), [-15.0])
    probes = tmp_path / "probes.csv"
    with probes.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", "joint", "measured_torque", "role", "qpos_0"])
        writer.writerows(
            [
                [0.15, "r_elbow_flex", 40, "calibration", 1.2],
                [0.2, "r_elbow_flex", 35, "validation", 1.2],
                [1.0, "r_elbow_flex", 35, "validation", 1.2],
                [2.2, "r_elbow_flex", 30, "validation", 1.2],
            ]
        )
    report = validate_probes(log_dir, probes)
    assert report["included"] == report["excluded"] == 2
    assert report["probes"][0]["exclusion"] == "used_for_calibration"
    assert report["probes"][2]["exclusion"] == "tracking_gap_or_quality"
    assert report["metrics"][0]["n"] == 2
    json.dumps(report, allow_nan=False)
    other = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint/><geom size=".1"/></body></worldbody></mujoco>'
    )
    with pytest.raises(ValueError, match="calibrated model"):
        mvc_capacity(
            read_session(log_dir), other, "joint", np.array([0.2]), np.zeros((1, 1))
        )
