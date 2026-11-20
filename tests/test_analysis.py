"""Session report, sensitivity replays and validation statistics."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite_shadow import ShadowModel, ShadowModelConfig, read_session
from myosuite_shadow.flags import Flag
from myosuite_shadow.report import format_report, session_report
from myosuite_shadow.sensitivity import DEFAULT_VARIANTS, Variant, run_sensitivity
from myosuite_shadow.validation import (
    concordance_correlation,
    emg_envelope,
    emg_median_frequency,
    repeated_measures_correlation,
    residual_capacity,
    sample_at,
)


def _study_session(model: mujoco.MjModel, log_dir: Path, **kwargs) -> None:
    """Two 30 s holds at 20 N m with a one-minute untracked rest between."""
    with ShadowModel(
        model, ShadowModelConfig(window=5), log_dir=log_dir, **kwargs
    ) as s:
        s.mark("set_start", set=1)
        for k in range(600):
            s.push(
                k / 20, np.array([1.2]), [-20.0], flags=Flag.GLITCH if k == 100 else 0
            )
        s.mark("set_start", set=2)
        for k in range(600):
            s.push(90 + k / 20, np.array([1.2]), [-20.0 if k < 500 else -80.0])


def test_report_summarises_quality(elbow: mujoco.MjModel, tmp_path: Path) -> None:
    _study_session(elbow, tmp_path)
    r = session_report(tmp_path)
    assert r["poses"] == 1200 and r["estimates"] == 1200 - 8
    assert r["gaps"] == {"count": 1, "total_s": pytest.approx(60.05)}
    assert r["input_flags"]["GLITCH"] == 1 and r["estimate_flags"]["GLITCH"] == 5
    assert r["estimate_flags"]["AFTER_GAP"] == 1
    assert r["estimate_flags"]["CAPACITY_EXCEEDED"] >= 90  # the 80 N m stretch
    assert r["events"] == ["set_start", "set_start"] and r["closed_cleanly"]
    assert any("need reserves" in w for w in r["warnings"])
    assert any("uncalibrated" in w for w in r["warnings"])
    text = format_report(r)
    assert "| GLITCH | 5 | 1 |" in text and "Gaps: 1 (60.0 s untracked)" in text


def test_report_flags_an_unclean_end(elbow: mujoco.MjModel, tmp_path: Path) -> None:
    shadow = ShadowModel(elbow, ShadowModelConfig(window=5), log_dir=tmp_path)
    for k in range(40):
        shadow.push(k / 20, np.array([1.2]))
    for f in shadow.log._files.values():
        f.close()
    assert any("not closed cleanly" in w for w in session_report(tmp_path)["warnings"])


def test_sensitivity_brackets_the_recorded_estimates(
    elbow: mujoco.MjModel, tmp_path: Path
) -> None:
    _study_session(elbow, tmp_path / "s")
    variants = (*DEFAULT_VARIANTS, Variant("window 9", window=9))
    res = run_sensitivity(tmp_path / "s", variants, out_dir=tmp_path / "out")
    recorded = read_session(tmp_path / "s")
    # The baseline replay is the recorded session.
    np.testing.assert_array_equal(res.activation[0], recorded.activation)
    np.testing.assert_array_equal(res.fatigue[0], recorded.fatigue)
    peak = {v.name: res.fatigue[i].max() for i, v in enumerate(variants)}
    assert peak["strength x0.8"] > peak["baseline"] > peak["strength x1.25"]
    assert peak["fatigue rate x2"] > peak["baseline"] > peak["fatigue rate x0.5"]
    # R recovers during activity too (r multiplies it at rest): faster recovery
    # lowers fatigue throughout and recovers a larger share during the rest.
    assert (
        res.fatigue[6][-1].max() < res.fatigue[0][-1].max() < res.fatigue[5][-1].max()
    )
    k = int(np.searchsorted(recorded.time, 30.0))  # first estimate after the rest
    j = int(res.fatigue[0][k - 1].argmax())

    def recovered(i: int) -> float:
        return 1 - res.fatigue[i][k, j] / res.fatigue[i][k - 1, j]

    assert recovered(6) > recovered(0) > recovered(5) > 0.3
    lo, hi = res.band()
    covered = np.isfinite(lo) & np.isfinite(hi)
    assert covered.any() and not covered.all()
    assert (lo[covered] <= res.fatigue[0][covered]).all()
    assert (res.fatigue[0][covered] <= hi[covered]).all()
    assert res.fatigue.shape == (len(variants), len(recorded.time), elbow.nu)
    assert (tmp_path / "out" / "summary.csv").exists()
    assert (tmp_path / "out" / "bands.csv").read_text().count("\n") == len(
        recorded.time
    ) + 1


def test_emg_envelope_and_median_frequency() -> None:
    fs = 2000.0
    t = np.arange(0, 4, 1 / fs)
    modulation = 0.5 + 0.4 * np.sin(2 * np.pi * 0.5 * t)
    rng = np.random.default_rng(0)
    emg = modulation * rng.standard_normal(len(t))
    env = emg_envelope(emg, fs)
    keep = slice(int(fs * 0.5), -int(fs * 0.5))
    assert np.corrcoef(env[keep], modulation[keep])[0, 1] > 0.95
    tone = np.sin(2 * np.pi * 120 * t) + 0.1 * rng.standard_normal(len(t))
    centres, mdf = emg_median_frequency(tone, fs)
    assert np.all(np.abs(mdf - 120) < 10) and centres[0] == pytest.approx(0.25)


def test_sample_at_respects_gaps() -> None:
    t = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
    v = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
    out = sample_at(t, v, np.array([-1.0, 0.5, 3.0, 5.5]), max_gap=1.5)
    np.testing.assert_allclose(out[[1, 3]], [0.5, 5.5])
    assert np.isnan(out[0]) and np.isnan(out[2])


def test_agreement_statistics() -> None:
    x = np.linspace(0, 1, 50)
    assert concordance_correlation(x, x) == pytest.approx(1.0)
    assert concordance_correlation(x, x + 0.5) < 0.6  # correlated but not agreeing
    # Within each participant y rises with x; between them the offsets hide it.
    rng = np.random.default_rng(1)
    subject = np.repeat(np.arange(8), 6)
    xs = rng.uniform(0, 1, len(subject))
    ys = xs - 3 * subject + 0.05 * rng.standard_normal(len(subject))
    xs = xs + subject
    res = repeated_measures_correlation(xs, ys, subject)
    assert res["r"] > 0.95 and np.corrcoef(xs, ys)[0, 1] < 0
    assert (
        res["dof"] == len(subject) - 8 - 1 and res["ci_low"] < res["r"] < res["ci_high"]
    )


def test_residual_capacity_of_a_group(elbow: mujoco.MjModel, tmp_path: Path) -> None:
    _study_session(elbow, tmp_path)
    s = read_session(tmp_path)
    flexors = ["BIClong", "BICshort", "BRA"]
    rc = residual_capacity(s, flexors, np.array([10.0, 29.0, 89.0]))
    assert 0 < rc[1] < rc[0] < 1 and np.isnan(rc[2])
    assert np.isnan(residual_capacity(s, flexors, np.array([1e4]))[0])
