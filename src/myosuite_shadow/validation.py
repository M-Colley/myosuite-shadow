"""Comparing estimates with measurements: EMG, MVC probes, endurance times.

Building blocks for validating the shadow model against people (the prototype's
``docs/validation_study.md``, hypotheses H1-H3):

* :func:`load_xdf` reads a LabRecorder file (EMG, dynamometer, these estimates
  and markers on one LSL time base; needs ``pyxdf``);
* :func:`emg_envelope` and :func:`emg_median_frequency` process surface EMG
  (SENIAM-style band-pass, rectification, low-pass; Welch spectra);
* :func:`sample_at` resamples any stream onto estimate or probe times;
* :func:`residual_capacity` gives the model's ``1 - MF`` of a muscle group;
* :func:`concordance_correlation` (Lin) and :func:`repeated_measures_correlation`
  (Bakdash and Marusich, via ``pingouin``) are the agreement statistics.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from myosuite_shadow.flags import Flag
from myosuite_shadow.log import Session


def load_xdf(path: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Streams of an XDF file by name: ``(timestamps, samples)`` on the LSL clock.

    Args:
        path: LabRecorder ``.xdf`` file.

    Returns:
        Stream name -> ``(t, x)`` with ``x`` of shape ``(n, channels)`` (string
        streams such as markers keep their values as an object array).
    """
    import pyxdf  # noqa: PLC0415

    streams, _ = pyxdf.load_xdf(str(path))
    out = {}
    for s in streams:
        name = s["info"]["name"][0]
        if name in out:
            raise ValueError(
                f"XDF contains duplicate stream name {name!r}; "
                "select streams by identity before analysis"
            )
        out[name] = (
            np.asarray(s["time_stamps"], dtype=float),
            np.asarray(s["time_series"]),
        )
    return out


def emg_envelope(
    emg: np.ndarray,
    fs: float,
    *,
    band: tuple[float, float] = (20.0, 450.0),
    lowpass: float = 6.0,
    order: int = 4,
) -> np.ndarray:
    """Linear envelope: zero-phase band-pass, full-wave rectification, low-pass.

    Args:
        emg: ``(n,)`` or ``(n, channels)`` raw EMG.
        fs: Sampling rate [Hz].
        band: Band-pass corners [Hz]; the upper one is capped below Nyquist.
        lowpass: Envelope cut-off [Hz].
        order: Butterworth order of each filter.

    Returns:
        The envelope, same shape as *emg*.
    """
    hi = min(band[1], 0.45 * fs)
    bp = signal.butter(order, (band[0], hi), btype="bandpass", fs=fs, output="sos")
    lp = signal.butter(order, lowpass, btype="lowpass", fs=fs, output="sos")
    x = np.abs(signal.sosfiltfilt(bp, np.asarray(emg, dtype=float), axis=0))
    return signal.sosfiltfilt(lp, x, axis=0)


def emg_median_frequency(
    emg: np.ndarray,
    fs: float,
    *,
    window_s: float = 0.5,
    step_s: float = 0.25,
    band: tuple[float, float] = (20.0, 450.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Median frequency of the EMG spectrum over sliding windows (falls with fatigue).

    Args:
        emg: ``(n,)`` raw EMG of one channel.
        fs: Sampling rate [Hz].
        window_s: Window length [s].
        step_s: Step between windows [s].
        band: Frequencies considered [Hz].

    Returns:
        Window centre times [s from the first sample] and median frequencies [Hz].
    """
    x = np.asarray(emg, dtype=float)
    if x.ndim != 1 or not np.isfinite(fs) or fs <= 0:
        raise ValueError("one EMG channel and a positive finite sample rate required")
    if not np.isfinite([window_s, step_s, *band]).all() or min(window_s, step_s) <= 0:
        raise ValueError("window/step and frequency band must be finite and positive")
    n, step = int(window_s * fs), int(step_s * fs)
    if n < 2 or step < 1 or not 0 <= band[0] < min(band[1], fs / 2):
        raise ValueError(
            "window, step and frequency band must resolve at this sample rate"
        )
    centres, mdf = [], []
    for start in range(0, len(x) - n + 1, step):
        centres.append((start + n / 2) / fs)
        segment = x[start : start + n]
        if not np.isfinite(segment).all():
            mdf.append(float("nan"))
            continue
        f, p = signal.welch(segment, fs=fs, nperseg=min(n, 256))
        keep = (f >= band[0]) & (f <= min(band[1], fs / 2))
        cum = np.cumsum(p[keep])
        if not len(cum) or not np.isfinite(cum[-1]) or cum[-1] <= 0:
            mdf.append(float("nan"))
            continue
        mdf.append(float(f[keep][np.searchsorted(cum, cum[-1] / 2)]))
    return np.array(centres), np.array(mdf)


def sample_at(
    t_src: np.ndarray,
    values: np.ndarray,
    t_dst: np.ndarray,
    *,
    max_gap: float | None = None,
    break_before: np.ndarray | None = None,
) -> np.ndarray:
    """Linear interpolation of a stream onto other times (NaN outside it or in gaps).

    Args:
        t_src: ``(n,)`` increasing source times.
        values: ``(n,)`` or ``(n, k)`` source values.
        t_dst: ``(m,)`` target times.
        max_gap: Source gaps longer than this [s] give NaN instead of a bridge.
        break_before: Boolean per source sample; do not interpolate across a
            reset immediately before that sample. Exact samples are retained.

    Returns:
        ``(m,)`` or ``(m, k)`` values.
    """
    t_src, t_dst = np.asarray(t_src, dtype=float), np.asarray(t_dst, dtype=float)
    v = np.asarray(values, dtype=float)
    if t_src.ndim != 1 or t_dst.ndim != 1 or v.ndim not in (1, 2):
        raise ValueError("times must be vectors and values a vector or matrix")
    if len(t_src) != len(v) or not np.isfinite(t_src).all():
        raise ValueError("source times must be finite and match values")
    if np.any(np.diff(t_src) <= 0):
        raise ValueError("source times must increase strictly")
    if max_gap is not None and (not np.isfinite(max_gap) or max_gap <= 0):
        raise ValueError("max_gap must be finite and positive")
    breaks = (
        np.zeros(len(t_src), dtype=bool)
        if break_before is None
        else np.asarray(break_before, dtype=bool)
    )
    if breaks.shape != t_src.shape:
        raise ValueError("break_before must match source times")
    flat = v.ndim == 1
    v = v[:, None] if flat else v
    if not len(t_src):
        shape = (len(t_dst),) if flat else (len(t_dst), v.shape[1])
        return np.full(shape, np.nan)
    out = np.column_stack(
        [
            np.interp(t_dst, t_src, v[:, j], left=np.nan, right=np.nan)
            for j in range(v.shape[1])
        ]
    )
    if len(t_src) > 1:
        idx = np.clip(np.searchsorted(t_src, t_dst), 1, len(t_src) - 1)
        inside = (t_dst > t_src[idx - 1]) & (t_dst < t_src[idx])
        invalid = breaks[idx]
        if max_gap is not None:
            invalid |= (t_src[idx] - t_src[idx - 1]) > max_gap
        out[inside & invalid] = np.nan
    out[~np.isfinite(t_dst)] = np.nan
    return out[:, 0] if flat else out


def residual_capacity(
    session: Session, muscles: Sequence[str], times: np.ndarray
) -> np.ndarray:
    """Descriptive mean ``1 - MF``; use ``mvc_capacity`` for predicted joint MVC.

    Args:
        session: A session read with :func:`~myosuite_shadow.read_session`.
        muscles: Actuator names of the group (e.g. the elbow flexors).
        times: Session times [s].

    Returns:
        ``(len(times),)`` residual capacity in [0, 1].
    """
    if session.fatigue is None:
        raise ValueError("the session has no fatigue estimates")
    idx = [session.actuators.index(m) for m in muscles]
    rc = 1.0 - session.fatigue[:, idx].mean(axis=1)
    return sample_at(
        session.time,
        rc,
        times,
        max_gap=session.record["config"].get("max_gap", 0.25),
        break_before=(session.flags & int(Flag.AFTER_GAP)) != 0,
    )


def mvc_capacity(
    session: Session,
    model: Any,
    joint: str,
    times: np.ndarray,
    postures: np.ndarray,
    *,
    direction: float = 1.0,
    reject_flags: int = int(
        Flag.INFERRED
        | Flag.LOW_CONFIDENCE
        | Flag.MISSING
        | Flag.IK_RESIDUAL
        | Flag.GLITCH
        | Flag.NOT_SOLVED
    ),
) -> np.ndarray:
    """Predict signed maximal joint torque at independent MVC probe postures.

    Uses muscle strength, moment arms, passive torque and each muscle's fatigue
    cap. Missing, flagged or untracked intervals return NaN.

    Args:
        session: Recorded estimates and model provenance.
        model: The session's calibrated model.
        joint: Measured scalar joint.
        times: Probe times on the session clock [s].
        postures: Actual measured probe postures, shape ``(len(times), nq)``.
        direction: Signed contraction direction.
        reject_flags: Quality bits excluded from probe prediction.

    Returns:
        Signed predicted net joint torques [N m or N], NaN if unsupported.
    """
    from myosuite.biomechanics.calibration import max_joint_torque

    from myosuite_shadow.log import model_sha256

    times, postures = np.asarray(times, float), np.asarray(postures, float)
    if postures.shape != (len(times), model.nq) or not np.isfinite(postures).all():
        raise ValueError("probe postures must be finite with shape (n, model.nq)")
    if model_sha256(model) != session.record["model_sha256"]:
        raise ValueError("MVC predictions require the session's calibrated model")
    if session.fatigue is None:
        raise ValueError("the session has no fatigue estimates")
    mf = session.fatigue.copy()
    mf[(session.flags & reject_flags) != 0] = np.nan
    sampled = sample_at(
        session.time,
        mf,
        times,
        max_gap=session.record["config"].get("max_gap", 0.25),
        break_before=(session.flags & int(Flag.AFTER_GAP)) != 0,
    )
    torque = np.full(len(times), np.nan)
    for i, (qpos, fatigue) in enumerate(zip(postures, sampled)):
        if np.isfinite(fatigue).all():
            torque[i] = max_joint_torque(
                model, qpos, joint, direction, max_activation=1.0 - fatigue
            )[0]
    return torque


def concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient (agreement, not just correlation)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return float("nan")
    cov = np.mean((x - x.mean()) * (y - y.mean()))
    denominator = x.var() + y.var() + (x.mean() - y.mean()) ** 2
    return float(2 * cov / denominator) if denominator > 0 else float("nan")


def repeated_measures_correlation(
    x: np.ndarray, y: np.ndarray, subject: Sequence[Any]
) -> dict[str, float]:
    """Within-participant correlation of paired measurements (needs ``pingouin``).

    Args:
        x, y: Paired values (e.g. model residual capacity and normalised MVC).
        subject: Participant of each pair.

    Returns:
        ``r``, ``dof``, ``pval`` and the 95 % confidence interval
        (``ci_low``, ``ci_high``).
    """
    import pandas as pd  # noqa: PLC0415
    import pingouin  # noqa: PLC0415

    df = pd.DataFrame({"x": x, "y": y, "subject": subject}).dropna()
    res = pingouin.rm_corr(data=df, x="x", y="y", subject="subject")
    ci = res["CI95" if "CI95" in res else "CI95%"].iloc[0]  # renamed in pingouin 0.6
    return {
        "r": float(res["r"].iloc[0]),
        "dof": float(res["dof"].iloc[0]),
        "pval": float(res["pval"].iloc[0]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
    }
