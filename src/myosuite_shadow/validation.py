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
    n, step = int(window_s * fs), int(step_s * fs)
    centres, mdf = [], []
    for start in range(0, len(x) - n + 1, step):
        f, p = signal.welch(x[start : start + n], fs=fs, nperseg=min(n, 256))
        keep = (f >= band[0]) & (f <= min(band[1], fs / 2))
        cum = np.cumsum(p[keep])
        mdf.append(float(f[keep][np.searchsorted(cum, cum[-1] / 2)]))
        centres.append((start + n / 2) / fs)
    return np.array(centres), np.array(mdf)


def sample_at(
    t_src: np.ndarray,
    values: np.ndarray,
    t_dst: np.ndarray,
    *,
    max_gap: float | None = None,
) -> np.ndarray:
    """Linear interpolation of a stream onto other times (NaN outside it or in gaps).

    Args:
        t_src: ``(n,)`` increasing source times.
        values: ``(n,)`` or ``(n, k)`` source values.
        t_dst: ``(m,)`` target times.
        max_gap: Source gaps longer than this [s] give NaN instead of a bridge.

    Returns:
        ``(m,)`` or ``(m, k)`` values.
    """
    t_src, t_dst = np.asarray(t_src, dtype=float), np.asarray(t_dst, dtype=float)
    v = np.asarray(values, dtype=float)
    flat = v.ndim == 1
    v = v[:, None] if flat else v
    out = np.column_stack(
        [
            np.interp(t_dst, t_src, v[:, j], left=np.nan, right=np.nan)
            for j in range(v.shape[1])
        ]
    )
    if max_gap is not None and len(t_src) > 1:
        idx = np.clip(np.searchsorted(t_src, t_dst), 1, len(t_src) - 1)
        out[(t_src[idx] - t_src[idx - 1]) > max_gap] = np.nan
    return out[:, 0] if flat else out


def residual_capacity(
    session: Session, muscles: Sequence[str], times: np.ndarray
) -> np.ndarray:
    """Mean ``1 - MF`` of *muscles* at *times* (session time), e.g. at MVC probes.

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
    return sample_at(session.time, rc, times)


def concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient (agreement, not just correlation)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    cov = np.mean((x - x.mean()) * (y - y.mean()))
    return float(2 * cov / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))


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
