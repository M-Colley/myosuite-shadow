"""Uncertainty of a session's estimates under plausible calibration errors.

Replays a recorded session's poses (bit-exact inputs) through the shadow model
with perturbed strength, fatigue and recovery rates or kinematic window, and
reports how far activations and fatigue move. Report estimates with these
bands; a conclusion that flips within them is not supported by the model::

    myosuite-shadow sensitivity --session logs/P07 --out logs/P07/sensitivity
"""

from __future__ import annotations

import csv
import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from myosuite.core.muscle_conditions import CumulativeFatigue

from myosuite_shadow.calibration import scale_strength
from myosuite_shadow.log import read_session
from myosuite_shadow.pipeline import (
    ShadowModel,
    ShadowModelConfig,
    _apply_fatigue_params,
)


@dataclass(frozen=True)
class Variant:
    """One perturbation of the recorded calibration.

    Attributes:
        name: Label in the outputs.
        strength: Factor on every muscle's peak force.
        fatigue_rate: Factor on 3CC-r's F.
        recovery_rate: Factor on 3CC-r's R.
        window: Kinematic window (``None``: the session's).
    """

    name: str
    strength: float = 1.0
    fatigue_rate: float = 1.0
    recovery_rate: float = 1.0
    window: int | None = None


DEFAULT_VARIANTS = (
    Variant("baseline"),
    Variant("strength x0.8", strength=0.8),
    Variant("strength x1.25", strength=1.25),
    Variant("fatigue rate x0.5", fatigue_rate=0.5),
    Variant("fatigue rate x2", fatigue_rate=2.0),
    Variant("recovery rate x0.5", recovery_rate=0.5),
    Variant("recovery rate x2", recovery_rate=2.0),
)
"""Strength within the spread of MVC-based calibration; F and R within the
spread of individual 3CC-r fits (factors of about two)."""


@dataclass(frozen=True)
class SensitivityResult:
    """Estimates of every variant on the baseline's estimate times.

    Attributes:
        variants: The variants, baseline first.
        time: ``(n,)`` estimate times of the baseline [s].
        activation: ``(V, n, nu)`` activations.
        fatigue: ``(V, n, nu)`` ``MF``.
        actuators: Actuator names.
    """

    variants: tuple[Variant, ...]
    time: np.ndarray
    activation: np.ndarray
    fatigue: np.ndarray
    actuators: tuple[str, ...]

    def band(self, quantity: str = "fatigue") -> tuple[np.ndarray, np.ndarray]:
        """Lowest and highest value over the variants, ``(n, nu)`` each."""
        values = getattr(self, quantity)
        return values.min(axis=0), values.max(axis=0)


def _variant_params(model, config: ShadowModelConfig, v: Variant) -> tuple:
    """Per-muscle F, R, r of the session, scaled for the variant."""
    f = CumulativeFatigue(model)
    if config.fatigue_params:
        _apply_fatigue_params(model, f, config.fatigue_params)
    names = [model.actuator(int(i)).name for i in f.muscle_ids]
    return tuple(
        (n, float(F) * v.fatigue_rate, float(R) * v.recovery_rate, float(r))
        for n, F, R, r in zip(names, f.F, f.R, f.r)
    )


def run_sensitivity(
    log_dir: str | Path,
    variants: Sequence[Variant] = DEFAULT_VARIANTS,
    out_dir: str | Path | None = None,
) -> SensitivityResult:
    """Replay a session under each variant.

    Args:
        log_dir: The recorded session (needs ``model.mjb``).
        variants: Perturbations; the first should be the baseline.
        out_dir: Optionally write ``summary.csv`` (per variant and muscle: peak
            and final ``MF``, mean activation) and ``bands.csv`` (per estimate
            time: lowest and highest ``MF`` and activation per muscle).

    Returns:
        The estimates of every variant.
    """
    session = read_session(log_dir)
    model0 = session.load_model(log_dir)
    base = ShadowModelConfig.from_dict(session.record["config"])
    base = dataclasses.replace(base, checkpoint_every=0.0)
    times, act, mf = None, [], []
    for v in variants:
        model = scale_strength(model0, v.strength) if v.strength != 1.0 else model0
        config = dataclasses.replace(
            base,
            window=v.window or base.window,
            lag=None if v.window else base.lag,
            fatigue_params=_variant_params(model, base, v),
        )
        shadow = ShadowModel(model, config)
        frames = []
        for k, (t, q) in enumerate(zip(session.input_time, session.input_qpos)):
            frame = shadow.push(t, q, session.input_qfrc[k])
            if frame is not None:
                frames.append(frame)
        t_v = np.array([f.state.time for f in frames])
        a_v = np.array([f.optimization.activation for f in frames])
        m_v = np.array([f.fatigue for f in frames])
        if times is None:
            times = t_v
        elif len(t_v) != len(times) or np.any(t_v != times):
            # Another window estimates other frames: compare on the baseline's times.
            a_v = np.array(
                [np.interp(times, t_v, a_v[:, j]) for j in range(model.nu)]
            ).T
            m_v = np.array(
                [np.interp(times, t_v, m_v[:, j]) for j in range(model.nu)]
            ).T
        act.append(a_v)
        mf.append(m_v)
    result = SensitivityResult(
        variants=tuple(variants),
        time=times,
        activation=np.array(act),
        fatigue=np.array(mf),
        actuators=tuple(session.actuators),
    )
    if out_dir is not None:
        _write(result, Path(out_dir))
    return result


def _write(result: SensitivityResult, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "actuator", "peak_mf", "final_mf", "mean_activation"])
        for v, a, m in zip(result.variants, result.activation, result.fatigue):
            for j, name in enumerate(result.actuators):
                w.writerow([v.name, name, m[:, j].max(), m[-1, j], a[:, j].mean()])
    lo_m, hi_m = result.band("fatigue")
    lo_a, hi_a = result.band("activation")
    with open(out / "bands.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        names = result.actuators
        w.writerow(
            ["time"]
            + [f"{p}_{n}" for p in ("mf_lo", "mf_hi", "a_lo", "a_hi") for n in names]
        )
        for k, t in enumerate(result.time):
            w.writerow([t, *lo_m[k], *hi_m[k], *lo_a[k], *hi_a[k]])
