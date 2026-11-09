"""Per-participant calibration: strength from an MVC, body mass, fatigue rates.

Without calibration, absolute activations and fatigue rates rest on population
muscle parameters and can be off by tens of percent. Typical use::

    qpos = posture_at(model, {"elbow_flexion": np.radians(90)})
    strength = strength_from_mvc(model, qpos, "elbow_flexion", measured_torque=62.0)
    model = scale_body_mass(scale_strength(model, strength.scale), 71.5)
    fit = fit_fatigue_parameters([0.3, 0.5], [205.0, 74.0], R=0.00225, r=4.93)
    config = ShadowModelConfig(fatigue_params=fatigue_params(model, flexors, fit))
    ShadowModel(model, config, calibration={"strength": strength.record(),
                                            "fatigue": fit.record()})

Muscle forces in MuJoCo scale with each muscle's peak force ``F0`` (active and
passive alike), so one strength factor scales the joint torque exactly.
Segment geometry is not scaled here; use MyoSuite's subject scaling for that.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import mujoco
import numpy as np
from myosuite.core.muscle_conditions import CumulativeFatigue
from scipy.optimize import minimize

_MUSCLE = mujoco.mjtDyn.mjDYN_MUSCLE


def _muscles(model: mujoco.MjModel, muscles: Sequence[str] | None) -> np.ndarray:
    if muscles is None:
        return np.flatnonzero(model.actuator_dyntype == _MUSCLE)
    ids = np.array([model.actuator(m).id for m in muscles], dtype=int)
    if (model.actuator_dyntype[ids] != _MUSCLE).any():
        raise ValueError("strength scaling applies to muscle actuators only")
    return ids


def posture_at(model: mujoco.MjModel, joints: Mapping[str, float]) -> np.ndarray:
    """``qpos0`` (or the first keyframe) with the named hinge/slide joints set.

    Args:
        model: The model.
        joints: Joint name -> position [rad or m].

    Returns:
        ``(nq,)`` posture.
    """
    qpos = (model.key_qpos[0] if model.nkey else model.qpos0).copy()
    for name, value in joints.items():
        qpos[model.jnt_qposadr[model.joint(name).id]] = value
    return qpos


def _actuator_torque(
    model: mujoco.MjModel, qpos: np.ndarray, activation: np.ndarray
) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.act[model.actuator_actadr[model.actuator_actadr >= 0]] = activation[
        model.actuator_actadr >= 0
    ]
    mujoco.mj_fwdPosition(model, data)
    mujoco.mj_fwdVelocity(model, data)
    mujoco.mj_fwdActuation(model, data)
    return data.qfrc_actuator.copy()


@dataclass(frozen=True)
class StrengthCalibration:
    """Strength scale from one maximal voluntary contraction.

    Attributes:
        joint: Joint the torque was measured about.
        posture: ``(nq,)`` posture of the measurement.
        measured_torque: Measured net torque [N m], signed in the joint's direction.
        model_torque: The model's net torque with every agonist fully active
            and the antagonists relaxed (including passive forces) [N m].
        scale: ``measured_torque / model_torque``.
        agonists: Muscles whose activation pushes in the measured direction.
    """

    joint: str
    posture: np.ndarray
    measured_torque: float
    model_torque: float
    scale: float
    agonists: tuple[str, ...]

    def record(self) -> dict:
        """JSON-serialisable form for the session record."""
        out = asdict(self)
        out["posture"] = self.posture.tolist()
        return out


def max_joint_torque(
    model: mujoco.MjModel, qpos: np.ndarray, joint: str, direction: float = 1.0
) -> tuple[float, tuple[str, ...]]:
    """The model's isometric net torque about *joint* with all agonists at 1.

    Muscle force is linear in activation at a fixed state, so each muscle's
    contribution is measured separately and the agonists (those pushing in
    *direction*) are summed onto the relaxed (passive) torque.

    Args:
        model: The model.
        qpos: ``(nq,)`` posture.
        joint: A hinge or slide joint.
        direction: +1 or -1, the direction of the contraction.

    Returns:
        Net torque and the agonist names.
    """
    dof = model.jnt_dofadr[model.joint(joint).id]
    ids = np.flatnonzero(model.actuator_dyntype == _MUSCLE)
    zero = np.zeros(model.nu)
    passive = _actuator_torque(model, qpos, zero)[dof]
    contribution = {}
    for i in ids:
        act = zero.copy()
        act[i] = 1.0
        contribution[i] = _actuator_torque(model, qpos, act)[dof] - passive
    agonists = [i for i in ids if np.sign(contribution[i]) == np.sign(direction)]
    total = passive + sum(contribution[i] for i in agonists)
    return float(total), tuple(model.actuator(i).name for i in agonists)


def strength_from_mvc(
    model: mujoco.MjModel, qpos: np.ndarray, joint: str, measured_torque: float
) -> StrengthCalibration:
    """Strength scale that makes the model's maximal torque equal a measured MVC.

    Args:
        model: The (uncalibrated) model.
        qpos: ``(nq,)`` posture of the measurement (e.g. elbow at 90 degrees).
        joint: Joint the dynamometer measured about.
        measured_torque: Net torque [N m], signed in the joint's direction
            (flexion positive for a flexion joint).

    Returns:
        The calibration; apply it with :func:`scale_strength`.

    Raises:
        ValueError: If the model cannot push in that direction.
    """
    direction = float(np.sign(measured_torque))
    model_torque, agonists = max_joint_torque(model, qpos, joint, direction)
    if model_torque * direction <= 0:
        raise ValueError(f"the model's muscles cannot push {joint} that way")
    return StrengthCalibration(
        joint=joint,
        posture=np.array(qpos, dtype=float),
        measured_torque=float(measured_torque),
        model_torque=model_torque,
        scale=float(measured_torque / model_torque),
        agonists=agonists,
    )


def scale_strength(
    model: mujoco.MjModel, scale: float, muscles: Sequence[str] | None = None
) -> mujoco.MjModel:
    """A copy with the peak force ``F0`` of *muscles* (default all) times *scale*."""
    out = copy.deepcopy(model)
    ids = _muscles(out, muscles)
    out.actuator_gainprm[ids, 2] *= scale
    out.actuator_biasprm[ids, 2] *= scale
    return out


def scale_body_mass(model: mujoco.MjModel, total_mass: float) -> mujoco.MjModel:
    """A copy whose bodies' masses and inertias are scaled to *total_mass* [kg].

    Segment proportions and geometry are kept; segment lengths need MyoSuite's
    subject scaling.
    """
    out = copy.deepcopy(model)
    ratio = total_mass / float(out.body_mass[1:].sum())
    out.body_mass[1:] *= ratio
    out.body_inertia[1:] *= ratio
    return out


# ----------------------------------------------------------------------
# Fatigue rates from endurance times
# ----------------------------------------------------------------------
_ONE_MUSCLE = (
    """<mujoco><worldbody>{bodies}</worldbody><actuator>{muscles}</actuator></mujoco>"""
)


def _fatigue_bank(n: int) -> CumulativeFatigue:
    """3CC-r state for *n* independent muscles (one per load)."""
    bodies = "".join(
        f'<body><joint name="j{i}" axis="0 1 0" range="0 1"/>'
        '<geom size=".1" mass="1"/></body>'
        for i in range(n)
    )
    muscles = "".join(
        f'<muscle name="m{i}" joint="j{i}" lengthrange="0 1"/>' for i in range(n)
    )
    model = mujoco.MjModel.from_xml_string(
        _ONE_MUSCLE.format(bodies=bodies, muscles=muscles)
    )
    return CumulativeFatigue(model)


def endurance_time(
    loads: Sequence[float],
    F: float,
    R: float,
    r: float,
    *,
    dt: float = 0.25,
    max_time: float = 1800.0,
) -> np.ndarray:
    """3CC-r time to task failure of sustained contractions (Frey-Law et al. 2012).

    A hold of ``load`` (fraction of MVC) fails when the residual capacity
    ``1 - MF`` falls below the load; the crossing is interpolated between steps.

    Args:
        loads: Target loads in (0, 1].
        F, R, r: 3CC-r fatigue, recovery and rest-recovery coefficients.
        dt: Step between failure checks [s] (integrated in MyoSuite's sub-steps).
        max_time: Give up after this long [s]; such holds return ``inf``.

    Returns:
        ``(len(loads),)`` endurance times [s].
    """
    loads = np.asarray(loads, dtype=float)
    bank = _fatigue_bank(len(loads))
    for setter, value in (
        (bank.set_FatigueCoefficient, F),
        (bank.set_RecoveryCoefficient, R),
        (bank.set_RecoveryMultiplier, r),
    ):
        setter(value)
    out = np.full(len(loads), np.inf)
    margin = (1.0 - bank.MF) - loads
    t = 0.0
    while t < max_time and np.isinf(out).any():
        bank.compute_act(loads, dt=dt)
        t += dt
        new = (1.0 - bank.MF) - loads
        crossed = np.isinf(out) & (new < 0)
        out[crossed] = t - dt * new[crossed] / (new[crossed] - margin[crossed])
        margin = new
    return out


@dataclass(frozen=True)
class FatigueFit:
    """3CC-r parameters fitted to endurance times.

    Attributes:
        F, R, r: Fitted (or fixed) coefficients.
        loads: Loads of the holds (fraction of MVC).
        observed: Observed endurance times [s].
        predicted: Endurance times of the fitted model [s].
        rmse_log: Root-mean-square error of the log endurance times.
        fitted: Names of the fitted coefficients.
    """

    F: float
    R: float
    r: float
    loads: tuple[float, ...]
    observed: tuple[float, ...]
    predicted: tuple[float, ...]
    rmse_log: float
    fitted: tuple[str, ...]

    def record(self) -> dict:
        """JSON-serialisable form for the session record."""
        return asdict(self)


def fit_fatigue_parameters(
    loads: Sequence[float],
    endurance_times: Sequence[float],
    *,
    F: float = 0.01,
    R: float = 0.002,
    r: float = 15.0,
    fit_recovery: bool = False,
) -> FatigueFit:
    """Fit 3CC-r's F (and optionally R) to a participant's sustained holds.

    Minimises the squared error of log endurance times (Nelder-Mead in log
    parameters). ``r`` acts only at rest, so sustained holds cannot inform it.

    Args:
        loads: Loads of the holds (fraction of the calibrated MVC).
        endurance_times: Observed times to task failure [s].
        F, R, r: Starting (or fixed) coefficients, e.g. MyoSuite's group row.
        fit_recovery: Also fit R (needs holds at three or more loads).

    Returns:
        The fit.
    """
    loads = np.asarray(loads, dtype=float)
    observed = np.asarray(endurance_times, dtype=float)
    max_time = 3.0 * float(observed.max())

    def unpack(x: np.ndarray) -> tuple[float, float]:
        return float(np.exp(x[0])), float(np.exp(x[1])) if fit_recovery else R

    def loss(x: np.ndarray) -> float:
        f_, r_cap = unpack(x)
        pred = endurance_time(loads, f_, r_cap, r, max_time=max_time)
        pred = np.where(np.isfinite(pred), pred, max_time)
        return float(np.mean((np.log(pred) - np.log(observed)) ** 2))

    x0 = np.log([F, R]) if fit_recovery else np.log([F])
    res = minimize(
        loss, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-10}
    )
    f_, r_cap = unpack(res.x)
    pred = endurance_time(loads, f_, r_cap, r, max_time=max_time)
    return FatigueFit(
        F=f_,
        R=r_cap,
        r=float(r),
        loads=tuple(loads.tolist()),
        observed=tuple(observed.tolist()),
        predicted=tuple(pred.tolist()),
        rmse_log=float(np.sqrt(res.fun)),
        fitted=("F", "R") if fit_recovery else ("F",),
    )


def fatigue_params(
    model: mujoco.MjModel, muscles: Sequence[str], fit: FatigueFit
) -> tuple[tuple[str, float, float, float], ...]:
    """``ShadowModelConfig.fatigue_params`` applying *fit* to *muscles*."""
    _muscles(model, muscles)
    return tuple((m, fit.F, fit.R, fit.r) for m in muscles)
