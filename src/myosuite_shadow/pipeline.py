"""Shadow model: live muscle estimates beside an experiment.

Chains MyoSuite's live joint kinematics
(:class:`~myosuite.physics.streaming_kinematics.StreamingKinematics`), static
optimization (:class:`~myosuite.physics.static_optimization.StaticOptimizer`) and
3CC-r fatigue (:class:`~myosuite.core.muscle_conditions.CumulativeFatigue`)::

    meta = {"participant": "P07"}
    with ShadowModel(model, log_dir="logs/P07", metadata=meta) as shadow:
        for t, qpos in tracker:  # IK of each tracked pose onto the subject model
            frame = shadow.push(t, qpos)
            if frame is not None:
                show(frame.fatigue)

Fatigue takes the optimised activations as target loads. With
``fatigue_limits_activation`` a fatigued muscle delivers at most ``1 - MF`` of
its force, so later frames shift load to fresher muscles and, once those
saturate, to the reserves. Estimates are delayed by the kinematics lag.
"""

from __future__ import annotations

import dataclasses
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from myosuite.core.muscle_conditions import CumulativeFatigue
from myosuite.physics.static_optimization import (
    StaticOptimizationConfig,
    StaticOptimizationResult,
    StaticOptimizer,
)
from myosuite.physics.streaming_kinematics import KinematicState, StreamingKinematics

from myosuite_shadow.log import SessionWriter


@dataclass(frozen=True)
class ShadowModelConfig:
    """Settings of the shadow model.

    Attributes:
        window: Poses per kinematic fit.
        lag: Frames of delay; ``None`` centres the fit (``window // 2``).
        polyorder: Degree of the kinematic fit.
        max_gap: A pause between poses longer than this [s] restarts the
            kinematic fit; fatigue is held across it.
        optimization: Static-optimization settings (``contact_forces`` for
            standing and walking).
        fatigue: Track 3CC-r fatigue of every muscle.
        fatigue_limits_activation: Cap each muscle's activation at ``1 - MF``.
        log_contact_bodies: Bodies whose contact force is logged.
    """

    window: int = 15
    lag: int | None = None
    polyorder: int = 3
    max_gap: float = 0.25
    optimization: StaticOptimizationConfig = field(
        default_factory=StaticOptimizationConfig
    )
    fatigue: bool = True
    fatigue_limits_activation: bool = True
    log_contact_bodies: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ShadowModelConfig:
        """Rebuild a config from ``dataclasses.asdict`` output (e.g. a log record).

        Args:
            values: Field values; ``optimization`` as a nested dict.

        Returns:
            The config.
        """
        values = dict(values)
        values["optimization"] = StaticOptimizationConfig(
            **values.get("optimization", {})
        )
        values["log_contact_bodies"] = tuple(values.get("log_contact_bodies", ()))
        return cls(**values)


@dataclass(frozen=True)
class ShadowFrame:
    """Estimates for one past frame.

    Attributes:
        state: Kinematics of the frame (``state.time`` is its timestamp).
        latest_time: Timestamp of the newest pose when it was computed [s].
        optimization: Activations, forces, reserves and contact forces.
        fatigue: ``(nu,)`` fatigued fraction ``MF`` per actuator (zero for
            non-muscles), or ``None`` without fatigue.
        compute_ms: Wall-clock time of this update [ms].
    """

    state: KinematicState
    latest_time: float
    optimization: StaticOptimizationResult
    fatigue: np.ndarray | None
    compute_ms: float

    @property
    def delay(self) -> float:
        """Time between the newest pose and the estimated frame [s]."""
        return self.latest_time - self.state.time


class ShadowModel:
    """Pose stream -> kinematics -> static optimization -> fatigue, logged per frame."""

    def __init__(
        self,
        model: mujoco.MjModel,
        config: ShadowModelConfig | None = None,
        *,
        log_dir: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
        save_model: bool = True,
    ) -> None:
        """Build the pipeline and, with *log_dir*, start the session log.

        Args:
            model: Compiled (subject) model the poses are expressed in.
            config: Settings (defaults if ``None``).
            log_dir: Directory for the session log (see :mod:`myosuite_shadow.log`).
            metadata: Extra JSON-serialisable fields for the record (e.g. a
                participant or condition id).
            save_model: Store the compiled model (``model.mjb``) with the log.
        """
        self.model, self.config = model, config or ShadowModelConfig()
        cfg = self.config
        self.kinematics = StreamingKinematics(
            model, window=cfg.window, lag=cfg.lag, polyorder=cfg.polyorder
        )
        self.optimizer = StaticOptimizer(model, cfg.optimization)
        muscles = model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self.fatigue = (
            CumulativeFatigue(model) if cfg.fatigue and muscles.any() else None
        )
        # Known external load of each pose in the window, matched to the estimate.
        self._external: deque[tuple[float, np.ndarray | None]] = deque(
            maxlen=cfg.window
        )
        self._last_pose: float | None = None
        self._last_estimate: float | None = None
        self._bodies = [model.body(b).id for b in cfg.log_contact_bodies]
        self.log = (
            None
            if log_dir is None
            else SessionWriter(
                Path(log_dir),
                model,
                config=dataclasses.asdict(cfg),
                fatigue=self.fatigue is not None,
                contact_bodies=cfg.log_contact_bodies,
                metadata=metadata or {},
                save_model=save_model,
            )
        )

    def reset_tracking(self) -> None:
        """Restart the kinematic fit after a tracking loss; fatigue is held."""
        self.kinematics.reset()
        self._external.clear()
        self._last_estimate = None

    def push(
        self, time_s: float, qpos: np.ndarray, external_qfrc: np.ndarray | None = None
    ) -> ShadowFrame | None:
        """Add the newest pose and estimate the frame ``lag`` poses before it.

        Args:
            time_s: Timestamp of the pose [s], increasing.
            qpos: ``(nq,)`` positions.
            external_qfrc: Optional ``(nv,)`` known external load at this pose
                (e.g. a hand-held weight mapped with ``mj_applyFT``).

        Returns:
            The estimate, or ``None`` while the kinematic window fills.
        """
        start = time.perf_counter()
        if (
            self._last_pose is not None
            and time_s - self._last_pose > self.config.max_gap
        ):
            self.reset_tracking()
        state = self.kinematics.push(time_s, qpos)  # validates the pose
        self._last_pose = time_s
        self._external.append((time_s, external_qfrc))
        if self.log is not None:
            self.log.write_input(time_s, qpos, external_qfrc)
        if state is None:
            return None
        external = next(e for t, e in self._external if t == state.time)

        f = self.fatigue
        cap = None
        if f is not None and self.config.fatigue_limits_activation:
            cap = np.full(self.model.nu, np.inf)
            cap[f.muscle_ids] = 1.0 - f.MF
        res = self.optimizer.solve(
            state.qpos, state.qvel, state.qacc, external, max_activation=cap
        )
        mf = None
        if f is not None:
            if self._last_estimate is not None:
                f.compute_act(
                    res.activation[f.muscle_ids], dt=state.time - self._last_estimate
                )
            mf = np.zeros(self.model.nu)
            mf[f.muscle_ids] = f.MF
        self._last_estimate = state.time

        frame = ShadowFrame(
            state=state,
            latest_time=float(time_s),
            optimization=res,
            fatigue=mf,
            compute_ms=1e3 * (time.perf_counter() - start),
        )
        if self.log is not None:
            self.log.write_estimate(frame, res.contact_force[self._bodies])
        return frame

    def close(self) -> None:
        """Close the session log."""
        if self.log is not None:
            self.log.close()

    def __enter__(self) -> ShadowModel:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
