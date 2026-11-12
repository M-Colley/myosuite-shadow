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

Fatigue is integrated over all elapsed time. Between estimates, each muscle
carries the load of the next estimate. Across a tracking gap longer than
``max_gap``, the last load continues until the last tracked pose, and the
untracked time is integrated as rest (``gap_load="rest"``: zero load, so
muscles recover) or as the last load held (``"hold"``, e.g. occlusion during a
sustained hold). The first estimate after a gap carries
:attr:`~myosuite_shadow.flags.Flag.AFTER_GAP` and the gap length.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import mujoco
import numpy as np
from myosuite.biomechanics.static_optimization import (
    StaticOptimizationConfig,
    StaticOptimizationResult,
    StaticOptimizer,
)
from myosuite.biomechanics.streaming_kinematics import (
    KinematicState,
    StreamingKinematics,
)
from myosuite.core.muscle_conditions import CumulativeFatigue

from myosuite_shadow.clock import now
from myosuite_shadow.flags import Flag
from myosuite_shadow.log import (
    MODEL,
    RECORD,
    SessionWriter,
    model_sha256,
    replay_operations,
)

FatigueParams = tuple[tuple[str, float, float, float], ...]


@dataclass(frozen=True)
class ShadowModelConfig:
    """Settings of the shadow model.

    Attributes:
        window: Poses per kinematic fit.
        lag: Frames of delay; ``None`` centres the fit (``window // 2``).
        polyorder: Degree of the kinematic fit.
        max_gap: A pause between poses longer than this [s] is a tracking gap:
            the kinematic fit restarts and fatigue integrates ``gap_load``.
        gap_load: ``"rest"`` (zero load, muscles recover) or ``"hold"`` (the
            last load continues) during tracking gaps.
        estimate_every: Optimize every n-th kinematic estimate (kinematics and
            logging still see every pose); fatigue covers the time between.
        optimization: Static-optimization settings (``contact_forces`` for
            standing and walking).
        fatigue: Track 3CC-r fatigue of every muscle.
        fatigue_limits_activation: Cap each muscle's activation at ``1 - MF``.
        fatigue_params: Per-muscle ``(actuator name, F, R, r)`` overriding
            MyoSuite's group parameters, e.g. from
            :func:`~myosuite_shadow.calibration.fit_fatigue_parameters`.
        reserve_limit: Reserve generalized force on a joint above which an
            estimate is flagged :attr:`~myosuite_shadow.flags.Flag.CAPACITY_EXCEEDED`
            [N m or N]. Free-joint residuals are logged but not flagged.
        log_contact_bodies: Bodies whose contact force is logged.
        checkpoint_every: Session time between checkpoints [s]; 0 disables.
    """

    window: int = 15
    lag: int | None = None
    polyorder: int = 3
    max_gap: float = 0.25
    gap_load: Literal["rest", "hold"] = "rest"
    estimate_every: int = 1
    optimization: StaticOptimizationConfig = field(
        default_factory=StaticOptimizationConfig
    )
    fatigue: bool = True
    fatigue_limits_activation: bool = True
    fatigue_params: FatigueParams = ()
    reserve_limit: float = 1.0
    log_contact_bodies: tuple[str, ...] = ()
    checkpoint_every: float = 5.0

    def __post_init__(self) -> None:
        if self.gap_load not in ("rest", "hold"):
            raise ValueError(
                f"gap_load must be 'rest' or 'hold', got {self.gap_load!r}"
            )
        if self.estimate_every < 1:
            raise ValueError("estimate_every must be at least 1")

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
        reserve = values["optimization"].reserve_optimal_force
        if isinstance(reserve, list):
            values["optimization"] = dataclasses.replace(
                values["optimization"], reserve_optimal_force=tuple(reserve)
            )
        values["log_contact_bodies"] = tuple(values.get("log_contact_bodies", ()))
        values["fatigue_params"] = tuple(
            (str(n), float(F), float(R), float(r))
            for n, F, R, r in values.get("fatigue_params", ())
        )
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
        flags: Quality flags of the estimate and of the poses in its fit.
        gap: Untracked time integrated into fatigue before this estimate [s].
        skipped: Kinematic estimates since the previous one that were not
            optimized (``estimate_every`` or a real-time backlog).
        frame_unix: Capture, else evaluation, else receive wall-clock time of
            the estimated frame. Evaluation time can describe a predicted state.
    """

    state: KinematicState
    latest_time: float
    optimization: StaticOptimizationResult
    fatigue: np.ndarray | None
    compute_ms: float
    flags: Flag = Flag.NONE
    gap: float = 0.0
    skipped: int = 0
    frame_unix: float = float("nan")

    @property
    def delay(self) -> float:
        """Time between the newest pose and the estimated frame [s]."""
        return self.latest_time - self.state.time


class Sink(Protocol):
    """Receives estimates and events as they happen (e.g. an LSL outlet)."""

    def estimate(self, frame: ShadowFrame) -> None:
        """Called after every estimate."""

    def event(self, unix: float, label: str, data: dict[str, Any]) -> None:
        """Called for every experimenter event."""

    def close(self) -> None:
        """Called when the shadow model closes."""


@dataclass
class _Pose:
    time: float
    qpos: np.ndarray
    external: np.ndarray | None
    flags: Flag
    unix: float


class ShadowModel:
    """Pose stream -> kinematics -> static optimization -> fatigue, logged per frame."""

    def __init__(
        self,
        model: mujoco.MjModel,
        config: ShadowModelConfig | None = None,
        *,
        log_dir: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
        calibration: dict[str, Any] | None = None,
        save_model: bool = True,
        sinks: tuple[Sink, ...] = (),
    ) -> None:
        """Build the pipeline and, with *log_dir*, start the session log.

        Args:
            model: Compiled (subject) model the poses are expressed in.
            config: Settings (defaults if ``None``).
            log_dir: Directory for the session log (see :mod:`myosuite_shadow.log`).
            metadata: Extra JSON-serialisable fields for the record (e.g. a
                participant or condition id).
            calibration: How the model was calibrated (see
                :mod:`myosuite_shadow.calibration`), stored in the record.
            save_model: Store the compiled model (``model.mjb``) with the log.
            sinks: Receivers of estimates and events, e.g.
                :class:`~myosuite_shadow.lsl.LslSink`.
        """
        self._setup(model, config or ShadowModelConfig(), sinks)
        self.log = None
        if log_dir is not None:
            self.log = SessionWriter(
                Path(log_dir),
                model,
                config=dataclasses.asdict(self.config),
                fatigue=self.fatigue is not None,
                contact_bodies=self.config.log_contact_bodies,
                metadata=metadata or {},
                calibration=calibration or {},
                save_model=save_model,
            )
            self.log.write_checkpoint(self._state())

    def _setup(
        self, model: mujoco.MjModel, config: ShadowModelConfig, sinks: tuple[Sink, ...]
    ) -> None:
        self.model, self.config, self.sinks = model, config, tuple(sinks)
        cfg = config
        self.kinematics = StreamingKinematics(
            model, window=cfg.window, lag=cfg.lag, polyorder=cfg.polyorder
        )
        self.optimizer = StaticOptimizer(model, cfg.optimization)
        self.config = dataclasses.replace(config, optimization=self.optimizer.config)
        self._closed = False
        self._initial_fatigue_logged = False
        muscles = model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self.fatigue = (
            CumulativeFatigue(model) if cfg.fatigue and muscles.any() else None
        )
        if self.fatigue is not None and cfg.fatigue_params:
            _apply_fatigue_params(model, self.fatigue, cfg.fatigue_params)
        free = np.zeros(model.nv, dtype=bool)
        for j in np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
            free[model.jnt_dofadr[j] : model.jnt_dofadr[j] + 6] = True
        self._free = free
        self._bodies = [model.body(b).id for b in cfg.log_contact_bodies]
        self._window: deque[_Pose] = deque(maxlen=cfg.window)
        self._last_pose: float | None = None
        self._fatigue_time: float | None = None  # fatigue is integrated up to here
        self._load: np.ndarray | None = None  # muscle activations of the last estimate
        self._pending_gap = 0.0
        self._since = cfg.estimate_every - 1  # first kinematic estimate is optimized
        self._skipped = 0
        self._restarted = False
        self._next_checkpoint = None

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------
    def reset_tracking(self, *, _record: bool = True) -> None:
        """Restart the kinematic fit (e.g. on a tracker's loss signal).

        Fatigue keeps integrating: the next estimate's load covers the time
        since the previous estimate, unless the pause exceeds ``max_gap``.
        """
        if self._closed:
            raise RuntimeError("the shadow model is closed")
        if _record and self.log is not None:
            self.log.write_operation("reset")
        self.kinematics.reset()
        self._window.clear()
        self._since = self.config.estimate_every - 1
        self._restarted = True

    def _bridge_gap(self, last_pose: float, resumed: float) -> None:
        """Integrate fatigue across a tracking gap from *last_pose* to *resumed*."""
        self._pending_gap += resumed - last_pose
        f = self.fatigue
        if f is None or self._fatigue_time is None:
            return  # nothing estimated yet: fatigue starts at the first estimate
        if last_pose > self._fatigue_time:  # still tracked: the last load continues
            f.compute_act(self._load, dt=last_pose - self._fatigue_time)
        gap = (
            np.zeros_like(self._load) if self.config.gap_load == "rest" else self._load
        )
        f.compute_act(gap, dt=resumed - last_pose)
        self._fatigue_time = resumed

    def push(
        self,
        time_s: float,
        qpos: np.ndarray,
        external_qfrc: np.ndarray | None = None,
        *,
        flags: int = Flag.NONE,
        capture_unix: float | None = None,
        evaluation_unix: float | None = None,
        receive_unix: float | None = None,
        estimate: bool = True,
    ) -> ShadowFrame | None:
        """Add the newest pose and estimate the frame ``lag`` poses before it.

        Args:
            time_s: Timestamp of the pose [s], increasing.
            qpos: ``(nq,)`` positions.
            external_qfrc: Optional ``(nv,)`` known external load at this pose
                (e.g. a hand-held weight mapped with ``mj_applyFT``).
            flags: Tracker or IK quality flags of this pose.
            capture_unix: Wall-clock time the tracker captured the pose, if known.
            evaluation_unix: Mapped time the tracked/predicted state describes;
                used for alignment when no measured capture time is available.
            receive_unix: Wall-clock time it arrived (default now).
            estimate: ``False`` updates kinematics, fatigue time and the log
                only (a real-time runner behind schedule).

        Returns:
            The estimate, or ``None`` while the kinematic window fills or when
            the optimization is skipped.
        """
        start = time.perf_counter()
        if self._closed:
            raise RuntimeError("the shadow model is closed")
        raw = np.array(qpos, dtype=np.float64)
        ext = None if external_qfrc is None else np.array(external_qfrc, dtype=float)
        if raw.shape != (self.model.nq,) or not np.isfinite(raw).all():
            raise ValueError(f"qpos must be finite with shape ({self.model.nq},)")
        if not np.isfinite(time_s):
            raise ValueError("time must be finite")
        if self._last_pose is not None and not time_s > self._last_pose:
            raise ValueError("time must increase")
        if ext is not None and (
            ext.shape != (self.model.nv,) or not np.isfinite(ext).all()
        ):
            raise ValueError(
                f"external_qfrc must be finite with shape ({self.model.nv},)"
            )
        receive = now() if receive_unix is None else float(receive_unix)
        if not np.isfinite(receive) or any(
            stamp is not None and not np.isfinite(stamp)
            for stamp in (capture_unix, evaluation_unix)
        ):
            raise ValueError(
                "capture, evaluation and receive timestamps must be finite"
            )
        if self.log is not None:
            if not self._initial_fatigue_logged:
                if self.fatigue is not None:
                    self.log.write_operation(
                        "initial_fatigue", state=self.fatigue.state_dict()
                    )
                self._initial_fatigue_logged = True
            self.log.write_input(
                time_s,
                receive,
                capture_unix,
                flags,
                raw,
                ext,
                estimate=estimate,
                evaluation_unix=evaluation_unix,
            )
        if (
            self._last_pose is not None
            and time_s - self._last_pose > self.config.max_gap
        ):
            self._bridge_gap(self._last_pose, float(time_s))
            self.reset_tracking(_record=False)
        state = self.kinematics.push(time_s, qpos)  # validates the pose
        self._last_pose = float(time_s)
        unix = next(
            float(stamp)
            for stamp in (capture_unix, evaluation_unix, receive)
            if stamp is not None
        )
        self._window.append(_Pose(float(time_s), raw, ext, Flag(flags), unix))
        frame = None
        if state is not None:
            self._since += 1
            if estimate and self._since >= self.config.estimate_every:
                self._since = 0
                frame = self._estimate(state, float(time_s), start)
            else:
                self._skipped += 1
        self._maybe_checkpoint()
        return frame

    def _estimate(
        self, state: KinematicState, latest_time: float, start: float
    ) -> ShadowFrame:
        pose = next(p for p in self._window if p.time == state.time)
        flags = Flag.NONE
        for p in self._window:
            flags |= p.flags
        if self._restarted or self._pending_gap > 0:
            flags |= Flag.AFTER_GAP
        f = self.fatigue
        cap = None
        if f is not None and self.config.fatigue_limits_activation:
            cap = np.full(self.model.nu, np.inf)
            cap[f.muscle_ids] = 1.0 - f.MF
        res = self.optimizer.solve(
            state.qpos, state.qvel, state.qacc, pose.external, max_activation=cap
        )
        reserve = float(np.abs(res.reserve[~self._free]).max(initial=0.0))
        residual = float(np.abs(res.reserve[self._free]).max(initial=0.0))
        if res.status != "solved":
            flags |= Flag.NOT_SOLVED
        if reserve > self.config.reserve_limit:
            flags |= Flag.CAPACITY_EXCEEDED
        mf = None
        if f is not None:
            load = res.activation[f.muscle_ids]
            if self._fatigue_time is not None and state.time > self._fatigue_time:
                f.compute_act(load, dt=state.time - self._fatigue_time)
            self._fatigue_time, self._load = state.time, load.copy()
            mf = np.zeros(self.model.nu)
            mf[f.muscle_ids] = f.MF
        frame = ShadowFrame(
            state=state,
            latest_time=latest_time,
            optimization=res,
            fatigue=mf,
            compute_ms=1e3 * (time.perf_counter() - start),
            flags=flags,
            gap=self._pending_gap,
            skipped=self._skipped,
            frame_unix=pose.unix,
        )
        self._pending_gap, self._skipped, self._restarted = 0.0, 0, False
        if self.log is not None:
            self.log.write_estimate(
                frame, now(), reserve, residual, res.contact_force[self._bodies]
            )
        for sink in self.sinks:
            sink.estimate(frame)
        return frame

    def mark(self, label: str, *, unix: float | None = None, **data: Any) -> None:
        """Log an experimenter event (set start, MVC probe, sync pulse, ...).

        Args:
            label: Event name.
            unix: Wall-clock time of the event (default now).
            **data: JSON-serialisable details.
        """
        unix = now() if unix is None else float(unix)
        if self.log is not None:
            self.log.write_event(unix, self._last_pose, label, data)
        for sink in self.sinks:
            sink.event(unix, label, data)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def _state(self) -> dict[str, Any]:
        f = self.fatigue
        return {
            "last_pose": self._last_pose,
            "initial_fatigue_logged": self._initial_fatigue_logged,
            "fatigue": None if f is None else f.state_dict(),
            "fatigue_time": self._fatigue_time,
            "load": None if self._load is None else self._load.tolist(),
            "pending_gap": self._pending_gap,
            "since": self._since,
            "skipped": self._skipped,
            "restarted": self._restarted,
            "window": [
                [
                    p.time,
                    p.qpos.tolist(),
                    None if p.external is None else p.external.tolist(),
                    int(p.flags),
                    p.unix,
                ]
                for p in self._window
            ],
        }

    def _maybe_checkpoint(self) -> None:
        every = self.config.checkpoint_every
        if self.log is None or every <= 0 or self._last_pose is None:
            return
        if self._next_checkpoint is None:
            self._next_checkpoint = self._last_pose + every
        elif self._last_pose >= self._next_checkpoint:
            self.log.write_checkpoint(self._state())
            self._next_checkpoint = self._last_pose + every

    def _restore(self, state: dict[str, Any]) -> None:
        f = self.fatigue
        if f is not None and state["fatigue"] is not None:
            f.load_state_dict(state["fatigue"])
        self._fatigue_time = state["fatigue_time"]
        self._load = None if state["load"] is None else np.array(state["load"])
        self._pending_gap = state["pending_gap"]
        self._since, self._skipped = state["since"], state["skipped"]
        self._restarted = state["restarted"]
        self._last_pose = state["last_pose"]
        self._initial_fatigue_logged = state.get(
            "initial_fatigue_logged", self._last_pose is not None
        )
        for t, q, ext, flags, unix in state["window"]:
            self.kinematics.push(t, np.array(q))
            external = None if ext is None else np.array(ext)
            self._window.append(_Pose(t, np.array(q), external, Flag(flags), unix))

    @classmethod
    def resume(
        cls,
        log_dir: str | Path,
        model: mujoco.MjModel | None = None,
        *,
        sinks: tuple[Sink, ...] = (),
    ) -> ShadowModel:
        """Continue a session after a crash, from its last checkpoint.

        The logs are cut back to the checkpoint and the poses received after it
        are replayed, so the session continues as if uninterrupted (to the
        optimizer's tolerance, as its warm start is not restored). The downtime
        until the next pose is a tracking gap. A ``resumed`` event and an entry
        in the record's ``resumes`` mark the restart.

        Args:
            log_dir: The session's directory.
            model: The session's model; default its ``model.mjb``.
            sinks: Receivers of estimates and events.

        Returns:
            The shadow model, ready for the next pose.

        Raises:
            ValueError: If *model* is not the session's model.
        """
        log_dir = Path(log_dir)
        record = json.loads((log_dir / RECORD).read_text(encoding="utf-8"))
        if model is None:
            model = mujoco.MjModel.from_binary_path(str(log_dir / MODEL))
        if model_sha256(model) != record["model_sha256"]:
            raise ValueError("model differs from the session's (model_sha256)")
        self = cls.__new__(cls)
        self._setup(model, ShadowModelConfig.from_dict(record["config"]), sinks)
        self.log, checkpoint, pending, operations = SessionWriter.reopen(
            log_dir,
            model.nv,
            fatigue_count=None
            if self.fatigue is None
            else len(self.fatigue.muscle_ids),
        )
        self._restore(checkpoint["state"])
        nq, nv = model.nq, model.nv
        inputs = []
        for row in pending:
            values = [float(v) if v != "" else None for v in row]
            qpos = np.array(values[4 : 4 + nq])
            ext = np.array(values[4 + nq : 4 + nq + nv])
            evaluation = values[4 + nq + nv] if len(values) > 4 + nq + nv else None
            inputs.append(
                (values[0], qpos, ext, int(values[3]), values[2], values[1], evaluation)
            )
        list(replay_operations(self, inputs, operations))
        self.mark("resumed", replayed_inputs=len(pending))
        return self

    def close(self) -> None:
        """Write a final checkpoint, close the session log and the sinks."""
        if self._closed:
            return
        self._closed = True
        if self.log is not None:
            self.log.close(self._state())
            self.log = None
        for sink in self.sinks:
            sink.close()

    def __enter__(self) -> ShadowModel:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _apply_fatigue_params(
    model: mujoco.MjModel, fatigue: CumulativeFatigue, params: FatigueParams
) -> None:
    """Override the 3CC-r F, R, r of the named muscles."""
    index = {int(a): i for i, a in enumerate(fatigue.muscle_ids)}
    F, R, r = (np.array(x, dtype=float) for x in (fatigue.F, fatigue.R, fatigue.r))
    for name, f_, r_cap, r_mult in params:
        i = index[model.actuator(name).id]
        F[i], R[i], r[i] = f_, r_cap, r_mult
    fatigue.set_FatigueCoefficient(F)
    fatigue.set_RecoveryCoefficient(R)
    fatigue.set_RecoveryMultiplier(r)
