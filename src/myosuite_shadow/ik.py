"""Tracker keypoints to model poses, with per-frame quality flags (needs ``mink``).

A tracker (headset body joints, motion-capture markers, video keypoints)
reports 3-D positions of named points. :class:`KeypointIK` solves for the
model pose that places matching model frames (bodies, sites or geoms) on them,
using mink's differential IK with the joint ranges as limits and the model's
joint couplings as hard equality constraints. Each solve is warm-started from,
and regularised towards, the previous pose::

    ik = KeypointIK(model, [Keypoint("elbow", "ulna"), Keypoint("wrist", "lunate")])
    for t, points, confidence in tracker:
        result = ik.solve(t, points, confidence)
        if result.usable:
            shadow.push(t, result.qpos, flags=result.flags)

Tracker points rarely sit exactly on a model frame; place sites on the
subject model where the tracker's points are (e.g. joint centres), or expect a
constant residual. Every joint must be observable: elbow, wrist and one hand
point leave forearm pronation free (40 deg RMS error on myoArm); points on the
radial and ulnar side of the hand fix it. Frames with too few usable
keypoints return ``usable=False``; skip them, and the shadow model treats the
pause as a tracking gap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mujoco
import numpy as np
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

from myosuite_shadow.flags import Flag

_FIRST_FRAME_POSTURE = 1e-4  # resolves redundancy only; no continuity prior yet

try:
    import mink
except ImportError as err:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "myosuite_shadow.ik needs mink: pip install 'myosuite-shadow[ik]'"
    ) from err


@dataclass(frozen=True)
class Keypoint:
    """A tracked point and the model frame it corresponds to.

    Attributes:
        name: Name in the tracker's data.
        frame: Model body, site or geom name.
        frame_type: ``"body"``, ``"site"`` or ``"geom"``.
    """

    name: str
    frame: str
    frame_type: str = "body"


@dataclass(frozen=True)
class IKConfig:
    """Solver settings and quality thresholds.

    Attributes:
        position_cost: Weight of a fully confident keypoint [1/m].
        smoothing: Penalty on joint speed between frames [s/rad]: the pull
            towards the previous pose has weight ``smoothing / dt``, so it
            means the same at any frame rate. It smooths tracker noise and
            resolves poorly observed joints, but lags fast motion. The first
            frame has no previous pose and only a weak pull to ``qpos0``. On
            myoArm reaches with six keypoints, 1e-4 lags noise-free motion by at
            most 0.3 deg at 30 and 90 Hz; with 3 mm keypoint noise the joint RMS
            errors are 0.6-2.4 deg (6-7 deg in pronation) and 1 % of 90 Hz
            frames are glitches. 3e-4 removes those but lags by up to 2.4 deg.
        damping: Levenberg-Marquardt damping of the IK step.
        solver: QP solver for mink (``qpsolvers`` name).
        max_iterations: IK steps per frame.
        tolerance: Stop when every used keypoint is closer than this [m].
        step_tolerance: Or when an IK step moves no coordinate more than this
            (noisy or inconsistent keypoints never reach ``tolerance``).
        min_confidence: Keypoints below this confidence are ignored and flagged.
        min_keypoints: Fewer usable keypoints make the frame unusable.
        max_residual: Larger residuals flag ``IK_RESIDUAL`` [m].
        max_joint_speed: Faster hinge/slide motion between frames flags
            ``GLITCH`` [rad/s or m/s].
        limit_margin: Joints this close to a range limit flag ``JOINT_LIMIT``.
        couplings: Enforce the model's equality constraints (joint couplings).
    """

    position_cost: float = 1.0
    smoothing: float = 1e-4
    damping: float = 1e-4
    solver: str = "daqp"
    max_iterations: int = 30
    tolerance: float = 1e-4
    step_tolerance: float = 1e-5
    min_confidence: float = 0.5
    min_keypoints: int = 2
    max_residual: float = 0.03
    max_joint_speed: float = 30.0
    limit_margin: float = 1e-3
    couplings: bool = True


@dataclass(frozen=True)
class IKResult:
    """Pose of one frame.

    Attributes:
        qpos: ``(nq,)`` solved pose (the previous pose when not usable).
        residual: Distance of each used keypoint from its model frame [m].
        flags: Quality flags of this frame.
        usable: Enough keypoints were available to solve.
        iterations: IK steps taken.
    """

    qpos: np.ndarray
    residual: dict[str, float]
    flags: Flag
    usable: bool
    iterations: int


class KeypointIK:
    """Frame-by-frame keypoint IK on one model."""

    def __init__(
        self,
        model: mujoco.MjModel,
        keypoints: Sequence[Keypoint],
        config: IKConfig | None = None,
        qpos0: np.ndarray | None = None,
    ) -> None:
        """Set up the tasks.

        Args:
            model: The (subject) model.
            keypoints: Tracked points and their model frames.
            config: Settings (defaults if ``None``).
            qpos0: Starting pose; default the first keyframe, else ``qpos0``.
        """
        self.model, self.keypoints = model, tuple(keypoints)
        self.config = cfg = config or IKConfig()
        if qpos0 is None:
            qpos0 = model.key_qpos[0] if model.nkey else model.qpos0
        self._couplings = JointCouplings.from_model(model)
        q0 = np.array(qpos0, dtype=float)
        project_joint_couplings(self._couplings, q0)
        self.configuration = mink.Configuration(model, q0)
        self.tasks = {
            kp.name: mink.FrameTask(
                kp.frame,
                kp.frame_type,
                position_cost=cfg.position_cost,
                orientation_cost=0.0,
            )
            for kp in self.keypoints
        }
        self.posture = mink.PostureTask(model, cost=_FIRST_FRAME_POSTURE)
        self.posture.set_target(q0)
        self.limits = [mink.ConfigurationLimit(model)]
        self.constraints = (
            [mink.EqualityConstraintTask(model, cost=1.0)]
            if cfg.couplings and model.neq
            else []
        )
        scalar = np.isin(
            model.jnt_type, (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        )
        limited = scalar & (model.jnt_limited != 0)
        self._limit_adr = model.jnt_qposadr[limited]
        self._limit_range = model.jnt_range[limited]
        self._scalar_dofs = model.jnt_dofadr[scalar]
        self._prev: tuple[float, np.ndarray] | None = None

    def _frame_position(self, kp: Keypoint) -> np.ndarray:
        pose = self.configuration.get_transform_frame_to_world(kp.frame, kp.frame_type)
        return pose.translation()

    def _residuals(self, used: list[Keypoint], targets: dict[str, np.ndarray]) -> dict:
        return {
            kp.name: float(np.linalg.norm(self._frame_position(kp) - targets[kp.name]))
            for kp in used
        }

    def _step(self, tasks: list) -> float:
        """One IK step; returns its largest coordinate change."""
        cfg = self.config
        try:
            v = mink.solve_ik(
                self.configuration,
                tasks,
                1.0,
                cfg.solver,
                cfg.damping,
                limits=self.limits,
                constraints=self.constraints or None,
            )
        except mink.NoSolutionFound:
            # Couplings and limits can conflict at range edges: solve without the
            # hard couplings; the projection below restores them.
            v = mink.solve_ik(
                self.configuration,
                tasks,
                1.0,
                cfg.solver,
                cfg.damping,
                limits=self.limits,
            )
        self.configuration.integrate_inplace(v, 1.0)
        return float(np.abs(v).max(initial=0.0))

    def solve(
        self,
        time_s: float,
        positions: Mapping[str, np.ndarray],
        confidence: Mapping[str, float] | None = None,
    ) -> IKResult:
        """Pose for one tracker frame.

        Args:
            time_s: Frame time [s], for the joint-speed check.
            positions: Keypoint name -> world position ``(3,)`` [m]; absent or
                non-finite points count as missing.
            confidence: Keypoint name -> tracker confidence in [0, 1] (default 1).

        Returns:
            The pose, residuals and flags.
        """
        cfg = self.config
        flags = Flag.NONE
        used, targets = [], {}
        for kp in self.keypoints:
            p = positions.get(kp.name)
            c = 1.0 if confidence is None else float(confidence.get(kp.name, 0.0))
            task = self.tasks[kp.name]
            if p is None or not np.all(np.isfinite(p)):
                flags |= Flag.MISSING
            elif c < cfg.min_confidence:
                flags |= Flag.LOW_CONFIDENCE
            else:
                targets[kp.name] = np.asarray(p, dtype=float)
                task.set_position_cost(cfg.position_cost * c)
                task.set_target(mink.SE3.from_translation(targets[kp.name]))
                used.append(kp)
        if len(used) < cfg.min_keypoints:
            return IKResult(self.configuration.q.copy(), {}, flags, False, 0)

        if self._prev is not None and time_s > self._prev[0]:
            self.posture.set_cost(cfg.smoothing / (time_s - self._prev[0]))
        tasks = [self.tasks[kp.name] for kp in used] + [self.posture]
        iterations = 0
        for iterations in range(1, cfg.max_iterations + 1):  # noqa: B007
            step = self._step(tasks)
            if (
                step < cfg.step_tolerance
                or max(self._residuals(used, targets).values()) < cfg.tolerance
            ):
                break
        q = self.configuration.q.copy()
        project_joint_couplings(self._couplings, q)
        self.configuration.update(q)
        self.posture.set_target(q)
        residual = self._residuals(used, targets)

        if max(residual.values()) > cfg.max_residual:
            flags |= Flag.IK_RESIDUAL
        x = q[self._limit_adr]
        lo, hi = self._limit_range.T
        if np.any((x - lo < cfg.limit_margin) | (hi - x < cfg.limit_margin)):
            flags |= Flag.JOINT_LIMIT
        if self._prev is not None and time_s > self._prev[0]:
            dq = np.zeros(self.model.nv)
            mujoco.mj_differentiatePos(
                self.model, dq, time_s - self._prev[0], self._prev[1], q
            )
            if np.abs(dq[self._scalar_dofs]).max(initial=0.0) > cfg.max_joint_speed:
                flags |= Flag.GLITCH
        self._prev = (float(time_s), q.copy())
        return IKResult(q, residual, flags, True, iterations)
