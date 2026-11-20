"""Connect registered tracker frames to a logged live shadow session."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from myosuite_shadow.clock import now
from myosuite_shadow.flags import Flag
from myosuite_shadow.ik import KeypointIK
from myosuite_shadow.live import LiveShadowModel
from myosuite_shadow.log import model_sha256
from myosuite_shadow.tracking.frames import Registration, TrackerFrame
from myosuite_shadow.tracking.io import RawFrameWriter, repair_jsonl_tail


@dataclass(frozen=True)
class AcquisitionConfig:
    """Protocol choices applied before any pose affects muscle or fatigue state.

    Attributes:
        allow_inferred: Permit inferred required landmarks, retaining their flag.
        max_prediction_s: Largest accepted prediction horizon [s].
        max_age_s: Largest capture/receive age before a frame is rejected [s].
        max_clock_uncertainty_s: Largest declared clock mapping uncertainty [s].
        held_mass_kg: Known stationary held mass; gravity only, no object inertia.
        load_site: Model site carrying that mass, required when mass is nonzero.
    """

    allow_inferred: bool = False
    max_prediction_s: float = 0.0
    max_age_s: float = 0.25
    max_clock_uncertainty_s: float = 0.01
    check_wallclock: bool = True
    require_mapped_time: bool = True
    required_fidelity: str | None = None
    held_mass_kg: float = 0.0
    load_site: str | None = None

    def __post_init__(self) -> None:
        values = (
            self.max_prediction_s,
            self.max_age_s,
            self.max_clock_uncertainty_s,
            self.held_mass_kg,
        )
        if not np.isfinite(values).all() or min(values) < 0 or self.max_age_s == 0:
            raise ValueError("acquisition thresholds must be finite and nonnegative")
        if self.held_mass_kg and not self.load_site:
            raise ValueError("a held mass requires its model load_site")
        if self.required_fidelity not in (None, "low", "high"):
            raise ValueError("required_fidelity must be low, high or None")


class TrackedSession:
    """Record raw evidence and submit only accepted registered poses.

    The caller owns ``live`` and closes it after closing this acquisition layer.
    A reference-space or skeleton revision change requires renewed registration
    using ``set_registration`` before processing further frames.

    ``clock`` optionally supplies a finite Unix policy time for reproducible
    receipt-time reprocessing. The default calls the current ``now()`` each time.
    ``external_loads(time_s, qpos)`` supplies known finite generalized forces for
    accepted poses; it is mutually exclusive with a configured held mass.
    """

    def __init__(
        self,
        live: LiveShadowModel,
        ik: KeypointIK,
        *,
        registration: Registration | None = None,
        config: AcquisitionConfig | None = None,
        raw_dir: str | Path | None = None,
        clock: Callable[[], float] | None = None,
        external_loads: Callable[[float, np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.live, self.ik = live, ik
        if clock is not None and not callable(clock):
            raise TypeError("acquisition clock must be callable")
        self._clock = clock
        self.config = config or AcquisitionConfig()
        if external_loads is not None and not callable(external_loads):
            raise TypeError("external loads must be callable")
        if external_loads is not None and self.config.held_mass_kg:
            raise ValueError("external loads cannot also include configured held mass")
        self._external_loads = external_loads
        if registration is not None and registration.target_frame != "model_world":
            raise ValueError("acquisition registration must target model_world")
        if ik.model is not live.shadow.model and (
            model_sha256(ik.model) != model_sha256(live.shadow.model)
        ):
            raise ValueError("IK and live pipeline must share the calibrated model")
        if self.config.load_site is not None:
            ik.model.site(self.config.load_site)
        self.registration = registration
        self.accepted, self.rejected = 0, 0
        self._identity: tuple[str, str] | None = None
        self._revision: tuple[int, int] | None = None
        self._last_sequence: int | None = None
        self._last_time: float | None = None
        self._last_accepted_unix: float | None = None
        self._last_rejection: str | None = None
        self._registration_needed = False
        self._clock_record: str | None = None
        self._raw: RawFrameWriter | None = None
        self._files: list[Any] = []
        if raw_dir is None and live.shadow.log is not None:
            raw_dir = live.shadow.log.log_dir
        if raw_dir is not None:
            root = Path(raw_dir)
            root.mkdir(parents=True, exist_ok=True)
            self._raw = RawFrameWriter(root / "tracking.jsonl", append=True)
            quality_path = root / "tracking_quality.jsonl"
            try:
                quality_recovery = repair_jsonl_tail(quality_path)
                self._files = [open(quality_path, "a", encoding="utf-8")]
                recoveries = [
                    {
                        "file": str(path),
                        "backup": str(backup),
                        "bytes": backup.stat().st_size,
                    }
                    for path, backup in (
                        (self._raw.path, self._raw.recovery_path),
                        (quality_path, quality_recovery),
                    )
                    if backup is not None
                ]
                if recoveries:
                    self.live.mark("raw_recovered", files=recoveries)
            except BaseException:
                self._raw.close()
                for stream in self._files:
                    stream.close()
                self._files = []
                raise
        setup = {
            "policy": dataclasses.asdict(self.config),
            "ik": dataclasses.asdict(ik.config),
            "keypoints": [dataclasses.asdict(k) for k in ik.keypoints],
            "registration": None if registration is None else registration.record(),
            "policy_clock": "injected" if clock is not None else "wall_clock",
            "external_loads": "provider"
            if external_loads is not None
            else "configuration",
        }
        self.live.mark("tracking_setup", **setup)

    def _now(self) -> float:
        value = float(now() if self._clock is None else self._clock())
        if not np.isfinite(value):
            raise ValueError("acquisition clock must be finite")
        return value

    def _write(self, index: int, record: dict[str, Any]) -> None:
        if self._files:
            stream = self._files[index]
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()

    def set_registration(self, registration: Registration) -> None:
        """Apply an explicitly renewed spatial calibration after recentering."""
        if registration.target_frame != "model_world":
            raise ValueError("acquisition registration must target model_world")
        self.registration = registration
        self._registration_needed = False
        self._revision = None
        self.ik.reset()
        self.live.reset_tracking()
        self.live.mark("registration", **registration.record())

    def submit(self, frame: TrackerFrame) -> bool:
        """Log a decoded frame, reject invalid evidence, then solve and queue IK."""
        if not isinstance(frame, TrackerFrame):
            raise TypeError("acquisition requires metric 3D TrackerFrame observations")
        if self._raw is not None:
            self._raw.write(frame)
        observed_time = self._now()
        reason = None
        identity = (frame.source_id, frame.person_id)
        if self._identity is not None and identity != self._identity:
            reason = "source_or_person_changed"
        elif (
            self._last_sequence is not None
            and frame.sequence <= self._last_sequence
            or self._last_time is not None
            and frame.time_s <= self._last_time
        ):
            reason = "out_of_order"
        ordering_admissible = reason is None
        time_admissible = True
        if frame.prediction_s > self.config.max_prediction_s:
            reason = reason or "prediction_horizon"
        if self.config.required_fidelity is not None and (
            frame.metadata.get("actual_fidelity") != self.config.required_fidelity
        ):
            reason = reason or "tracking_fidelity"
        stamp = frame.capture_unix
        if stamp is None:
            stamp = frame.evaluation_unix
        if (
            self.config.check_wallclock
            and self.config.require_mapped_time
            and stamp is None
        ):
            time_admissible = False
            reason = reason or "unmapped_pose_clock"
        if stamp is None:
            stamp = frame.receive_unix
        if (
            self.config.check_wallclock
            and stamp is not None
            and (
                observed_time - stamp > self.config.max_age_s
                or stamp > observed_time + self.config.max_clock_uncertainty_s
            )
        ):
            time_admissible = False
            reason = reason or "stale_or_future_timestamp"
        clock = frame.metadata.get("clock_mapping", frame.metadata.get("clock", {}))
        if not isinstance(clock, dict):
            time_admissible = False
            reason = reason or "invalid_clock_mapping"
            clock = {}
        uncertainty = clock.get("uncertainty_s", 0)
        if isinstance(uncertainty, bool) or not isinstance(uncertainty, int | float):
            uncertainty = None
        else:
            try:
                uncertainty = float(uncertainty)
            except OverflowError:
                uncertainty = None
        if uncertainty is None or not np.isfinite(uncertainty) or uncertainty < 0:
            time_admissible = False
            reason = reason or "invalid_clock_mapping"
        elif uncertainty > self.config.max_clock_uncertainty_s:
            time_admissible = False
            reason = reason or "clock_uncertainty"
        # Invalid clock evidence cannot advance ordering or spatial revisions.
        if ordering_admissible and time_admissible:
            if self._identity is None:
                self._identity = identity
            self._last_sequence, self._last_time = frame.sequence, frame.time_s
            revision = (frame.transform_revision, frame.skeleton_revision)
            if self._revision is not None and revision != self._revision:
                self._registration_needed = True
            self._revision = revision
        if self._registration_needed:
            reason = reason or "registration_required"
        clock_record = json.dumps(clock, sort_keys=True)
        if reason is None and clock:
            if self._clock_record is not None and self._clock_record != clock_record:
                self.ik.reset()
                self.live.reset_tracking()
                self.live.mark("clock_mapping", mapping=clock)
            self._clock_record = clock_record
        normalized = frame
        if reason is None:
            if self.registration is not None:
                try:
                    normalized = self.registration.apply(frame)
                except ValueError:
                    reason = "registration_required"
            elif frame.coordinate_frame != "model_world":
                reason = "registration_required"
        if reason == "registration_required":
            self._registration_needed = True
        names = {k.name for k in self.ik.keypoints}
        required = set(self.ik.config.required_keypoints) or names
        missing = {
            n
            for n in required
            if not normalized.position_valid.get(n, False)
            or normalized.confidence.get(n, 0) < self.ik.config.min_confidence
        }
        if missing:
            reason = reason or "required_landmarks"
        inferred = names.intersection(normalized.inferred_joints)
        if inferred and not self.config.allow_inferred:
            reason = reason or "inferred_landmarks"
        result = None
        if reason is None:
            positions = {
                n: p
                for n, p in normalized.positions.items()
                if normalized.position_valid[n]
            }
            result = self.ik.solve(normalized.time_s, positions, normalized.confidence)
            if not result.usable:
                reason = "ik_quality"
        external = None
        if reason is None:
            if self._external_loads is not None:
                external = np.asarray(
                    self._external_loads(frame.time_s, result.qpos.copy()), dtype=float
                ).copy()
                if (
                    external.shape != (self.ik.model.nv,)
                    or not np.isfinite(external).all()
                ):
                    raise ValueError("external loads require a finite model.nv vector")
            if self.config.held_mass_kg:
                from myosuite.biomechanics.external_loads import held_mass_qfrc

                external = held_mass_qfrc(
                    self.ik.model,
                    result.qpos,
                    self.config.held_mass_kg,
                    site=self.config.load_site,
                )
        decision = {
            "source_id": frame.source_id,
            "person_id": frame.person_id,
            "sequence": frame.sequence,
            "time_s": frame.time_s,
            "accepted": reason is None,
            "reason": reason,
            "flags": int(result.flags) if result is not None else 0,
            "residual_m": result.residual if result is not None else {},
            "inferred_landmarks": sorted(inferred),
        }
        self._write(0, decision)
        if reason is not None:
            self._last_rejection = reason
            self.rejected += 1
            if self.rejected == 1:
                self.live.mark("tracking_rejected", reason=reason)
            return False
        flags = result.flags | (
            Flag.INFERRED if inferred or frame.prediction_s > 0 else Flag.NONE
        )
        self.live.submit(
            frame.time_s,
            result.qpos,
            external,
            flags=flags,
            capture_unix=frame.capture_unix,
            receive_unix=frame.receive_unix,
            evaluation_unix=frame.evaluation_unix,
        )
        self.accepted += 1
        self._last_rejection = None
        self._last_accepted_unix = observed_time
        return True

    @property
    def health(self) -> dict[str, Any]:
        """Worker health plus the age of the last accepted acquisition."""
        result = dict(self.live.health)
        age = (
            None
            if self._last_accepted_unix is None
            else self._now() - self._last_accepted_unix
        )
        result.update(
            accepted=self.accepted,
            rejected=self.rejected,
            tracking_age_s=age,
            last_rejection=self._last_rejection,
            registration_required=self._registration_needed,
        )
        if result.get("status") not in {"failed", "closed", "closing"} and (
            age is None or age > self.config.max_age_s
        ):
            result["status"] = "tracking_stale"
        return result

    def close(self) -> None:
        """Close raw/quality records; the live runner remains caller-owned."""
        for stream in self._files:
            stream.close()
        self._files = []
        if self._raw is not None:
            self._raw.close()
            self._raw = None

    def __enter__(self) -> TrackedSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
