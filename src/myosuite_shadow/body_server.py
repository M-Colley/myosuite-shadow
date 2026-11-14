"""Live body transport around the existing acquisition and shadow pipeline."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite_shadow.acquisition import TrackedSession
from myosuite_shadow.body_protocol import SCHEMA, BodyClock, BodySource, message
from myosuite_shadow.flags import Flag, describe
from myosuite_shadow.live import LiveShadowModel
from myosuite_shadow.log import model_sha256
from myosuite_shadow.pipeline import ShadowModel
from myosuite_shadow.preflight import preflight
from myosuite_shadow.tracking_config import TrackingSetup


class BodyServer:
    """One configured participant, one producer, bounded input and logged output.

    Heavy IK runs on a separate thread. Clock/control messages stay on the event
    loop while newer poses replace an unprocessed pose. Restart the server for a
    new participant or a restarted source clock/sequence epoch.
    """

    def __init__(
        self,
        setup: TrackingSetup,
        *,
        log_dir: str | Path,
        source: BodySource | None = None,
        regions: tuple | None = None,
        profile: str = "full_body",
        strict_pose_timing: bool = False,
        output_hz: float = 10.0,
    ) -> None:
        if not np.isfinite(output_hz) or not 1 <= output_hz <= 60:
            raise ValueError("output_hz must be between 1 and 60")
        if (
            not setup.acquisition.check_wallclock
            or not setup.acquisition.require_mapped_time
        ):
            raise ValueError(
                "live body acquisition requires mapped wall-clock freshness"
            )
        self.setup, self.profile = setup, profile
        self.source = source or BodySource(**setup.settings["body_source"])
        free_root = np.any(setup.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if free_root and (
            setup.shadow_config.root_residual_force_limit is None
            or setup.shadow_config.root_residual_moment_limit is None
        ):
            raise ValueError(
                "live full-body models require root force/moment residual limits"
            )
        self.excluded_actuators = ()
        if regions is None:
            from myosuite_shadow.body_profiles import profile_for_model

            description = profile_for_model(setup.model, profile)
            regions = description.regions
            self.excluded_actuators = description.excluded_actuators
        self.regions = tuple(regions)
        self._region_ids = []
        for region in self.regions:
            ids = [setup.model.actuator(name).id for name in region.actuators]
            if not ids:
                raise ValueError("a body region requires muscle actuators")
            for actuator in ids:
                if setup.model.actuator_dyntype[actuator] != mujoco.mjtDyn.mjDYN_MUSCLE:
                    raise ValueError("body regions may contain only muscle actuators")
            self._region_ids.append(ids)
        if len({r.id for r in self.regions}) != len(self.regions):
            raise ValueError("body region IDs must be unique")
        self.strict_pose_timing = bool(strict_pose_timing)
        self.output_hz = float(output_hz)
        self.session_id = uuid.uuid4().hex
        self.model_hash = model_sha256(setup.model)
        self.log_dir = Path(log_dir).resolve()
        self._server = None
        self._active = False
        self._closed = False
        self._live = None
        self._acquisition = None
        self._wire = None
        self._history = deque(maxlen=2048)
        self._origin = time.perf_counter()
        self._unix_origin = time.time()
        self._reason = "waiting_for_producer"
        self._dropped = 0
        self._source_metadata = {}
        self._history_flags = Flag.NONE
        self._assumed_load_s = 0.0
        self.preflight = preflight(setup.config_path, live=False)

    def _on_estimate(self, frame) -> None:
        self._history_flags |= frame.flags
        self._assumed_load_s += frame.gap

    def _record(self, direction: str, packet: Any) -> None:
        if self._wire is not None:
            self._wire.write(
                json.dumps(
                    {
                        "unix": time.time(),
                        "direction": direction,
                        "packet": packet,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            self._wire.flush()

    async def start(self, host: str = "127.0.0.1", port: int = 9766) -> BodyServer:
        from websockets.asyncio.server import serve

        if self._server is not None or self._closed:
            raise RuntimeError("body server cannot be started twice")
        self.log_dir.mkdir(parents=True, exist_ok=False)
        try:
            self._wire = (self.log_dir / "body_wire.jsonl").open("x", encoding="utf-8")
            (self.log_dir / "body_preflight.json").write_text(
                json.dumps(self.preflight, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            shadow = ShadowModel(
                self.setup.model,
                self.setup.shadow_config,
                log_dir=self.log_dir,
                metadata={
                    "body_bridge_schema": SCHEMA,
                    "session_id": self.session_id,
                    "body_source": vars(self.source),
                    "profile": self.profile,
                    "tracking_config": self.setup.settings,
                    "strict_pose_timing": self.strict_pose_timing,
                    "region_aggregation": "maximum muscle fraction",
                    "regions": [vars(region) for region in self.regions],
                    "excluded_actuators": self.excluded_actuators,
                },
            )
            self._live = LiveShadowModel(
                shadow,
                max_queue=8,
                stale_after=self.setup.acquisition.max_age_s,
                on_frame=self._on_estimate,
            )
            self._acquisition = TrackedSession(
                self._live,
                self.setup.ik,
                registration=self.setup.registration,
                config=self.setup.acquisition,
                raw_dir=self.log_dir,
            )
            self._server = await serve(
                self._handle,
                host,
                port,
                max_size=2**20,
                max_queue=8,
                ping_interval=10,
                ping_timeout=10,
            )
        except BaseException:
            await self.close()
            raise
        return self

    @property
    def uri(self) -> str:
        if self._server is None:
            raise RuntimeError("body server is not listening")
        host, port = self._server.sockets[0].getsockname()[:2]
        return f"ws://{'[' + host + ']' if ':' in host else host}:{port}"

    def state(self) -> dict:
        """Freshness and region masks are evaluated for every outgoing state."""
        live = self._live
        frame = None if live is None else live.latest
        health = {} if self._acquisition is None else self._acquisition.health
        status = "waiting" if frame is None else "running"
        age = None if frame is None else max(0.0, time.time() - frame.frame_unix)
        flags = [] if frame is None else describe(frame.flags)
        reason = self._reason
        if health.get("status") == "failed":
            status, reason = "failed", health.get("worker_error")
        elif reason in ("waiting_for_producer", "clock_warmup"):
            status = "waiting"
        elif reason not in (None, "waiting_for_producer", "clock_warmup"):
            status = "rejected"
        elif frame is not None:
            fatal = Flag(frame.flags) & ~Flag.INFERRED
            if fatal:
                status, reason = "uncertain", "model_quality_flags"
            if (
                age > self.setup.acquisition.max_age_s
                or health.get("status") == "tracking_stale"
            ):
                status, reason = "stale", "tracking_stale"
        if not self._active:
            status, reason = (
                ("waiting" if frame is None else "stale"),
                "producer_disconnected",
            )
        if self.strict_pose_timing and status == "running":
            status, reason = "uncertain", "native_pose_capture_time_unverified"
        if self.preflight["status"] == "blocked" and status == "running":
            status, reason = "uncertain", "configuration_preflight_blocked"
        evidence = None
        if frame is not None:
            evidence = next(
                (
                    f
                    for f in reversed(self._history)
                    if abs(f.time_s - frame.state.time) < 1e-8
                ),
                None,
            )
        regions = []
        for region, ids in zip(self.regions, self._region_ids, strict=True):
            region_status = "unavailable"
            activation = fatigue = None
            if evidence is not None and all(
                evidence.position_valid.get(n, False)
                and evidence.confidence.get(n, 0.0)
                >= self.setup.ik.config.min_confidence
                for n in region.required_landmarks
            ):
                region_status = (
                    "tracked"
                    if all(
                        evidence.position_tracked.get(n, False)
                        for n in region.required_landmarks
                    )
                    else "inferred"
                )
            if status == "running" and region_status == "tracked":
                a = np.asarray(frame.optimization.activation)[ids]
                f = None if frame.fatigue is None else np.asarray(frame.fatigue)[ids]
                tolerance = max(
                    1e-8, 10 * self.setup.shadow_config.optimization.tolerance
                )
                if (
                    np.isfinite(a).all()
                    and ((-tolerance <= a) & (a <= 1 + tolerance)).all()
                ):
                    activation = float(np.clip(a.max(), 0.0, 1.0))
                if (
                    f is not None
                    and np.isfinite(f).all()
                    and ((0 <= f) & (f <= 1)).all()
                ):
                    fatigue = float(f.max())
                if activation is None or (
                    frame.fatigue is not None and fatigue is None
                ):
                    status, reason = "failed", "invalid_model_output"
            regions.append(
                {
                    "id": region.id,
                    "label": region.label,
                    "anchor": region.anchor,
                    "status": region_status,
                    "activation": activation,
                    "fatigue": fatigue,
                }
            )
        if status != "running":
            for region in regions:
                region["activation"] = region["fatigue"] = None
        return {
            "type": "shadow_state",
            "schema": SCHEMA,
            "session_id": self.session_id,
            "model_sha256": self.model_hash,
            "profile": self.profile,
            "time_s": None if frame is None else float(frame.state.time),
            "quality": {
                "status": status,
                "flags": flags,
                "age_s": age,
                "max_age_s": self.setup.acquisition.max_age_s,
                "reason": reason,
                "age_basis": "client_observation_time",
                "capture_age_verified": False,
                "preflight_status": self.preflight["status"],
                "history_flags": describe(self._history_flags),
                "assumed_load_s": self._assumed_load_s,
                "excluded_muscle_count": len(self.excluded_actuators),
                "dropped_input_frames": self._dropped,
                "accepted": health.get("accepted", 0),
                "rejected": health.get("rejected", 0),
                "limitations": [
                    "Native pose capture age is unverified.",
                    "Reference estimates need participant and task validation.",
                    "Contact and external loads are assumptions, not measurements.",
                    "Fatigued fraction MF is not measured strength loss.",
                    "Distal muscles are excluded; nuisance poses are regularized.",
                ],
            },
            "regions": regions,
        }

    async def _handle(self, websocket) -> None:
        from websockets.exceptions import ConnectionClosed

        if self._active:
            await websocket.close(
                code=1013, reason="one configured body producer is already active"
            )
            return
        self._active = True
        stopped = asyncio.Event()
        available = asyncio.Event()
        pending = [None]
        generation = [0]
        clock_queue = asyncio.Queue(maxsize=8)
        send_lock = asyncio.Lock()
        tasks = []
        clock = BodyClock(
            self.source.clock_id,
            self._unix_origin,
            max_uncertainty_s=self.setup.acquisition.max_clock_uncertainty_s,
        )

        async def send(packet):
            async with send_lock:
                await websocket.send(json.dumps(packet, allow_nan=False))
                self._record("out", packet)

        async def process():
            while not stopped.is_set():
                await available.wait()
                available.clear()
                item, pending[0] = pending[0], None
                if item is None:
                    continue
                frame, version = item
                try:
                    accepted = await asyncio.to_thread(self._acquisition.submit, frame)
                    if accepted:
                        self._history.append(frame)
                        if version == generation[0]:
                            self._reason = None
                    elif version == generation[0]:
                        self._reason = self._acquisition.health["last_rejection"]
                except Exception as error:
                    self._reason = f"acquisition_failed: {error}"
                    self._record("error", {"reason": self._reason})
                    stopped.set()
                    await websocket.close(code=1011, reason="body acquisition failed")

        async def publish():
            while not stopped.is_set():
                await send(self.state())
                await asyncio.sleep(1 / self.output_hz)

        async def fit_clocks():
            while not stopped.is_set():
                reply = await clock_queue.get()
                if reply is None:
                    return
                packet, received = reply
                try:
                    # LP fitting must not suspend packet receipt or feedback.
                    await asyncio.to_thread(clock.reply, packet, received)
                except (ValueError, TypeError, KeyError) as error:
                    generation[0] += 1
                    self._reason = f"invalid_clock: {error}"
                    self._record("error", {"reason": self._reason})

        async def synchronize():
            while not stopped.is_set():
                await send(clock.probe(time.perf_counter() - self._origin))
                await asyncio.sleep(0.25)

        try:
            hello = message(await asyncio.wait_for(websocket.recv(), timeout=5))
            self._source_metadata = self.source.hello(hello)
            self._record("in", hello)
            self._reason = "clock_warmup"
            tasks = [
                asyncio.create_task(process()),
                asyncio.create_task(fit_clocks()),
                asyncio.create_task(publish()),
                asyncio.create_task(synchronize()),
            ]
            async for raw in websocket:
                try:
                    packet = message(raw)
                    self._record("in", packet)
                    if packet.get("type") == "clock_reply":
                        await clock_queue.put(
                            (packet, time.perf_counter() - self._origin)
                        )
                    elif packet.get("type") == "body_frame":
                        if clock.mapping is None:
                            self._reason = "clock_warmup"
                            continue
                        frame = self.source.frame(
                            packet,
                            clock.mapping,
                            receive_unix=time.time(),
                            source_metadata=self._source_metadata,
                        )
                        if pending[0] is not None:
                            self._dropped += 1
                        pending[0] = (frame, generation[0])
                        available.set()
                    else:
                        raise ValueError("unexpected body message type")
                except (ValueError, TypeError, KeyError) as error:
                    generation[0] += 1
                    self._reason = f"invalid_packet: {error}"
                    self._record("error", {"reason": self._reason})
        except (ConnectionClosed, TimeoutError):
            pass
        except (ValueError, TypeError) as error:
            self._record("error", {"reason": str(error)})
            await websocket.close(code=1008, reason="invalid body hello")
        finally:
            stopped.set()
            available.set()
            # Do not cancel an in-flight to_thread IK solve: join it before
            # accepting another producer or closing its raw log/pipeline.
            if not clock_queue.full():
                clock_queue.put_nowait(None)
            for task in tasks[2:]:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._active = False
            self._reason = "producer_disconnected"
            self._record("event", {"type": "producer_disconnected"})

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        try:
            if self._acquisition is not None:
                self._acquisition.close()
            if self._live is not None:
                await asyncio.to_thread(self._live.close)
        finally:
            if self._wire is not None:
                self._wire.close()
                self._wire = None
