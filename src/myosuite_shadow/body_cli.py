"""Configure, register, serve and exercise a live full-body shadow."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from myosuite_shadow.body_protocol import SCHEMA, BodySource, finite
from myosuite_shadow.body_server import BodyServer
from myosuite_shadow.tracking import fit_registration
from myosuite_shadow.tracking_config import load_tracking_setup


def initialize(
    out_dir: Path,
    *,
    profile: str = "full_body",
    source: str = "unity",
    person_id: str = "P01",
) -> Path:
    """Export a reference model and an explicit, editable study configuration."""
    from myosuite_shadow.body_profiles import load_body_profile

    body = load_body_profile(profile)
    out_dir.mkdir(parents=True, exist_ok=False)
    model_path = out_dir / "model.mjb"
    mujoco.mj_saveModel(body.model, str(model_path), None)
    synthetic = source == "synthetic"
    config = {
        "model": "model.mjb",
        "keypoints": [dataclasses.asdict(p) for p in body.keypoints],
        "ik": {
            "required_keypoints": [p.name for p in body.keypoints],
            "quality_dofs": list(body.target_dofs),
        },
        "acquisition": {"allow_inferred": False, "max_age_s": 0.25},
        "shadow": {
            "window": 9,
            "lag": 3,
            "root_residual_force_limit": 50.0,
            "root_residual_moment_limit": 10.0,
            "optimization": {"contact_forces": True},
            "gap_load": "hold",
        },
        "body_source": dataclasses.asdict(
            BodySource(
                source_id={
                    "unity": "quest-unity",
                    "webxr": "quest-webxr",
                    "synthetic": "synthetic-model",
                }[source],
                person_id=person_id,
                clock_id={
                    "unity": "unity_realtime",
                    "webxr": "webxr_performance",
                    "synthetic": "synthetic_monotonic",
                }[source],
                coordinate_frame="model_world" if synthetic else "quest_local_floor",
            )
        ),
        "body_profile": body.name,
        "preflight": {"target_dofs": list(body.target_dofs), "require_full_rank": True},
        "model_status": "reference_model_not_participant_calibrated",
        "synthetic": synthetic,
        "notes": [
            "Register measured source landmarks to the matching model posture.",
            "Review landmark sites, participant scaling, contacts and external loads.",
            "Client observation time does not verify native pose capture latency.",
            "Hold during gaps is an explicit unobserved-load assumption.",
        ],
    }
    if synthetic:
        # Start the synthetic demonstration inside joint ranges.
        pose = (
            body.model.key_qpos[0].copy()
            if body.model.nkey
            else body.model.qpos0.copy()
        )
        for name in (
            "elbow_flexion_l",
            "elbow_flexion_r",
            "knee_angle_l",
            "knee_angle_r",
        ):
            pose[body.model.jnt_qposadr[body.model.joint(name).id]] = 0.05
        config["initial_qpos"] = pose.tolist()
    path = out_dir / "body.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def register(
    config_path: Path, pairs_path: Path, output: Path, max_error_m: float
) -> None:
    """Fit only explicitly paired metric landmarks; never invent a posture."""
    if output.exists():
        raise ValueError("registration output already exists")
    setup = load_tracking_setup(config_path)
    pairs = json.loads(pairs_path.read_text(encoding="utf-8-sig"))
    source = BodySource(**setup.settings["body_source"])
    if pairs["source_frame"] != source.coordinate_frame:
        raise ValueError("registration pairs use another source coordinate frame")
    names = sorted(pairs["source"])
    if set(names) != set(pairs["target"]):
        raise ValueError(
            "registration needs identical named source and target landmarks"
        )
    registration = fit_registration(
        np.array([pairs["source"][n] for n in names]),
        np.array([pairs["target"][n] for n in names]),
        source_frame=source.coordinate_frame,
        revision=pairs["revision"],
        max_error_m=max_error_m,
    )
    settings = {
        **setup.settings,
        "model": str(setup.model_path),
        "registration": registration.record(),
        "registration_evidence": {
            "pairs": pairs,
            "max_error_m": max_error_m,
            "meaning": "rigid spatial alignment; participant calibration is separate",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(settings, indent=2) + "\n")


async def synthetic_stream(config_path: Path, uri: str, seconds: float = 10.0) -> None:
    """Send FK landmarks through the real wire protocol, with a real clock fit."""
    from websockets.asyncio.client import connect

    seconds = finite(seconds, "synthetic seconds")
    if seconds <= 0:
        raise ValueError("synthetic seconds must be positive")
    setup = load_tracking_setup(config_path)
    source = BodySource(**setup.settings["body_source"])
    if not setup.settings.get("synthetic") or source.coordinate_frame != "model_world":
        raise ValueError(
            "synthetic sender requires an explicitly synthetic configuration"
        )
    data = mujoco.MjData(setup.model)
    data.qpos[:] = setup.ik.configuration.q
    mujoco.mj_forward(setup.model, data)
    points = {}
    for point in setup.ik.keypoints:
        kind = {
            "body": mujoco.mjtObj.mjOBJ_BODY,
            "site": mujoco.mjtObj.mjOBJ_SITE,
            "geom": mujoco.mjtObj.mjOBJ_GEOM,
        }[point.frame_type]
        index = mujoco.mj_name2id(setup.model, kind, point.frame)
        points[point.name] = getattr(
            data,
            {"body": "xpos", "site": "site_xpos", "geom": "geom_xpos"}[
                point.frame_type
            ],
        )[index].tolist()
    async with connect(uri, max_size=2**20) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "hello",
                    "schema": SCHEMA,
                    "source_id": source.source_id,
                    "person_id": source.person_id,
                    "clock_id": source.clock_id,
                    "metadata": {
                        "synthetic": True,
                        "pose_time_semantics": "client_observation_estimate",
                    },
                }
            )
        )

        async def receive():
            async for raw in websocket:
                packet = json.loads(raw)
                if packet.get("type") == "clock_probe":
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "clock_reply",
                                "schema": SCHEMA,
                                "clock_id": source.clock_id,
                                "nonce": packet["nonce"],
                                "source_time_s": time.perf_counter(),
                            }
                        )
                    )

        reader = asyncio.create_task(receive())
        start = time.perf_counter()
        sequence = 0
        try:
            while time.perf_counter() - start < seconds:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "body_frame",
                            "schema": SCHEMA,
                            **dataclasses.asdict(source),
                            "sequence": sequence,
                            "time_s": time.perf_counter(),
                            "positions": points,
                            "position_tracked": dict.fromkeys(points, True),
                            "confidence": dict.fromkeys(points, 1.0),
                            "transform_revision": 0,
                            "skeleton_revision": 0,
                            "actual_fidelity": "high",
                            "metadata": {"synthetic": True},
                        }
                    )
                )
                sequence += 1
                await asyncio.sleep(1 / 30)
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)


async def serve(
    config_path: Path, log_dir: Path, host: str, port: int, strict_pose_timing: bool
) -> None:
    setup = load_tracking_setup(config_path)
    source = BodySource(**setup.settings["body_source"])
    if source.coordinate_frame != "model_world" and setup.registration is None:
        raise ValueError(
            "measured tracker-to-model registration is required before serving"
        )
    server = BodyServer(
        setup,
        log_dir=log_dir,
        source=source,
        profile=setup.settings.get("body_profile", "full_body"),
        strict_pose_timing=strict_pose_timing,
    )
    try:
        await server.start(host, port)
        print(f"Body shadow listening at {server.uri}", flush=True)
        await asyncio.Event().wait()
    finally:
        await server.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser(
        "init", help="export a reference model and configuration"
    )
    init.add_argument("--out-dir", type=Path, required=True)
    init.add_argument("--profile", choices=["full_body"], default="full_body")
    init.add_argument(
        "--source", choices=["unity", "webxr", "synthetic"], default="unity"
    )
    init.add_argument("--person-id", default="P01")
    align = commands.add_parser(
        "register", help="fit explicitly paired metric landmarks"
    )
    align.add_argument("--config", type=Path, required=True)
    align.add_argument("--pairs", type=Path, required=True)
    align.add_argument("--out", type=Path, required=True)
    align.add_argument("--max-error-m", type=float, default=0.02)
    run = commands.add_parser("serve", help="serve the live body shadow over WebSocket")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--log-dir", type=Path, required=True)
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=9766)
    run.add_argument("--strict-pose-timing", action="store_true")
    demo = commands.add_parser(
        "synthetic", help="exercise the complete protocol using model FK"
    )
    demo.add_argument("--config", type=Path, required=True)
    demo.add_argument("--uri", default="ws://127.0.0.1:9766")
    demo.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            print(
                initialize(
                    args.out_dir,
                    profile=args.profile,
                    source=args.source,
                    person_id=args.person_id,
                )
            )
        elif args.command == "register":
            register(args.config, args.pairs, args.out, args.max_error_m)
        elif args.command == "synthetic":
            asyncio.run(synthetic_stream(args.config, args.uri, args.seconds))
        else:
            asyncio.run(
                serve(
                    args.config,
                    args.log_dir,
                    args.host,
                    args.port,
                    args.strict_pose_timing,
                )
            )
    except KeyboardInterrupt:
        pass
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"body bridge failed: {error}\n")


if __name__ == "__main__":
    main()
