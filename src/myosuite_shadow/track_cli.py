"""Run a Quest UDP stream or reprocess recorded canonical tracker frames."""

from __future__ import annotations

import argparse
import dataclasses
import json
from contextlib import ExitStack, nullcontext
from pathlib import Path

import mujoco

from myosuite_shadow.acquisition import AcquisitionConfig, TrackedSession
from myosuite_shadow.ik import IKConfig, Keypoint, KeypointIK
from myosuite_shadow.live import LiveShadowModel
from myosuite_shadow.log import SessionWriter
from myosuite_shadow.pipeline import ShadowModel, ShadowModelConfig
from myosuite_shadow.tracking import (
    ClockMapping,
    QuestClockSynchronizer,
    QuestDecoder,
    Registration,
    UdpSource,
    read_frames,
)


def main(argv: list[str] | None = None) -> None:
    """Run with an explicit subject model, landmark map, clock and registration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--frames", type=Path, help="canonical 3D JSONL for offline reprocessing"
    )
    source.add_argument("--quest-port", type=int, help="UDP port of the Quest receiver")
    parser.add_argument(
        "--host", default="127.0.0.1", help="explicit LAN interface for a remote Quest"
    )
    parser.add_argument(
        "--max-frames", type=int, help="stop after this many received frames"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume this session with fresh registration",
    )
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.frames is not None:
        tracking_output = args.log_dir / "tracking.jsonl"
        if args.frames.resolve() == tracking_output.resolve() or (
            args.frames.exists()
            and tracking_output.exists()
            and args.frames.samefile(tracking_output)
        ):
            parser.error("--frames must differ from the session tracking.jsonl output")
    settings = json.loads(args.config.read_text(encoding="utf-8-sig"))
    model_path = Path(settings["model"])
    if not model_path.is_absolute():
        model_path = args.config.parent / model_path
    model = (
        mujoco.MjModel.from_binary_path(str(model_path))
        if model_path.suffix == ".mjb"
        else mujoco.MjModel.from_xml_path(str(model_path))
    )
    ik_values = dict(settings.get("ik", {}))
    if "required_keypoints" in ik_values:
        ik_values["required_keypoints"] = tuple(ik_values["required_keypoints"])
    ik = KeypointIK(
        model, [Keypoint(**p) for p in settings["keypoints"]], IKConfig(**ik_values)
    )
    registration = (
        Registration(**settings["registration"]) if "registration" in settings else None
    )
    policy = AcquisitionConfig(**settings.get("acquisition", {}))
    if args.frames is not None:
        policy = dataclasses.replace(policy, check_wallclock=False)
        decoder = None
    else:
        quest = settings["quest"]
        decoder = QuestDecoder(
            ClockMapping(**quest["clock"]),
            expected_source_id=quest.get("source_id"),
            expected_person_id=quest.get("person_id"),
        )
    if args.resume:
        recorded = SessionWriter.read_record(args.log_dir)
        if "shadow" in settings:
            with ShadowModel(
                model, ShadowModelConfig.from_dict(settings["shadow"])
            ) as candidate:
                expected = candidate.config
            if expected != ShadowModelConfig.from_dict(recorded["config"]):
                parser.error(
                    "resume settings must match the recorded shadow configuration"
                )
    with ExitStack() as stack:
        clock_client = None
        if args.frames is None and quest.get("clock_host"):
            sync_options = dict(quest.get("clock_sync", {}))
            sync_options.setdefault("max_uncertainty_s", policy.max_clock_uncertainty_s)
            clock_client = stack.enter_context(
                QuestClockSynchronizer(
                    quest["clock_host"],
                    initial=decoder.clock,
                    **sync_options,
                )
            )
            decoder.clock_provider = clock_client.get_mapping
        stream = (
            nullcontext(read_frames(args.frames))
            if args.frames is not None
            else UdpSource(decoder, host=args.host, port=args.quest_port)
        )
        frames = stack.enter_context(stream)
        if args.resume:
            shadow = ShadowModel.resume(args.log_dir, model)
        else:
            shadow = ShadowModel(
                model,
                ShadowModelConfig.from_dict(settings.get("shadow", {})),
                log_dir=args.log_dir,
                metadata={
                    **settings.get("metadata", {}),
                    "tracking_config": settings,
                    "acquisition_mode": "offline_reprocess" if args.frames else "live",
                },
                calibration=settings.get("calibration", {}),
            )
        try:
            live = LiveShadowModel(shadow)
        except BaseException:
            shadow.close()
            raise
        stack.enter_context(live)
        with TrackedSession(
            live, ik, registration=registration, config=policy
        ) as session:
            if args.resume:
                live.reset_tracking()
            try:
                for count, frame in enumerate(frames, 1):
                    session.submit(frame)
                    if args.frames is not None:
                        live.drain()
                    elif count == 1 or count % 90 == 0:
                        health = session.health
                        if clock_client is not None:
                            health["clock"] = clock_client.health
                        print(json.dumps(health, allow_nan=False))
                    if args.max_frames is not None and count >= args.max_frames:
                        break
            except (TimeoutError, KeyboardInterrupt):
                live.mark("acquisition_stopped", reason="timeout_or_operator")
            finally:
                live.mark(
                    "acquisition_summary",
                    accepted=session.accepted,
                    rejected=session.rejected,
                )
                print(f"accepted {session.accepted}, rejected {session.rejected}")


if __name__ == "__main__":
    main()
