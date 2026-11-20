"""Run a Quest UDP stream or reprocess recorded canonical tracker frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack, nullcontext
from pathlib import Path

from myosuite_shadow.acquisition import TrackedSession
from myosuite_shadow.live import LiveShadowModel
from myosuite_shadow.log import SessionWriter, model_sha256
from myosuite_shadow.pipeline import ShadowModel, ShadowModelConfig
from myosuite_shadow.preflight import preflight
from myosuite_shadow.study_protocol import TrialJournal
from myosuite_shadow.tracking import (
    ClockMapping,
    QuestClockSynchronizer,
    QuestDecoder,
    UdpSource,
    read_frames,
)
from myosuite_shadow.tracking_config import load_tracking_setup


def main(argv: list[str] | None = None) -> None:
    """Run with an explicit subject model, landmark map, clock and registration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--study-run", type=Path, help="durable study trial journal")
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
    setup = load_tracking_setup(args.config, offline=args.frames is not None)
    settings, model, ik = setup.settings, setup.model, setup.ik
    registration, policy = setup.registration, setup.acquisition
    metadata = dict(settings.get("metadata", {}))
    journal = TrialJournal.open(args.study_run) if args.study_run else None
    if journal is not None:
        requirements = journal.definition.data["requirements"]
        if requirements.get("model_sha256") not in (None, model_sha256(model)):
            parser.error("study requires a different calibrated model")
        if requirements.get("tracking_config_sha256") not in (
            None,
            hashlib.sha256(args.config.read_bytes()).hexdigest(),
        ):
            parser.error("tracking configuration does not match the frozen study")
        calibration = settings.get("calibration", {})
        if not isinstance(calibration, dict) or any(
            not calibration.get(name) for name in requirements["required_calibrations"]
        ):
            parser.error("study is missing required calibration records")
        check = preflight(args.config, live=args.frames is None)
        if check["status"] == "blocked":
            parser.error(
                "study preflight is blocked; run myosuite-shadow-preflight for details"
            )
        metadata["study_preflight"] = check
        if "study" in metadata and metadata["study"] != journal.context:
            parser.error("configured study context disagrees with --study-run")
        metadata["study"] = journal.context
        if (
            journal.session_path
            and Path(journal.session_path).resolve() != args.log_dir.resolve()
        ):
            parser.error("study run is already bound to another session")
    if args.frames is not None:
        decoder = None
    else:
        quest = settings["quest"]
        decoder = QuestDecoder(
            ClockMapping(**quest["clock"]),
            expected_source_id=quest.get("source_id"),
            expected_person_id=quest.get("person_id"),
            expected_joint_set=quest.get("joint_set"),
        )
    if args.resume:
        recorded = SessionWriter.read_record(args.log_dir)
        if journal is None and recorded.get("metadata", {}).get("study"):
            parser.error("resuming a study-bound session requires --study-run")
        if (
            journal is not None
            and recorded.get("metadata", {}).get("study") != journal.context
        ):
            parser.error("resume study context must match the original session")
        if "shadow" in settings:
            with ShadowModel(
                model, ShadowModelConfig.from_dict(settings["shadow"])
            ) as candidate:
                expected = candidate.config
            if expected != ShadowModelConfig.from_dict(recorded["config"]):
                parser.error(
                    "resume settings must match the recorded shadow configuration"
                )
    if journal is not None:
        if not args.resume and (args.log_dir / "record.json").exists():
            parser.error("session exists; use --resume")
        journal.bind_session(args.log_dir)
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
                    **metadata,
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
