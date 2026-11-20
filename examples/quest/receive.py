"""Synchronize a Quest sender clock or record its validated canonical frames."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

from myosuite_shadow.tracking import (
    ClockMapping,
    QuestClockSynchronizer,
    QuestDecoder,
    RawFrameWriter,
    UdpSource,
    synchronize_quest_clock,
)


def sync_clock(args: argparse.Namespace) -> None:
    """Measure remote/host offset and rate with uncertainty and a validity bound."""
    epoch = args.session_epoch
    clock = synchronize_quest_clock(
        args.host,
        port=args.clock_port,
        session_epoch=epoch,
        count=args.count,
        interval_s=args.interval,
        validity_s=args.validity,
        max_rtt_s=args.max_rtt,
        max_uncertainty_s=args.max_uncertainty,
    )
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(clock.record(), output, indent=2)
    print(
        json.dumps(
            {
                "clock": str(args.out),
                "uncertainty_s": clock.uncertainty_s,
                "rate": clock.rate,
                "session_epoch": epoch,
            }
        )
    )


def receive(args: argparse.Namespace) -> None:
    """Record pre-IK frames; registration and study acceptance happen downstream."""
    clock = ClockMapping(**json.loads(args.clock.read_text(encoding="utf-8")))
    with ExitStack() as stack:
        provider = None
        if args.clock_host:
            client = stack.enter_context(
                QuestClockSynchronizer(
                    args.clock_host,
                    initial=clock,
                    port=args.clock_port,
                    session_epoch=args.session_epoch,
                    refresh_s=args.clock_refresh_s,
                )
            )
            provider = client.get_mapping
        decoder = QuestDecoder(
            clock, expected_person_id=args.person, clock_provider=provider
        )
        source = stack.enter_context(
            UdpSource(decoder, host=args.bind, port=args.port, timeout_s=args.timeout)
        )
        output = stack.enter_context(RawFrameWriter(args.out))
        for i, frame in enumerate(source):
            output.write(frame)
            if i % 90 == 0:
                print(
                    json.dumps(
                        {
                            "sequence": frame.sequence,
                            "time_s": frame.time_s,
                            "joints": len(frame.positions),
                            "inferred": len(frame.inferred_joints),
                            "fidelity": frame.metadata["actual_fidelity"],
                        }
                    )
                )


def main() -> None:
    """Run explicit clock calibration or acquisition."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sync = commands.add_parser("sync")
    sync.add_argument("--host", required=True, help="Quest IPv4 address")
    sync.add_argument("--clock-port", type=int, default=9764)
    sync.add_argument(
        "--session-epoch",
        type=float,
        default=0,
        help="shared host perf_counter epoch (default raw host monotonic)",
    )
    sync.add_argument("--count", type=int, default=30)
    sync.add_argument("--interval", type=float, default=0.2)
    sync.add_argument("--validity", type=float, default=60)
    sync.add_argument("--max-rtt", type=float, default=0.05)
    sync.add_argument("--max-uncertainty", type=float, default=0.01)
    sync.add_argument("--out", type=Path, required=True)
    sync.set_defaults(run=sync_clock)
    record = commands.add_parser("receive")
    record.add_argument("--clock", type=Path, required=True)
    record.add_argument("--person", required=True)
    record.add_argument("--bind", default="127.0.0.1")
    record.add_argument("--port", type=int, default=9763)
    record.add_argument("--timeout", type=float, default=2)
    record.add_argument("--out", type=Path, required=True)
    record.add_argument("--clock-host", help="Quest IP for automatic clock refresh")
    record.add_argument("--clock-port", type=int, default=9764)
    record.add_argument("--clock-refresh-s", type=float, default=20)
    record.add_argument("--session-epoch", type=float, default=0)
    record.set_defaults(run=receive)
    args = parser.parse_args()
    if args.command == "sync" and (args.count < 3 or args.interval <= 0):
        parser.error(
            "clock sync needs at least three exchanges and a positive interval"
        )
    args.run(args)


if __name__ == "__main__":
    main()
