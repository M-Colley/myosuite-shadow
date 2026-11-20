"""Freeze, audit or reprocess externally stored study recordings."""

from __future__ import annotations

import argparse
import json

from myosuite_shadow.recordings import (
    audit_recording,
    freeze_recording,
    run_recording_regression,
)


def main(argv: list[str] | None = None) -> int:
    """Return 1 for failed regression assertions, 2 for invalid inputs, else 0."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="hash externally supplied files")
    freeze.add_argument("draft")
    freeze.add_argument("destination")
    audit = commands.add_parser("audit", help="verify identities and frozen hashes")
    audit.add_argument("manifest")
    replay = commands.add_parser("run", help="reprocess and compare a frozen recording")
    replay.add_argument("manifest")
    replay.add_argument("out_dir", help="new output directory")
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze_recording(args.draft, args.destination)
        elif args.command == "audit":
            result = audit_recording(args.manifest)
        else:
            result = run_recording_regression(args.manifest, args.out_dir)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        parser.exit(2, f"recording {args.command} failed: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if result.get("passed") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
