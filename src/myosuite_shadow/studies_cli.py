"""Prepare, record and compare versioned studies with explicit trial timing."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from myosuite_shadow.study_analysis import study_report
from myosuite_shadow.study_catalog import StudyCatalog, check_learning_eligibility
from myosuite_shadow.study_protocol import StudyDefinition, TrialJournal


def main(argv: list[str] | None = None) -> int:
    """The study SDK and CLI share the same durable trial journal."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser(
        "init", help="freeze a study definition and run identity"
    )
    init.add_argument("definition", type=Path)
    init.add_argument("journal", type=Path)
    init.add_argument("--participant", required=True)
    init.add_argument("--run-id", required=True)
    init.add_argument("--session-id", default="session")
    init.add_argument("--participant-link-key")
    bind = commands.add_parser(
        "bind", help="bind a run to its intended session directory"
    )
    bind.add_argument("journal", type=Path)
    bind.add_argument("session", type=Path)
    status = commands.add_parser(
        "status", help="read run identity, states and intervals"
    )
    status.add_argument("journal", type=Path)
    trial = commands.add_parser(
        "trial", help="record a transition at mapped session time"
    )
    trial.add_argument("journal", type=Path)
    trial.add_argument("action", choices=("start", "pause", "resume", "end", "abort"))
    trial.add_argument(
        "--time",
        type=float,
        required=True,
        help="mapped session seconds, not Unix time",
    )
    trial.add_argument("--trial-id", help="required for start")
    report = commands.add_parser(
        "report", help="summarize a bound, closed session by trial"
    )
    report.add_argument("journal", type=Path)
    report.add_argument("out", type=Path)
    add = commands.add_parser(
        "catalog-add", help="index an immutable trial report locally"
    )
    add.add_argument("catalog", type=Path)
    add.add_argument("report", type=Path)
    listing = commands.add_parser(
        "catalog-list", help="list locally indexed study runs"
    )
    listing.add_argument("catalog", type=Path)
    compare = commands.add_parser(
        "compare", help="paired participant-level condition contrast"
    )
    compare.add_argument("catalog", type=Path)
    compare.add_argument("outcome")
    compare.add_argument("condition_a")
    compare.add_argument("condition_b")
    compare.add_argument("--study-id", action="append", dest="study_ids")
    learning = commands.add_parser(
        "learning-check", help="check declared lineage and frozen splits"
    )
    learning.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            journal = TrialJournal.create(
                args.journal,
                StudyDefinition.load(args.definition),
                participant_id=args.participant,
                run_id=args.run_id,
                participant_link_key=args.participant_link_key,
                session_id=args.session_id,
            )
            result = journal.snapshot()
        elif args.command == "bind":
            journal = TrialJournal.open(args.journal)
            journal.bind_session(args.session)
            result = journal.snapshot()
        elif args.command == "status":
            result = TrialJournal.open(args.journal).snapshot()
        elif args.command == "trial":
            journal = TrialJournal.open(args.journal)
            if args.action == "start":
                if not args.trial_id:
                    raise ValueError("start requires --trial-id")
                journal.start(args.trial_id, args.time)
            else:
                if args.trial_id:
                    raise ValueError("--trial-id is only used for start")
                getattr(journal, args.action)(args.time)
            result = journal.snapshot()
        elif args.command == "report":
            result = study_report(args.journal)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
        elif args.command == "catalog-add":
            with StudyCatalog(args.catalog) as catalog:
                result = {
                    "report_sha256": catalog.add(
                        json.loads(args.report.read_text(encoding="utf-8-sig"))
                    )
                }
        elif args.command == "catalog-list":
            with StudyCatalog(args.catalog) as catalog:
                result = {"runs": catalog.list()}
        elif args.command == "compare":
            with StudyCatalog(args.catalog) as catalog:
                result = catalog.compare(
                    args.outcome,
                    args.condition_a,
                    args.condition_b,
                    study_ids=args.study_ids,
                )
        else:
            result = check_learning_eligibility(
                **json.loads(args.manifest.read_text(encoding="utf-8-sig"))
            )
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        RuntimeError,
        sqlite3.Error,
    ) as error:
        parser.exit(2, f"study {args.command} failed: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return (
        1
        if result.get("eligible") is False
        or result.get("status") in {"not_comparable", "no_pairs"}
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
