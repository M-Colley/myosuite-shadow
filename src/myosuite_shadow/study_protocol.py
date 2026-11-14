"""Frozen study contracts and durable, ordered trial transitions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_APPLICATION_ID = 0x4D595354
_OUTCOME_FIELDS = {
    "name",
    "selector",
    "semantic_id",
    "unit",
    "coordinate_frame",
    "normalization",
    "method_version",
    "statistic",
}
_SELECTOR = re.compile(
    r"(?:tau_|reserve_)(?:0|[1-9][0-9]*)"
    r"|(?:activation|fatigue):[^:]+|contact:[^:]+:[xyz]"
)
_CONTEXT_FIELDS = {
    "study_id",
    "protocol_version",
    "definition_sha256",
    "participant_id",
    "participant_link_key",
    "run_id",
    "session_id",
}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _object(value: Any, fields: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) - fields:
        raise ValueError(f"{name} must have only fields {sorted(fields)}")
    return value


def _metadata(value: Any) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int | float):
        _number(value, "condition metadata number")
        return value
    if isinstance(value, list):
        return [_metadata(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _metadata(item) for key, item in value.items()}
    raise ValueError("condition metadata must contain finite JSON values")


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class StudyDefinition:
    """A normalized immutable contract; dictionaries are returned as copies."""

    _json: str

    @classmethod
    def from_dict(cls, value: dict) -> StudyDefinition:
        value = _object(
            value,
            {
                "schema_version",
                "study_id",
                "protocol_version",
                "conditions",
                "trials",
                "quality",
                "outcomes",
                "requirements",
            },
            "study definition",
        )
        if (
            isinstance(value.get("schema_version"), bool)
            or value.get("schema_version") != 1
        ):
            raise ValueError("study schema_version must be 1")
        if not isinstance(value["schema_version"], int):
            raise ValueError("study schema_version must be an integer")
        normalized = {
            "schema_version": 1,
            "study_id": _text(value.get("study_id"), "study_id"),
            "protocol_version": _text(
                value.get("protocol_version"), "protocol_version"
            ),
        }
        conditions = value.get("conditions")
        if not isinstance(conditions, dict) or not conditions:
            raise ValueError("conditions must map IDs to metadata objects")
        normalized["conditions"] = {}
        for name, metadata in conditions.items():
            _text(name, "condition ID")
            if not isinstance(metadata, dict):
                raise ValueError("condition metadata must be an object")
            normalized["conditions"][name] = _metadata(metadata)
        trials = value.get("trials")
        if not isinstance(trials, list) or not trials:
            raise ValueError("trials must be a nonempty ordered list")
        seen = set()
        normalized["trials"] = []
        for trial in trials:
            trial = _object(trial, {"trial_id", "condition"}, "trial")
            trial_id = _text(trial.get("trial_id"), "trial_id")
            if trial_id in seen:
                raise ValueError("trial IDs must be unique")
            seen.add(trial_id)
            condition = _text(trial.get("condition"), "trial condition")
            if condition not in conditions:
                raise ValueError("trial condition must select a declared condition")
            normalized["trials"].append({"trial_id": trial_id, "condition": condition})
        quality = _object(
            value.get("quality", {}),
            {
                "max_gap_s",
                "min_coverage",
                "max_rejected_fraction",
                "reject_flags",
                "require_tracking",
            },
            "quality",
        )
        normalized["quality"] = {
            "max_gap_s": _number(quality.get("max_gap_s", 0.1), "max_gap_s"),
            "min_coverage": _number(quality.get("min_coverage", 0.9), "min_coverage"),
            "max_rejected_fraction": _number(
                quality.get("max_rejected_fraction", 0.1), "max_rejected_fraction"
            ),
            "reject_flags": quality.get("reject_flags", 511),
            "require_tracking": quality.get("require_tracking", True),
        }
        quality = normalized["quality"]
        if quality["max_gap_s"] <= 0 or not all(
            0 <= quality[key] <= 1 for key in ("min_coverage", "max_rejected_fraction")
        ):
            raise ValueError(
                "max_gap_s must be positive; quality fractions must be in [0,1]"
            )
        if (
            isinstance(quality["reject_flags"], bool)
            or not isinstance(quality["reject_flags"], int)
            or quality["reject_flags"] < 0
        ):
            raise ValueError("reject_flags must be a nonnegative integer")
        if not isinstance(quality["require_tracking"], bool):
            raise ValueError("require_tracking must be boolean")
        requirements = _object(
            value.get("requirements", {}),
            {
                "model_sha256",
                "tracking_config_sha256",
                "required_calibrations",
            },
            "requirements",
        )
        normalized["requirements"] = {}
        for name in ("model_sha256", "tracking_config_sha256"):
            digest = requirements.get(name)
            if digest is not None:
                if not isinstance(digest, str) or not re.fullmatch(
                    r"[0-9a-fA-F]{64}", digest
                ):
                    raise ValueError(f"{name} must be a 64-character SHA256")
                digest = digest.lower()
            normalized["requirements"][name] = digest
        calibrations = requirements.get("required_calibrations", [])
        if not isinstance(calibrations, list):
            raise ValueError("required_calibrations must be a list")
        calibrations = [_text(name, "required calibration") for name in calibrations]
        if len(set(calibrations)) != len(calibrations):
            raise ValueError("required calibrations must be unique")
        normalized["requirements"]["required_calibrations"] = calibrations
        outcomes = value.get("outcomes")
        if not isinstance(outcomes, list):
            raise ValueError("outcomes must be an explicit list")
        normalized["outcomes"], seen = [], set()
        for outcome in outcomes:
            outcome = _object(outcome, _OUTCOME_FIELDS, "outcome")
            if set(outcome) != _OUTCOME_FIELDS:
                raise ValueError("every outcome identity field must be explicit")
            outcome = {
                key: _text(outcome[key], f"outcome {key}") for key in _OUTCOME_FIELDS
            }
            if outcome["name"] in seen:
                raise ValueError("outcome names must be unique")
            seen.add(outcome["name"])
            if not _SELECTOR.fullmatch(outcome["selector"]) or not all(
                part.strip() for part in outcome["selector"].split(":")
            ):
                raise ValueError("unsupported outcome selector")
            if outcome["statistic"] not in {"time_mean", "rms", "peak"}:
                raise ValueError("unsupported outcome statistic")
            normalized["outcomes"].append(outcome)
        return cls(_canonical(normalized))

    @classmethod
    def load(cls, path: str | Path) -> StudyDefinition:
        return cls.from_dict(
            json.loads(
                Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_pairs
            )
        )

    def record(self) -> dict[str, Any]:
        return json.loads(self._json)

    @property
    def data(self) -> dict[str, Any]:
        return self.record()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self._json.encode("utf-8")).hexdigest()


def _derive(
    definition: StudyDefinition,
    context: dict,
    events: list[dict],
    session_path: str | None,
) -> dict:
    trials = [
        dict(trial, state="pending", intervals=[], incomplete=False)
        for trial in definition.data["trials"]
    ]
    by_id = {trial["trial_id"]: trial for trial in trials}
    active, segment, previous = None, None, None
    for index, event in enumerate(events, 1):
        stamp = _number(event["time_s"], "event time_s")
        _number(event["recorded_unix"], "event recorded_unix")
        if event["event_id"] != index or previous is not None and stamp < previous:
            raise ValueError("trial events must be contiguous and monotonic")
        previous = stamp
        kind, trial_id = event["kind"], event["trial_id"]
        if trial_id not in by_id:
            raise ValueError("unknown trial ID")
        if kind == "start":
            pending = next(
                (trial for trial in trials if trial["state"] == "pending"), None
            )
            if active is not None or pending is None or pending["trial_id"] != trial_id:
                raise ValueError(
                    "start requires the next pending trial and no active trial"
                )
            active, segment = by_id[trial_id], stamp
            active["state"] = "running"
        else:
            if active is None or active["trial_id"] != trial_id:
                raise ValueError("transition requires an active trial")
            if kind == "pause" and active["state"] == "running":
                active["intervals"].append([segment, stamp])
                active["state"], segment = "paused", None
            elif kind == "resume" and active["state"] == "paused":
                active["state"], segment = "running", stamp
            elif kind in {"end", "abort"} and active["state"] in {"running", "paused"}:
                if segment is not None:
                    active["intervals"].append([segment, stamp])
                active["state"] = "completed" if kind == "end" else "aborted"
                active, segment = None, None
            else:
                raise ValueError("invalid trial transition")
    if active is not None:
        active["incomplete"] = True
    state = (
        active["state"]
        if active is not None
        else (
            "ready"
            if any(trial["state"] == "pending" for trial in trials)
            else "completed"
        )
    )
    return {
        "context": context,
        "definition": definition.record(),
        "definition_sha256": definition.sha256,
        "state": state,
        "trials": trials,
        "events": events,
        "session_path": session_path,
    }


class TrialJournal:
    """Atomic ordered events; unclosed running segments remain incomplete."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def create(
        cls,
        path: str | Path,
        definition: StudyDefinition,
        *,
        participant_id: str,
        run_id: str,
        participant_link_key: str | None = None,
        session_id: str = "session",
    ) -> TrialJournal:
        if not isinstance(definition, StudyDefinition):
            raise TypeError("definition must be a StudyDefinition")
        definition = StudyDefinition.from_dict(definition.record())
        context = {
            "study_id": definition.data["study_id"],
            "protocol_version": definition.data["protocol_version"],
            "definition_sha256": definition.sha256,
            "participant_id": _text(participant_id, "participant_id"),
            "participant_link_key": None
            if participant_link_key is None
            else _text(participant_link_key, "participant_link_key"),
            "run_id": _text(run_id, "run_id"),
            "session_id": _text(session_id, "session_id"),
        }
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        journal = cls(path)
        try:
            with journal._transaction() as connection:
                connection.execute(f"PRAGMA application_id={_APPLICATION_ID}")
                connection.execute("PRAGMA user_version=1")
                connection.execute(
                    "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE events (event_id INTEGER PRIMARY KEY, "
                    "kind TEXT NOT NULL, trial_id TEXT NOT NULL, "
                    "time_s REAL NOT NULL, recorded_unix REAL NOT NULL)"
                )
                connection.executemany(
                    "INSERT INTO metadata VALUES (?,?)",
                    [
                        ("definition", definition._json),
                        ("context", _canonical(context)),
                        ("session_path", "null"),
                    ],
                )
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return journal

    @classmethod
    def open(cls, path: str | Path) -> TrialJournal:
        journal = cls(Path(path).resolve())
        if not journal.path.is_file():
            raise FileNotFoundError(journal.path)
        journal.snapshot()
        return journal

    @contextmanager
    def _transaction(self, *, write: bool = True):
        connection = sqlite3.connect(
            self.path.as_uri() + "?mode=rw", uri=True, timeout=10, isolation_level=None
        )
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _read(self, connection) -> dict:
        if (
            connection.execute("PRAGMA application_id").fetchone()[0] != _APPLICATION_ID
            or connection.execute("PRAGMA user_version").fetchone()[0] != 1
        ):
            raise ValueError("unsupported trial journal")
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if set(metadata) != {"definition", "context", "session_path"}:
            raise ValueError("trial journal metadata is incomplete")
        definition = StudyDefinition.from_dict(json.loads(metadata["definition"]))
        context = json.loads(metadata["context"])
        if not isinstance(context, dict) or set(context) != _CONTEXT_FIELDS:
            raise ValueError("invalid trial journal context")
        for key in ("participant_id", "run_id", "session_id"):
            _text(context[key], key)
        if context["participant_link_key"] is not None:
            _text(context["participant_link_key"], "participant_link_key")
        for key in ("study_id", "protocol_version"):
            if context[key] != definition.data[key]:
                raise ValueError("trial journal context differs from frozen definition")
        if context["definition_sha256"] != definition.sha256:
            raise ValueError("trial journal definition digest mismatch")
        session_path = json.loads(metadata["session_path"])
        if session_path is not None:
            _text(session_path, "session_path")
            if not Path(session_path).is_absolute():
                raise ValueError("journal session path must be absolute")
        fields = ("event_id", "kind", "trial_id", "time_s", "recorded_unix")
        events = [
            dict(zip(fields, row))
            for row in connection.execute(
                "SELECT event_id,kind,trial_id,time_s,recorded_unix "
                "FROM events ORDER BY event_id"
            )
        ]
        return _derive(definition, context, events, session_path)

    @property
    def definition(self) -> StudyDefinition:
        return StudyDefinition.from_dict(self.snapshot()["definition"])

    @property
    def context(self) -> dict:
        return self.snapshot()["context"]

    @property
    def session_path(self) -> Path | None:
        path = self.snapshot()["session_path"]
        return None if path is None else Path(path)

    def bind_session(self, path: str | Path) -> None:
        path = str(Path(path).resolve())
        with self._transaction() as connection:
            snapshot = self._read(connection)
            if snapshot["session_path"] not in (None, path):
                raise ValueError(
                    "trial journal is already bound to a different session"
                )
            if snapshot["session_path"] is None:
                connection.execute(
                    "UPDATE metadata SET value=? WHERE key='session_path'",
                    (_canonical(path),),
                )

    def snapshot(self) -> dict[str, Any]:
        with self._transaction(write=False) as connection:
            return self._read(connection)

    def _transition(
        self, kind: str, time_s: float, trial_id: str | None = None
    ) -> dict:
        stamp = _number(time_s, "time_s")
        with self._transaction() as connection:
            snapshot = self._read(connection)
            if trial_id is None:
                active = [
                    trial
                    for trial in snapshot["trials"]
                    if trial["state"] in {"running", "paused"}
                ]
                if len(active) != 1:
                    raise ValueError("transition requires an active trial")
                trial_id = active[0]["trial_id"]
            event = {
                "event_id": len(snapshot["events"]) + 1,
                "kind": kind,
                "trial_id": _text(trial_id, "trial_id"),
                "time_s": stamp,
                "recorded_unix": _number(time.time(), "recorded_unix"),
            }
            result = _derive(
                StudyDefinition.from_dict(snapshot["definition"]),
                snapshot["context"],
                snapshot["events"] + [event],
                snapshot["session_path"],
            )
            connection.execute(
                "INSERT INTO events VALUES (?,?,?,?,?)",
                tuple(
                    event[key]
                    for key in (
                        "event_id",
                        "kind",
                        "trial_id",
                        "time_s",
                        "recorded_unix",
                    )
                ),
            )
        return result

    def start(self, trial_id: str, time_s: float) -> dict:
        return self._transition("start", time_s, trial_id)

    def pause(self, time_s: float) -> dict:
        return self._transition("pause", time_s)

    def resume(self, time_s: float) -> dict:
        return self._transition("resume", time_s)

    def end(self, time_s: float) -> dict:
        return self._transition("end", time_s)

    def abort(self, time_s: float) -> dict:
        return self._transition("abort", time_s)
