"""Immutable study evidence and conservative checks of declared learning lineage."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_SHA = re.compile(r"[0-9a-fA-F]{64}\Z")
_APPLICATION_ID = 0x4D594341
_CONTRACT = {
    "semantic_id",
    "unit",
    "coordinate_frame",
    "normalization",
    "method_version",
    "statistic",
}


def _canonical(value: Any) -> str:
    def check(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("JSON numbers must be finite")
        elif item is not None and not isinstance(item, str | bool | int):
            raise ValueError("only JSON scalars, lists and objects are supported")

    check(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _identity(contract: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, dict) or not _CONTRACT.issubset(contract):
        raise ValueError("outcome contract lacks semantic identity fields")
    for key in _CONTRACT:
        _text(contract[key], key)
    return {
        key: value for key, value in contract.items() if key not in {"name", "selector"}
    }


def _definitions(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    outcomes = report["definition"]["outcomes"]
    if not isinstance(outcomes, list):
        raise ValueError("definition.outcomes must be a list")
    result = {}
    for outcome in outcomes:
        name = _text(outcome["name"], "outcome name")
        if name in result:
            raise ValueError("duplicate outcome name")
        result[name] = _identity(outcome.get("contract", outcome))
    return result


def _validate_report(report: dict[str, Any]) -> None:
    _canonical(report)
    if (
        not isinstance(report, dict)
        or isinstance(report.get("schema_version"), bool)
        or report.get("schema_version") != 1
        or not isinstance(report["schema_version"], int)
    ):
        raise ValueError("study report requires schema_version=1")
    context, definition = report["context"], report["definition"]
    if not isinstance(context, dict) or not isinstance(definition, dict):
        raise ValueError("report context and definition must be objects")
    for key in (
        "study_id",
        "protocol_version",
        "participant_id",
        "run_id",
        "session_id",
    ):
        _text(context[key], key)
        if key in definition and definition[key] != context[key]:
            raise ValueError("report context contradicts definition")
    digest = context["definition_sha256"]
    if (
        not isinstance(digest, str)
        or not _SHA.fullmatch(digest)
        or digest.lower() != _digest(definition)
    ):
        raise ValueError("definition_sha256 does not match the canonical definition")
    if context.get("participant_link_key") is not None:
        _text(context["participant_link_key"], "participant_link_key")
    definitions = _definitions(report)
    if not isinstance(definition["conditions"], dict) or not isinstance(
        report["trials"], list
    ):
        raise ValueError("conditions must be an object and trials a list")
    planned = {t["trial_id"]: t["condition"] for t in definition.get("trials", [])}
    seen = set()
    for trial in report["trials"]:
        trial_id = _text(trial["trial_id"], "trial_id")
        if trial_id in seen:
            raise ValueError("duplicate trial_id within a report")
        seen.add(trial_id)
        if trial["condition"] not in definition["conditions"]:
            raise ValueError("trial condition is not defined")
        if planned and planned.get(trial_id) != trial["condition"]:
            raise ValueError("trial assignment contradicts the frozen protocol")
        for name, outcome in trial["outcomes"].items():
            if name not in definitions:
                raise ValueError("trial outcome is not defined")
            contract = _identity(outcome["contract"])
            if any(
                contract.get(key) != value for key, value in definitions[name].items()
            ):
                raise ValueError("trial outcome contradicts its declared contract")
    if planned and set(planned) != seen:
        raise ValueError("report must retain all planned trials")


class StudyCatalog:
    """SQLite catalog of immutable, strictly JSON-compatible study reports."""

    def __init__(self, path: str | Path) -> None:
        self._db = sqlite3.connect(str(path), timeout=30)
        try:
            self._db.execute("PRAGMA synchronous=FULL")
            with self._db:
                self._db.execute("BEGIN IMMEDIATE")
                application = self._db.execute("PRAGMA application_id").fetchone()[0]
                version = self._db.execute("PRAGMA user_version").fetchone()[0]
                tables = self._db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
                if application == 0 and version == 0 and not tables:
                    self._db.execute(f"PRAGMA application_id={_APPLICATION_ID}")
                    self._db.execute("PRAGMA user_version=1")
                elif application != _APPLICATION_ID or version != 1:
                    raise ValueError("database is not a supported study catalog")
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS reports (sha TEXT PRIMARY KEY, "
                    "study TEXT NOT NULL, run TEXT NOT NULL, payload TEXT NOT NULL, "
                    "UNIQUE(study, run))"
                )
        except BaseException:
            self._db.close()
            raise

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> StudyCatalog:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def add(self, report: dict[str, Any]) -> str:
        """Insert once; reject any changed report for an existing study/run."""
        _validate_report(report)
        payload, digest = _canonical(report), _digest(report)
        context = report["context"]
        with self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO reports VALUES (?, ?, ?, ?)",
                (digest, context["study_id"], context["run_id"], payload),
            )
            existing = self._db.execute(
                "SELECT sha, payload FROM reports WHERE study=? AND run=?",
                (context["study_id"], context["run_id"]),
            ).fetchone()
            if existing != (digest, payload):
                raise ValueError(
                    "immutable study_id/run_id already has a different report"
                )
        return digest

    def _reports(self) -> list[tuple[str, dict[str, Any]]]:
        result = []
        for digest, payload in self._db.execute(
            "SELECT sha, payload FROM reports ORDER BY study, run"
        ):
            report = json.loads(payload)
            if _digest(report) != digest:
                raise ValueError("catalog report hash mismatch")
            _validate_report(report)
            result.append((digest, report))
        return result

    def list(self) -> list[dict[str, Any]]:
        return [
            {
                "report_sha256": digest,
                **report["context"],
                "trials": len(report["trials"]),
            }
            for digest, report in self._reports()
        ]

    def compare(
        self,
        outcome_name: str,
        condition_a: str,
        condition_b: str,
        *,
        study_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Reduce repeated trials, then describe paired participant differences."""
        if condition_a == condition_b:
            raise ValueError("contrast requires distinct conditions")
        selected = None if study_ids is None else set(study_ids)
        reports = [
            r
            for _, r in self._reports()
            if selected is None or r["context"]["study_id"] in selected
        ]
        reports = [r for r in reports if outcome_name in _definitions(r)]
        counts: Counter = Counter(reports=len(reports))
        exclusions: Counter = Counter()
        samples = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        local_links = defaultdict(set)
        for item in reports:
            context = item["context"]
            link = context.get("participant_link_key")
            if link:
                local_links[(context["study_id"], context["participant_id"])].add(link)
        if any(len(links) > 1 for links in local_links.values()):
            raise ValueError("conflicting_participant_linkage")
        contract_json, conditions_json = None, None
        used, missing_links = set(), []
        study_set = {r["context"]["study_id"] for r in reports}
        for report in reports:
            context, definition = report["context"], report["definition"]
            try:
                conditions = _canonical(
                    [
                        definition["conditions"][condition_a],
                        definition["conditions"][condition_b],
                    ]
                )
            except KeyError as error:
                raise ValueError("condition_definition_mismatch") from error
            if conditions_json is not None and conditions != conditions_json:
                raise ValueError("condition_definition_mismatch")
            conditions_json = conditions
            links = local_links[(context["study_id"], context["participant_id"])]
            link = next(iter(links)) if links else None
            if link is None:
                missing_links.append(
                    context["study_id"] + ":" + context["participant_id"]
                )
            participant = (
                "link:" + link
                if link
                else "study:" + context["study_id"] + ":" + context["participant_id"]
            )
            source_digest = (
                report.get("provenance", {}).get("session_files", {}).get("record.json")
            )
            source = (
                ("file", source_digest)
                if isinstance(source_digest, str) and _SHA.fullmatch(source_digest)
                else (
                    "declared",
                    context["study_id"],
                    context["participant_id"],
                    context["session_id"],
                )
            )
            for trial in report["trials"]:
                counts["trials"] += 1
                identity = (*source, trial["trial_id"])
                if identity in used:
                    raise ValueError(
                        "conflicting_analysis_versions for the same session/trial"
                    )
                used.add(identity)
                outcome = trial["outcomes"].get(outcome_name)
                if outcome is not None:
                    identity_json = _canonical(_identity(outcome["contract"]))
                    if contract_json is not None and identity_json != contract_json:
                        raise ValueError("outcome_contract_mismatch")
                    contract_json = identity_json
                condition = trial["condition"]
                if condition not in {condition_a, condition_b}:
                    exclusions["other_condition"] += 1
                elif trial["state"] != "completed":
                    exclusions["not_completed"] += 1
                elif trial["quality"]["status"] != "usable":
                    exclusions["quality_not_usable"] += 1
                elif outcome is None or not _finite(outcome.get("value")):
                    exclusions["missing_or_nonfinite_outcome"] += 1
                else:
                    samples[participant][context["study_id"]][condition].append(
                        outcome["value"]
                    )
                    counts["included_trials"] += 1
        result = {
            "outcome_name": outcome_name,
            "condition_a": condition_a,
            "condition_b": condition_b,
            "contrast": "condition_b - condition_a",
            "contract": json.loads(contract_json) if contract_json else None,
            "counts": {
                **counts,
                "studies": len(study_set),
                "participant_groups": len(samples),
                "participants": len(samples)
                if len(study_set) <= 1 or not missing_links
                else None,
            },
            "exclusions": dict(exclusions),
            "participants": [],
            "mean_paired_difference": None,
            "interpretation": (
                "Descriptive participant-level paired differences; "
                "no population inference or automatic parameter promotion."
            ),
            "generalization_independence": "declared_linkage_checked"
            if not missing_links
            else "unresolved",
        }
        if len(study_set) > 1 and missing_links:
            return {
                **result,
                "status": "not_comparable",
                "reason": "missing_participant_linkage",
                "missing_links": sorted(set(missing_links)),
            }
        unpaired_studies = 0
        paired_trials = 0
        for participant, by_study in sorted(samples.items()):
            strata = []
            for study, values in sorted(by_study.items()):
                if not all(values.get(c) for c in (condition_a, condition_b)):
                    unpaired_studies += 1
                    continue
                a = math.fsum(v / len(values[condition_a]) for v in values[condition_a])
                b = math.fsum(v / len(values[condition_b]) for v in values[condition_b])
                if not _finite(b - a):
                    raise ValueError(
                        "participant contrast is outside the finite numeric range"
                    )
                strata.append(
                    {
                        "study_id": study,
                        "condition_a_mean": a,
                        "condition_b_mean": b,
                        "difference": b - a,
                        "trials_a": len(values[condition_a]),
                        "trials_b": len(values[condition_b]),
                    }
                )
            if not strata:
                continue
            a = math.fsum(s["condition_a_mean"] / len(strata) for s in strata)
            b = math.fsum(s["condition_b_mean"] / len(strata) for s in strata)
            difference = math.fsum(s["difference"] / len(strata) for s in strata)
            n_a, n_b = (
                sum(s["trials_a"] for s in strata),
                sum(s["trials_b"] for s in strata),
            )
            paired_trials += n_a + n_b
            result["participants"].append(
                {
                    "participant_key": participant,
                    "study_ids": [s["study_id"] for s in strata],
                    "study_contrasts": strata,
                    "condition_a_mean": a,
                    "condition_b_mean": b,
                    "difference": difference,
                    "trials_a": n_a,
                    "trials_b": n_b,
                }
            )
        paired = result["participants"]
        result["counts"].update(
            paired_participants=len(paired),
            unpaired_participants=len(samples) - len(paired),
            unpaired_participant_studies=unpaired_studies,
            paired_trials=paired_trials,
        )
        if paired:
            result["mean_paired_difference"] = math.fsum(
                p["difference"] / len(paired) for p in paired
            )
        result["status"] = "complete" if paired else "no_pairs"
        return result


def check_learning_eligibility(
    artifacts: list[dict[str, Any]],
    *,
    targets: list[str],
    model_inputs: list[str],
    fit_inputs: list[str],
    splits: dict[str, str],
    participant_keys: dict[str, str],
) -> dict[str, Any]:
    """Check declared lineage, hash aliases and grouped splits for future learning."""
    _canonical(
        {
            "artifacts": artifacts,
            "targets": targets,
            "model_inputs": model_inputs,
            "fit_inputs": fit_inputs,
            "splits": splits,
            "participant_keys": participant_keys,
        }
    )
    if (
        not isinstance(artifacts, list)
        or not isinstance(splits, dict)
        or not isinstance(participant_keys, dict)
    ):
        raise ValueError(
            "artifacts, splits and participant_keys require declared container types"
        )
    nodes, aliases = {}, defaultdict(set)
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("each artifact must be an object")
        name = _text(artifact.get("id"), "artifact id")
        if name in nodes:
            raise ValueError("duplicate artifact id")
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or not _SHA.fullmatch(digest):
            raise ValueError("artifact sha256 must have 64 hexadecimal characters")
        _text(artifact.get("kind"), "artifact kind")
        parents = artifact.get("parents", [])
        if (
            not isinstance(parents, list)
            or any(not isinstance(p, str) or not p for p in parents)
            or len(set(parents)) != len(parents)
        ):
            raise ValueError("artifact parents must be distinct identifiers")
        nodes[name] = {**artifact, "parents": parents, "sha256": digest.lower()}
        aliases[digest.lower()].add(name)
    for node in nodes.values():
        if set(node["parents"]) - nodes.keys():
            raise ValueError("artifact has missing parents")
    indegree = {name: len(node["parents"]) for name, node in nodes.items()}
    children = defaultdict(list)
    for name, node in nodes.items():
        for parent in node["parents"]:
            children[parent].append(name)
    pending, visited = [name for name, degree in indegree.items() if degree == 0], 0
    while pending:
        name = pending.pop()
        visited += 1
        for child in children[name]:
            indegree[child] -= 1
            if indegree[child] == 0:
                pending.append(child)
    if visited != len(nodes):
        raise ValueError("artifact lineage contains a cycle")
    for group in (targets, model_inputs, fit_inputs):
        if not isinstance(group, list) or any(
            not isinstance(name, str) or name not in nodes for name in group
        ):
            raise ValueError("selected artifacts must be declared identifiers")
    if any(
        name not in nodes or split not in {"train", "tune", "test"}
        for name, split in splits.items()
    ):
        raise ValueError("splits must map declared artifact IDs to train/tune/test")
    if set(participant_keys) - nodes.keys():
        raise ValueError("participant_keys contains undeclared artifacts")

    def ancestors(names: list[str]) -> set[str]:
        seen, pending = set(), list(names)
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            pending.extend(nodes[name]["parents"])
            pending.extend(aliases[nodes[name]["sha256"]] - seen)
        return seen

    def is_data(name: str) -> bool:
        infrastructure = {"software", "model", "configuration"}
        return any(
            nodes[alias]["kind"] not in infrastructure
            for alias in aliases[nodes[name]["sha256"]]
        )

    issues = []

    def issue(code: str, names) -> None:
        item = {"code": code, "artifacts": sorted(names)}
        if item not in issues:
            issues.append(item)

    if not targets:
        issue("no_targets", [])
    selected = set(targets + model_inputs + fit_inputs)
    if selected - splits.keys():
        issue("missing_split", selected - splits.keys())
    allowed = {"measured_force", "measured_torque", "emg_proxy"}
    for target in targets:
        group = aliases[nodes[target]["sha256"]]
        if any(nodes[name]["kind"] not in allowed for name in group):
            issue("target_is_not_empirical_measurement_or_declared_emg_proxy", group)
        if len({nodes[name]["kind"] for name in group}) > 1:
            issue("hash_alias_quantity_kind_conflict", group)
        proxies = {
            name for name in ancestors([target]) if nodes[name]["kind"] == "emg_proxy"
        }
        if proxies and nodes[target]["kind"] != "emg_proxy":
            issue("emg_proxy_cannot_become_measured_force_or_torque", proxies)
        estimated = {
            name
            for name in ancestors([target])
            if nodes[name]["kind"].startswith(("estimated", "latent"))
        }
        if estimated:
            issue("estimated_ancestor_cannot_become_empirical_truth", estimated)
    overlap = {
        name
        for name in ancestors(targets) & ancestors(model_inputs + fit_inputs)
        if is_data(name)
    }
    if overlap:
        issue("target_input_or_fit_lineage_overlap", overlap)
    lineage_splits, lineage_owners = defaultdict(set), defaultdict(set)
    for name, split in splits.items():
        for ancestor in ancestors([name]):
            if not is_data(ancestor):
                continue
            lineage_splits[ancestor].add(split)
            lineage_owners[ancestor].add(name)
    for ancestor, groups in lineage_splits.items():
        if len(groups) > 1:
            issue("aliases_or_derived_versions_cross_splits", lineage_owners[ancestor])
    relevant = ancestors(list(selected | splits.keys()))
    required_links = {name for name in relevant if is_data(name)}
    missing = {
        name
        for name in required_links
        if not isinstance(participant_keys.get(name), str)
        or not participant_keys[name].strip()
    }
    if missing:
        issue("incomplete_participant_linkage", missing)
    participant_splits, participant_artifacts = defaultdict(set), defaultdict(set)
    for name, split in splits.items():
        for ancestor in ancestors([name]):
            if not is_data(ancestor):
                continue
            key = participant_keys.get(ancestor)
            if isinstance(key, str) and key.strip():
                participant_splits[key].add(split)
                participant_artifacts[key].add(name)
    for key, groups in participant_splits.items():
        if len(groups) > 1:
            issue("participant_crosses_splits", participant_artifacts[key])
    for group in aliases.values():
        keys = {
            participant_keys[name]
            for name in group
            if isinstance(participant_keys.get(name), str)
            and participant_keys[name].strip()
        }
        if len(keys) > 1:
            issue("hash_alias_participant_identity_conflict", group)
    return {
        "eligible": not issues,
        "issues": issues,
        "artifact_count": len(nodes),
        "scope": "declared_lineage_and_pseudonymous_links_only",
        "generalization_independence": "declared_links_checked"
        if not issues
        else "unresolved",
        "target_kinds": {name: nodes[name]["kind"] for name in targets},
        "automatic_parameter_promotion": False,
        "limitations": (
            "Checks do not authenticate evidence or establish empirical "
            "independence. EMG remains a proxy."
        ),
    }
