"""Synthetic participant summaries and declared lineage checks."""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from myosuite_shadow.study_catalog import StudyCatalog, check_learning_eligibility


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def report(person="p1", study="s1", run=None, link=None, a=(0,), b=(2,)):
    contract = {
        "name": "effort",
        "selector": "tau_0",
        "semantic_id": "right_elbow_torque",
        "unit": "Nm",
        "coordinate_frame": "model_joint",
        "normalization": "none",
        "method_version": "v1",
        "statistic": "time_mean",
        "quantity_class": "required_joint_torque",
    }
    definition = {
        "outcomes": [contract],
        "conditions": {
            "a": {"load": {"value": 0, "unit": "N"}},
            "b": {"load": {"value": 2, "unit": "N"}},
        },
    }
    context = {
        "study_id": study,
        "protocol_version": "1",
        "definition_sha256": digest(definition),
        "participant_id": person,
        "run_id": run or person,
        "session_id": "session_" + person,
    }
    if link is not None:
        context["participant_link_key"] = link
    trials = []
    for condition, values in (("a", a), ("b", b)):
        for index, value in enumerate(values):
            trials.append(
                {
                    "trial_id": condition + str(index),
                    "condition": condition,
                    "state": "completed",
                    "quality": {"status": "usable"},
                    "outcomes": {
                        "effort": {
                            "value": value,
                            "contract": copy.deepcopy(contract),
                            "coverage": 1.0,
                        }
                    },
                }
            )
    return {
        "schema_version": 1,
        "context": context,
        "definition": definition,
        "trials": trials,
    }


def rehash(value):
    value["context"]["definition_sha256"] = digest(value["definition"])
    return value


def test_immutable_idempotent_insert_and_reopen(tmp_path):
    path, value = tmp_path / "catalog.sqlite", report()
    with StudyCatalog(path) as catalog:
        sha = catalog.add(value)
        assert sha == digest(value)
        assert catalog.add(json.loads(json.dumps(value))) == sha
        changed = copy.deepcopy(value)
        changed["trials"][0]["outcomes"]["effort"]["value"] = 10
        with pytest.raises(ValueError, match="immutable"):
            catalog.add(changed)
    with StudyCatalog(path) as reopened:
        assert reopened.list()[0]["report_sha256"] == sha


def test_concurrent_identical_inserts_are_idempotent(tmp_path):
    path = tmp_path / "catalog.sqlite"

    def add(_):
        with StudyCatalog(path) as catalog:
            return catalog.add(report())

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(add, range(2)))
    assert results[0] == results[1]


def test_other_sqlite_databases_are_not_modified(tmp_path):
    path = tmp_path / "instrument.sqlite"
    with sqlite3.connect(path) as database:
        database.execute("CREATE TABLE instrument_measurements (value REAL)")
        database.execute("INSERT INTO instrument_measurements VALUES (12.5)")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="not a supported study catalog"):
        StudyCatalog(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "invalid", [float("nan"), float("inf"), object(), {1: "nonstring key"}]
)
def test_strict_json_rejects_before_insertion(tmp_path, invalid):
    with StudyCatalog(tmp_path / "catalog.sqlite") as catalog:
        value = report()
        value["invalid"] = invalid
        with pytest.raises(ValueError):
            catalog.add(value)
        assert catalog.list() == []


@pytest.mark.parametrize("schema", [True, 0, 2, "1"])
def test_report_schema_is_exactly_one(tmp_path, schema):
    value = report()
    value["schema_version"] = schema
    with StudyCatalog(tmp_path / "catalog.sqlite") as catalog:
        with pytest.raises(ValueError, match="schema_version"):
            catalog.add(value)


def test_definition_hash_is_bound_to_payload(tmp_path):
    value = report()
    value["definition"]["conditions"]["b"]["load"]["value"] = 99
    with StudyCatalog(tmp_path / "catalog.sqlite") as catalog:
        with pytest.raises(ValueError, match="definition_sha256"):
            catalog.add(value)


def test_trial_averaging_does_not_weight_people_by_trial_count(tmp_path):
    p1 = report(a=(0, 0, 0), b=(2, 2, 2))
    for index, (state, quality, value) in enumerate(
        [
            ("aborted", "usable", 999),
            ("completed", "review", 999),
            ("completed", "usable", None),
        ]
    ):
        trial = copy.deepcopy(p1["trials"][0])
        trial.update(
            trial_id="bad" + str(index), state=state, quality={"status": quality}
        )
        trial["outcomes"]["effort"]["value"] = value
        p1["trials"].append(trial)
    with StudyCatalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.add(p1)
        catalog.add(report(person="p2", a=(0,), b=(10,)))
        result = catalog.compare("effort", "a", "b")
    assert result["mean_paired_difference"] == pytest.approx(6)
    assert [p["difference"] for p in result["participants"]] == [2, 10]
    assert result["counts"]["paired_participants"] == 2
    assert result["counts"]["included_trials"] == 8
    assert sum(result["exclusions"].values()) == 3


def test_multistudy_requires_explicit_participant_linkage(tmp_path):
    with StudyCatalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.add(report(study="s1"))
        catalog.add(report(study="s2"))
        result = catalog.compare("effort", "a", "b")
        assert result["status"] == "not_comparable"
        assert result["counts"]["participants"] is None
        assert result["mean_paired_difference"] is None
        assert (
            catalog.compare("effort", "a", "b", study_ids=["s1"])["status"]
            == "complete"
        )


def test_linked_person_across_studies_is_one_person(tmp_path):
    with StudyCatalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.add(report(study="s1", link="shared-person", b=(2,)))
        catalog.add(report(study="s2", person="alias2", link="shared-person", b=(6,)))
        result = catalog.compare("effort", "a", "b")
    assert result["counts"]["paired_participants"] == 1
    assert result["mean_paired_difference"] == pytest.approx(4)


@pytest.mark.parametrize(
    "field,value",
    [
        ("unit", "N"),
        ("statistic", "rms"),
        ("normalization", "body_mass"),
        ("quantity_class", "measured_torque"),
    ],
)
def test_incompatible_contracts_are_refused(tmp_path, field, value):
    other = report(person="p2")
    other["definition"]["outcomes"][0][field] = value
    for trial in other["trials"]:
        trial["outcomes"]["effort"]["contract"][field] = value
    with StudyCatalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.add(report())
        catalog.add(rehash(other))
        with pytest.raises(ValueError, match="outcome_contract_mismatch"):
            catalog.compare("effort", "a", "b")


def test_model_selectors_may_differ_when_semantics_match(tmp_path):
    other = report(person="p2")
    other["definition"]["outcomes"][0]["selector"] = "tau_9"
    with StudyCatalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.add(report())
        catalog.add(rehash(other))
        assert catalog.compare("effort", "a", "b")["counts"]["paired_participants"] == 2


def test_condition_definition_mismatch_is_refused(tmp_path):
    other = report(person="p2")
    other["definition"]["conditions"]["b"]["load"]["value"] = 3
    with StudyCatalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.add(report())
        catalog.add(rehash(other))
        with pytest.raises(ValueError, match="condition_definition_mismatch"):
            catalog.compare("effort", "a", "b")


def test_reprocessed_session_trial_cannot_be_counted_twice(tmp_path):
    with StudyCatalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.add(report())
        catalog.add(report(run="reprocessed"))
        with pytest.raises(ValueError, match="conflicting_analysis_versions"):
            catalog.compare("effort", "a", "b")


def test_linkage_is_consistent_across_runs_of_the_same_local_participant(tmp_path):
    first, second = report(run="r1"), report(run="r2", link="stable", b=(4,))
    second["context"]["session_id"] = "day2"
    with StudyCatalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.add(first)
        catalog.add(second)
        result = catalog.compare("effort", "a", "b")
    assert result["counts"]["paired_participants"] == 1
    assert result["mean_paired_difference"] == pytest.approx(3)


def test_conflicting_linkage_is_rejected(tmp_path):
    first, second = report(run="r1", link="one"), report(run="r2", link="two")
    second["context"]["session_id"] = "day2"
    with StudyCatalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.add(first)
        catalog.add(second)
        with pytest.raises(ValueError, match="conflicting_participant_linkage"):
            catalog.compare("effort", "a", "b")


def test_study_baselines_and_unequal_trial_counts_cannot_bias_contrasts(tmp_path):
    with StudyCatalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.add(report(study="s1", link="person", a=(0, 0, 0), b=(2,)))
        catalog.add(report(study="s2", link="person", a=(10,), b=(12, 12, 12)))
        result = catalog.compare("effort", "a", "b")
    assert result["mean_paired_difference"] == pytest.approx(2)
    assert len(result["participants"][0]["study_contrasts"]) == 2


def test_conditions_from_different_studies_do_not_fabricate_a_pair(tmp_path):
    with StudyCatalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.add(report(study="s1", link="person", a=(0,), b=()))
        catalog.add(report(study="s2", link="person", a=(), b=(2,)))
        result = catalog.compare("effort", "a", "b")
    assert result["status"] == "no_pairs"
    assert result["counts"]["unpaired_participant_studies"] == 2


def learning():
    names = ["t1", "t2", "i1", "i2"]
    artifacts = [
        {
            "id": n,
            "sha256": hashlib.sha256(n.encode()).hexdigest(),
            "parents": [],
            "kind": "measured_torque" if n.startswith("t") else "input",
        }
        for n in names
    ]
    args = {
        "targets": ["t1", "t2"],
        "model_inputs": ["i1", "i2"],
        "fit_inputs": [],
        "splits": {"t1": "train", "i1": "train", "t2": "test", "i2": "test"},
        "participant_keys": {"t1": "p1", "i1": "p1", "t2": "p2", "i2": "p2"},
    }
    return artifacts, args


def codes(result):
    return {issue["code"] for issue in result["issues"]}


def test_declared_disjoint_participants_and_lineage_are_eligible():
    artifacts, args = learning()
    result = check_learning_eligibility(artifacts, **args)
    assert result["eligible"]
    assert not result["automatic_parameter_promotion"]


@pytest.mark.parametrize("kind", ["estimated", "latent"])
def test_estimates_and_latent_states_are_not_empirical_targets(kind):
    artifacts, args = learning()
    artifacts[0]["kind"] = kind
    assert not check_learning_eligibility(artifacts, **args)["eligible"]


def test_alias_and_fit_lineage_overlap_are_rejected():
    artifacts, args = learning()
    artifacts[2]["sha256"] = artifacts[0]["sha256"]
    assert "target_input_or_fit_lineage_overlap" in codes(
        check_learning_eligibility(artifacts, **args)
    )
    artifacts, args = learning()
    artifacts[2]["parents"] = ["t1"]
    args["fit_inputs"] = ["i1"]
    assert "target_input_or_fit_lineage_overlap" in codes(
        check_learning_eligibility(artifacts, **args)
    )


def test_derived_versions_and_participants_cannot_cross_splits():
    artifacts, args = learning()
    artifacts.append(
        {"id": "raw", "sha256": "f" * 64, "kind": "measured_torque", "parents": []}
    )
    artifacts[0]["parents"], artifacts[1]["parents"] = ["raw"], ["raw"]
    args["participant_keys"]["raw"] = "p1"
    result = check_learning_eligibility(artifacts, **args)
    assert "aliases_or_derived_versions_cross_splits" in codes(result)
    assert "participant_crosses_splits" in codes(result)


def test_missing_participant_linkage_fails_closed():
    artifacts, args = learning()
    del args["participant_keys"]["t2"]
    result = check_learning_eligibility(artifacts, **args)
    assert not result["eligible"]
    assert result["generalization_independence"] == "unresolved"


def test_estimated_ancestor_cannot_be_relabelled_as_measurement():
    artifacts, args = learning()
    artifacts.append(
        {"id": "teacher", "sha256": "f" * 64, "kind": "estimated", "parents": []}
    )
    artifacts[0]["parents"] = ["teacher"]
    args["participant_keys"]["teacher"] = "p1"
    assert "estimated_ancestor_cannot_become_empirical_truth" in codes(
        check_learning_eligibility(artifacts, **args)
    )


def test_emg_hash_alias_cannot_be_relabelled_as_torque():
    artifacts, args = learning()
    artifacts.append({**artifacts[0], "id": "emg_alias", "kind": "emg_proxy"})
    args["participant_keys"]["emg_alias"] = "p1"
    result = check_learning_eligibility(artifacts, **args)
    assert "hash_alias_quantity_kind_conflict" in codes(result)
    assert not result["eligible"]


def test_shared_software_is_not_participant_data_leakage():
    artifacts, args = learning()
    artifacts.append(
        {"id": "software", "sha256": "f" * 64, "kind": "software", "parents": []}
    )
    for node in artifacts[:-1]:
        node["parents"] = ["software"]
    assert check_learning_eligibility(artifacts, **args)["eligible"]


def test_infrastructure_ancestors_do_not_hide_data_leakage():
    artifacts, args = learning()
    artifacts.append(
        {"id": "model", "sha256": "f" * 64, "kind": "model", "parents": ["t1"]}
    )
    artifacts[2]["parents"] = ["model"]
    assert "target_input_or_fit_lineage_overlap" in codes(
        check_learning_eligibility(artifacts, **args)
    )


@pytest.mark.parametrize("problem", ["missing_parent", "cycle", "bad_sha", "bad_json"])
def test_invalid_declared_graph_is_rejected(problem):
    artifacts, args = learning()
    if problem == "missing_parent":
        artifacts[0]["parents"] = ["missing"]
    elif problem == "cycle":
        artifacts[0]["parents"], artifacts[2]["parents"] = ["i1"], ["t1"]
    elif problem == "bad_sha":
        artifacts[0]["sha256"] = "invalid"
    else:
        artifacts[0]["metadata"] = float("nan")
    with pytest.raises(ValueError):
        check_learning_eligibility(artifacts, **args)
