"""Protocol checks exercise real Jacobians without opening acquisition outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from myosuite_shadow.preflight import main, preflight
from myosuite_shadow.tracking import ClockMapping

_ARM = """<mujoco><compiler angle="radian"/><worldbody>
<body name="arm"><joint name="bend" axis="0 1 0"/>
<geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/>
<body name="forearm" pos=".3 0 0"><joint name="twist" axis="1 0 0"/>
<geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/>
<site name="axis"/><site name="tip" pos=".3 0 0"/>
<site name="radial" pos=".3 .05 0"/></body></body>
</worldbody></mujoco>"""


@pytest.fixture
def configuration(tmp_path: Path) -> Path:
    (tmp_path / "arm.xml").write_text(_ARM, encoding="utf-8")
    value = {
        "model": "arm.xml",
        "keypoints": [
            {"name": n, "frame": n, "frame_type": "site"}
            for n in ("axis", "tip", "radial")
        ],
        "registration": {
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation_m": [0, 0, 0],
        },
        "shadow": {"window": 5, "fatigue": False},
        "calibration": {"geometry": {"origin": "synthetic test"}},
        "preflight": {
            "target_joints": ["bend", "twist"],
            "postures": [[0.3, 0.8]],
            "required_calibrations": ["geometry"],
        },
    }
    path = tmp_path / "study.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _update(path: Path, update) -> None:
    value = json.loads(path.read_text())
    update(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def _live_config(path: Path, mapping: ClockMapping) -> None:
    def update(value):
        for point, name in zip(
            value["keypoints"], ("left_elbow", "left_wrist", "left_index_tip")
        ):
            point["name"] = name
        value["registration"]["source_frame"] = "quest_stage"
        value["quest"] = {
            "clock": mapping.record(),
            "clock_host": "192.0.2.1",
            "joint_set": "upper_body",
            "source_id": "quest",
            "person_id": "P07",
        }

    _update(path, update)


def test_visible_task_scope_passes_without_creating_session_files(configuration):
    before = {p.name: p.read_bytes() for p in configuration.parent.iterdir()}
    report = preflight(configuration)
    assert report["status"] == "passed"
    assert report["observability"][0]["rank"] == 2
    assert "jacobian" not in report["observability"][0]
    assert before == {p.name: p.read_bytes() for p in configuration.parent.iterdir()}
    json.dumps(report, allow_nan=False)


def test_axial_rotation_cannot_pass_with_only_on_axis_landmarks(configuration):
    _update(configuration, lambda v: v["keypoints"].pop())
    report = preflight(configuration)
    assert report["status"] == "blocked"
    assert "twist" in report["observability"][0]["ambiguous_dofs"]
    assert any(c["code"] == "unobservable_target" for c in report["checks"])


def test_unspecified_scope_remains_diagnostic(configuration):
    _update(configuration, lambda v: v["keypoints"].pop())
    _update(configuration, lambda v: v["preflight"].pop("target_joints"))
    report = preflight(configuration)
    assert report["status"] == "needs_attention" and report["errors"] == 0


def test_explicit_fixed_assumption_and_optional_matrices_are_recorded(configuration):
    _update(configuration, lambda v: v["keypoints"].pop())
    _update(
        configuration,
        lambda v: v["preflight"].update(target_joints=["bend"], fixed_joints=["twist"]),
    )
    result = preflight(configuration, include_matrices=True)["observability"][0]
    assert result["status"] == "full_rank"
    assert result["fixed_dofs"][0]["joint"] == "twist"
    assert "jacobian" in result and "nullspace" in result


@pytest.mark.parametrize("damage", ["unknown", "target", "calibration", "load_site"])
def test_bad_protocol_choices_are_rejected(configuration, damage):
    if damage == "unknown":
        _update(configuration, lambda v: v["preflight"].update(misspelled=True))
    elif damage == "target":
        _update(
            configuration, lambda v: v["preflight"].update(target_joints=["absent"])
        )
    elif damage == "calibration":
        _update(configuration, lambda v: v.pop("calibration"))
        assert preflight(configuration)["status"] == "blocked"
        return
    else:
        _update(configuration, lambda v: v.update(acquisition={"load_site": "absent"}))
    with pytest.raises((ValueError, KeyError)):
        preflight(configuration)


def test_live_clock_expiry_and_probe_epoch_continuity(configuration, monkeypatch):
    monkeypatch.setattr("myosuite_shadow.preflight.now", lambda: 1050.0)
    mapping = ClockMapping(
        1,
        100,
        "quest",
        unix_anchor=1000,
        valid_from_source_s=0,
        valid_until_source_s=100,
    )
    _live_config(configuration, mapping)
    assert preflight(configuration, live=True)["status"] == "passed"
    refreshed = ClockMapping(
        60,
        159,
        "quest",
        unix_anchor=1059,
        valid_from_source_s=40,
        valid_until_source_s=120,
    )
    calls = []

    def sync(host, **kwargs):
        calls.append(host)
        return refreshed

    monkeypatch.setattr("myosuite_shadow.preflight.synchronize_quest_clock", sync)
    report = preflight(configuration, live=True, probe_clock=True)
    assert calls == ["192.0.2.1"] and report["source_probe"] == "clock_only"
    _update(
        configuration, lambda v: v["quest"]["clock"].update(valid_until_source_s=20)
    )
    assert preflight(configuration, live=True)["status"] == "blocked"
    refreshed = ClockMapping(60, 1159, "quest", unix_anchor=2059)
    with pytest.raises(ValueError, match="session time axis"):
        preflight(configuration, live=True, probe_clock=True)


@pytest.mark.parametrize("damage", ["missing_source", "names", "frame", "layout"])
def test_impossible_live_quest_configuration_cannot_pass(configuration, damage):
    mapping = ClockMapping(0, 0, "quest", unix_anchor=1000)
    _live_config(configuration, mapping)
    if damage == "missing_source":
        _update(configuration, lambda v: v.pop("quest"))
    elif damage == "names":
        _update(
            configuration, lambda v: v["keypoints"][0].update(name="not_a_quest_joint")
        )
    elif damage == "frame":
        _update(
            configuration,
            lambda v: v["registration"].update(source_frame="tracker_world"),
        )
    else:
        _update(configuration, lambda v: v["keypoints"][0].update(name="left_knee"))
    assert preflight(configuration, live=True)["status"] == "blocked"


@pytest.mark.parametrize(
    "options",
    [{"refresh_s": 100, "validity_s": 60}, {"refresh_s": None}, {"unknown": 1}],
)
def test_live_clock_schedule_is_validated_even_without_probing(configuration, options):
    _live_config(configuration, ClockMapping(0, 0, "quest", unix_anchor=1000))
    _update(configuration, lambda v: v["quest"].update(clock_sync=options))
    with pytest.raises((TypeError, ValueError)):
        preflight(configuration, live=True)
    with pytest.raises((TypeError, ValueError)):
        preflight(configuration, live=True, probe_clock=True)


def test_cli_writes_blocked_report_and_never_overwrites_input(configuration):
    _update(configuration, lambda v: v["keypoints"].pop())
    report = configuration.parent / "preflight.json"
    with pytest.raises(SystemExit) as error:
        main(["--config", str(configuration), "--out", str(report)])
    assert error.value.code == 1
    assert json.loads(report.read_text())["status"] == "blocked"
    original = configuration.read_bytes()
    with pytest.raises(FileExistsError):
        main(["--config", str(configuration), "--out", str(configuration)])
    assert configuration.read_bytes() == original
