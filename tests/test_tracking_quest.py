"""Quest schemas preserve version/capability, tracking and timestamp semantics."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from myosuite_shadow.tracking import ClockMapping, QuestDecoder, fit_clock_mapping


def packet(joint_set="upper_body", device="Quest 3", sdk="84.0") -> dict:
    return dict(
        schema_version=1,
        kind="quest_body",
        source_id="quest",
        person_id="P07",
        sequence=1,
        clock_id="android_monotonic",
        time_s=100.0,
        time_kind="estimate",
        prediction_s=0,
        base_space="stage",
        space_revision=0,
        skeleton_revision=4,
        joint_set=joint_set,
        capabilities={"body_tracking": True, "full_body": joint_set == "full_body"},
        actual_fidelity="low",
        is_active=True,
        confidence=0.8,
        device=device,
        sdk_version=sdk,
        joints=[
            dict(
                id=i,
                position=[0, 1, 0.2],
                orientation=[0, 0, 0, 1],
                location_flags=15 if i == 15 else 3,
            )
            for i in range(84 if joint_set == "full_body" else 70)
        ],
    )


def decoder() -> QuestDecoder:
    return QuestDecoder(
        ClockMapping(100, 2, "android_monotonic", unix_anchor=1000),
        expected_person_id="P07",
    )


def test_declared_joint_layout_must_match_actual_packet():
    value = QuestDecoder(
        ClockMapping(100, 2, "android_monotonic"), expected_joint_set="upper_body"
    )
    assert value.decode(packet()).metadata["joint_set"] == "upper_body"
    with pytest.raises(ValueError, match="unexpected Quest joint set"):
        value.decode(packet("full_body"))
    with pytest.raises(ValueError, match="expected_joint_set"):
        QuestDecoder(
            ClockMapping(100, 2, "android_monotonic"), expected_joint_set="other"
        )


@pytest.mark.parametrize(
    "device,sdk,joint_set",
    [
        ("Quest 2", "59", "upper_body"),
        ("Quest Pro", "71", "upper_body"),
        ("Quest 3", "84", "full_body"),
        ("Quest 3S", "85", "full_body"),
        ("Future Quest", "next", "full_body"),
    ],
)
def test_quest_versions_use_capabilities_not_device_guesses(device, sdk, joint_set):
    value = decoder().decode(packet(joint_set, device, sdk), receive_unix=1001)
    assert value.time_s == 2 and value.capture_unix is None
    assert value.evaluation_unix == 1000
    assert value.position_tracked["right_shoulder"]
    assert "right_elbow" in value.inferred_joints
    assert value.metadata["actual_fidelity"] == "low"
    assert value.metadata["raw_packet"]["sdk_version"] == sdk
    assert value.skeleton_revision == 4
    assert ("left_knee" in value.positions) == (joint_set == "full_body")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(schema_version=2),
        lambda p: p.update(clock_id="unknown"),
        lambda p: p.update(person_id="P08"),
        lambda p: p.update(joint_set="future"),
        lambda p: p["joints"].pop(),
        lambda p: p["joints"][0].update(id=1),
        lambda p: p["joints"][0].update(location_flags=8),
        lambda p: p["joints"][0].update(position=[1, 2]),
        lambda p: p.update(confidence=float("nan")),
        lambda p: p.update(capabilities={"body_tracking": False}),
    ],
)
def test_quest_rejects_malformed_ambiguous_or_incompatible_packets(mutate):
    p = packet()
    mutate(p)
    with pytest.raises((ValueError, KeyError)):
        decoder().decode(p)


def test_predictions_never_claim_sensor_capture_time_and_inactive_is_empty():
    p = packet()
    p.update(time_kind="predicted", prediction_s=0.02)
    assert decoder().decode(p).capture_unix is None
    stale = decoder().decode(p, receive_unix=2000)
    assert stale.evaluation_unix == 1000 and stale.receive_unix == 2000
    p.update(time_kind="capture", prediction_s=0)
    assert decoder().decode(p).capture_unix == 1000
    p.update(is_active=False)
    assert decoder().decode(p).positions == {}
    p = packet("full_body")
    p["capabilities"]["full_body"] = False
    with pytest.raises(ValueError, match="capabilities"):
        decoder().decode(p)


def test_ping_clock_fit_retains_asymmetry_and_rejects_drift_or_rtt():
    # Source clock is 500 s ahead and runs at 100 ppm below the host.
    exchanges = [
        (500 + i, 10 + i * 1.0001 - 0.003, 10 + i * 1.0001 + 0.007) for i in range(5)
    ]
    fit = fit_clock_mapping(
        exchanges,
        clock_id="android_monotonic",
        host_unix_anchor=(10, 1000),
        validity_s=10,
    )
    # The midpoint has a 2 ms asymmetric-delay bias; true time stays in the bound.
    estimated, _ = fit.map(503)
    assert abs(estimated - 13.0003) <= fit.uncertainty_s
    assert fit.rate == pytest.approx(1.0001)
    slow = copy.deepcopy(exchanges)
    slow[0] = (500, 9, 11)
    assert fit_clock_mapping(slow, clock_id="x", validity_s=10).rate == pytest.approx(
        fit.rate
    )
    with pytest.raises(ValueError, match="three low-RTT"):
        fit_clock_mapping([(i, i, i + 0.5) for i in range(5)], clock_id="x")
    with pytest.raises(ValueError, match="drift"):
        fit_clock_mapping([(i, 2 * i, 2 * i + 0.01) for i in range(5)], clock_id="x")


def test_clock_bound_covers_all_feasible_affine_rates_at_validity_end():
    exchanges = [(i, i - 0.002, i + 0.002) for i in range(3)]
    with pytest.raises(ValueError, match="uncertainty"):
        fit_clock_mapping(exchanges, clock_id="x", validity_s=60)
    fit = fit_clock_mapping(
        exchanges, clock_id="x", validity_s=60, max_uncertainty_s=0.1
    )
    # This 1000 ppm clock is consistent with all three original RTT intervals.
    for source in np.linspace(0, 62, 101):
        assert abs(1.001 * source - fit.map(source)[0]) <= fit.uncertainty_s
    with pytest.raises(ValueError, match="precedes"):
        fit.map(-1)


def test_decoder_snapshots_clock_provider_once_per_packet():
    calls = []

    def provider():
        calls.append(1)
        assert len(calls) == 1
        return ClockMapping(100, 2, "android_monotonic", unix_anchor=1000)

    value = QuestDecoder(ClockMapping(0, 0, "wrong"), clock_provider=provider)
    assert value.decode(packet()).evaluation_unix == 1000
