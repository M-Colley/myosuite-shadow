"""Synthetic instrument signals test alignment, never human measurement accuracy."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from myosuite_shadow.references import (
    ReferenceStream,
    align_reference,
    compare_reference,
    load_reference,
    process_emg,
)


def _spec(**changes):
    return {
        "kind": "torque",
        "role": "validation",
        "instrument": "SYNTHETIC_dynamometer_calibration_v1",
        "clock": {
            "source_anchor_s": 10,
            "session_anchor_s": 0,
            "clock_id": "synthetic_instrument",
            "uncertainty_s": 0.001,
        },
        "channels": [{"name": "elbow", "column": "raw", "unit": "Nm"}],
    } | changes


def _load(path, spec):
    return load_reference(
        path, spec, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_calibrated_csv_clock_holdout_gaps_and_units(tmp_path):
    path = tmp_path / "synthetic.csv"
    path.write_text(
        "time,raw,role\n10,1,validation\n10.1,2,calibration\n"
        "10.2,3,validation\n10.3,,validation\n10.4,5,validation\n"
    )
    spec = _spec(
        channels=[
            {"name": "elbow", "column": "raw", "unit": "Nm", "scale": -2, "offset": 1}
        ]
    )
    stream = _load(path, spec)
    np.testing.assert_allclose(stream.time_s, [0, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(
        stream.values[:, 0], [-1, -5, np.nan, -9], equal_nan=True
    )
    assert stream.provenance["excluded_role_rows"] == 1
    aligned = align_reference(stream, np.array([0, 0.1, 0.2, 0.3]), max_gap_s=0.3)
    assert np.isnan(aligned[1, 0])  # calibration row cannot be bridged
    assert np.isnan(aligned[3, 0])
    session = SimpleNamespace(
        time=stream.time_s.copy(),
        flags=np.zeros(4, dtype=int),
        columns={"tau_0": np.array([-1, -5, 0, -9])},
        record={"model": {"joints": [{"dofadr": 0, "type": 3}]}},
    )
    comparison = compare_reference(
        stream,
        session,
        channel="elbow",
        prediction="tau_0",
        max_gap_s=0.3,
        independent=True,
    )
    assert comparison["paired"] == 3 and comparison["rmse"] == 0
    with pytest.raises(ValueError, match="independent"):
        compare_reference(
            stream,
            session,
            channel="elbow",
            prediction="tau_0",
            max_gap_s=0.3,
            independent=False,
        )
    stream = _load(path, _spec(role="calibration"))
    with pytest.raises(ValueError, match="validation"):
        compare_reference(
            stream,
            session,
            channel="elbow",
            prediction="tau_0",
            max_gap_s=0.3,
            independent=True,
        )
    force = _load(
        path,
        _spec(kind="force", channels=[{"name": "elbow", "column": "raw", "unit": "N"}]),
    )
    with pytest.raises(ValueError, match="units"):
        compare_reference(
            force,
            session,
            channel="elbow",
            prediction="tau_0",
            max_gap_s=0.3,
            independent=True,
        )


def test_xdf_stream_adapter_requires_explicit_clock_and_channels(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.xdf"
    path.write_bytes(b"synthetic test stub, not an acquired XDF")
    monkeypatch.setattr(
        "myosuite_shadow.references.load_xdf",
        lambda _: {"loadcell": (np.array([10, 11]), np.array([[9, 1], [9, 2]]))},
    )
    spec = _spec(
        format="xdf",
        stream="loadcell",
        kind="force",
        channels=[{"name": "z", "column": 1, "unit": "N"}],
    )
    stream = _load(path, spec)
    np.testing.assert_array_equal(stream.values[:, 0], [1, 2])
    np.testing.assert_array_equal(stream.time_s, [0, 1])
    with pytest.raises(ValueError, match="hash"):
        load_reference(path, spec, expected_sha256="0" * 64)
    spec["channels"][0]["column"] = True
    with pytest.raises(ValueError, match="indices"):
        _load(path, spec)


def test_explicit_missing_role_column_cannot_mix_calibration_into_holdout(tmp_path):
    path = tmp_path / "synthetic.csv"
    path.write_text("time,raw,role\n10,10,calibration\n11,20,validation\n")
    with pytest.raises(ValueError, match="declared role column"):
        _load(path, _spec(role_column="misspelled_role"))
    stream = _load(path, _spec())
    assert stream.values.tolist() == [[20]]


def test_emg_processing_splits_missing_and_role_intervals():
    time = np.arange(2000) / 1000
    signal = np.sin(2 * np.pi * 80 * time)[:, None]
    signal[900:1000] = np.nan
    breaks = np.zeros(2000, bool)
    breaks[1500] = True
    stream = ReferenceStream(
        time,
        signal,
        ("biceps",),
        ("uV",),
        "emg",
        "validation",
        {"origin": "synthetic"},
        breaks,
    )
    envelope, (centres, frequency) = process_emg(
        stream, sampling_hz=1000, max_gap_s=0.002
    )
    assert np.isfinite(envelope.values[:900]).all()
    assert np.isnan(envelope.values[900:1000]).all()
    assert np.isfinite(envelope.values[1000:]).all()
    assert not np.any((centres > 0.9) & (centres < 1.0))
    np.testing.assert_allclose(frequency, 80, atol=3)
    with pytest.raises(ValueError, match="physical force"):
        compare_reference(
            stream,
            None,
            channel="biceps",
            prediction="tau_0",
            max_gap_s=0.002,
            independent=True,
        )
