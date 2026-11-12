# Independent instrument measurements

`myosuite_shadow.references` reads explicitly calibrated CSV/XDF instrument
exports, maps their measured clock to session time, and uses existing validation
functions for alignment and agreement. It does not acquire hardware data or fit
model parameters to held-out outcomes. Repository examples/tests use synthetic
signals and do not establish agreement with real people.

```python
import hashlib
from pathlib import Path

from myosuite_shadow.log import read_session
from myosuite_shadow.references import load_reference, compare_reference

path = Path("D:/study/heldout_torque.csv")
# In a frozen study workflow, use the reviewed manifest's existing SHA256.
digest = hashlib.sha256(path.read_bytes()).hexdigest()
stream = load_reference(path, {
    "kind": "torque",
    "role": "validation",
    "instrument": "actual_instrument_serial_and_calibration_id",
    "clock": {
        "source_anchor_s": 50.0,
        "session_anchor_s": 0.0,
        "rate": 1.0,
        "uncertainty_s": 0.002,
        "clock_id": "measured_instrument_clock"
    },
    "channels": [{"name": "elbow", "column": "torque", "unit": "Nm",
                  "scale": 1.0, "offset": 0.0}]
}, expected_sha256=digest)
result = compare_reference(
    stream, read_session("D:/study/session"), channel="elbow",
    prediction="tau_0", max_gap_s=0.05, independent=True
)
```

Supply actual clock synchronization and instrument calibration values. The code
does not infer clock origin, lag, sign, lever arms, or instrument calibration from
validation outcomes. Hash checks establish file identity, not authenticity. Keep
the calibration and holdout assignments as study records.

CSV defaults to `time`, selecting named channel columns. A `role` column, if
present, must contain `calibration`/`validation`; only rows of the requested role
are selected. Removed roles create interpolation breaks. `time_column` and
`role_column` override their names. Blank values remain missing observations.
An explicitly named role column must exist; a typo cannot disable holdout filtering.
For XDF use `format: "xdf"`, a unique `stream` name and numeric channel `column`
indices. Both formats require the explicit existing ClockMapping fields and
preserve source hashes, clock mapping, calibration identifier and channel metadata.

`align_reference(stream, times, max_gap_s=...)` delegates to existing
`validation.sample_at`: it never extrapolates or bridges missing measurements,
excluded role intervals or gaps exceeding the declared limit.

`compare_reference` requires a validation role and `independent=True`. It compares
scalar `tau_<dof>` (hinge Nm, slide N) or `contact:<body>:<x|y|z>` (N), requires
matching physical units, excludes poor-quality model estimates, and reports paired
counts, bias, RMSE and existing concordance. It does not add population statistics
or arbitrary pass thresholds. A model input force reused as a reference is not an
independent measurement. For maximal dynamometer probes use the existing
`study.validate_probes` posture-matched workflow and separate calibration rows.

`process_emg(stream, sampling_hz=..., max_gap_s=...)` delegates to existing envelope
and median-frequency functions for each intact epoch. Missing data, role breaks
and time-step deviations exceeding 2% of the declared sampling interval split
epochs; short epochs retain missing processed outputs. Sampling rate is on the
mapped session clock. Supported EMG units are V/uV; no MVC normalization is inferred.
EMG amplitude is a proxy signal and is not labelled measured force or model
activation agreement.
