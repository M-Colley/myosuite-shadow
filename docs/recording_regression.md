# Recording regression and independent measurements

Use this workflow to preserve a real recording as an external regression input.
Participant data stays in your controlled storage; the repository contains only
synthetic software tests. No real recording or independent human measurement has
been validated by those tests.

Start from the canonical `tracking.jsonl` recorded by a study session, the exact
tracking configuration, and its calibrated subject model. Quest packets must first
be decoded with their measured clock mapping; Sapiens2 image predictions must first
be reconstructed with calibrated cameras and stable person assignments. See
[tracking](tracking.md) and [study sessions](study_session.md).

Create a draft manifest next to the external files:

```json
{
  "schema_version": 1,
  "origin": "real",
  "recording_id": "study01_session01",
  "source_id": "quest01",
  "person_id": "pseudonymous_id",
  "source_versions": {
    "device": "operator_supplied_actual_model",
    "runtime": "operator_supplied_actual_version",
    "application": "operator_supplied_actual_commit"
  },
  "timing": "recorded_receive",
  "files": {
    "frames": {"path": "tracking.jsonl", "role": "recording"},
    "config": {"path": "tracking_config.json", "role": "configuration"},
    "model": {"path": "subject.mjb", "role": "configuration"}
  },
  "expected_quality": [
    {
      "event": "annotated occlusion",
      "rows": [120, 130],
      "reason": "required_landmarks",
      "min_count": 1,
      "max_count": 10
    }
  ],
  "expected_events": [
    {"event": "measured clock refresh", "label": "clock_mapping"}
  ]
}
```

`origin` is an explicit operator declaration (`real` or `synthetic`), not inferred
from filenames. Replace all identifiers/versions with actual values. The source
and person must match every canonical frame. For Sapiens2 include actual model,
checkpoint and upstream versions, not the JSON parser's reference revision.
Include camera calibration, timestamp sidecars, person assignments and the
reconstruction provenance manifest as additional hashed `files` artifacts.
Artifact IDs must be simple names; roles are `recording`, `configuration`,
`calibration`, `validation` or `baseline`. Prefer the compiled `.mjb` subject model;
freeze any additional XML/assets used by an XML model separately. Freeze also
records the compiled model digest, and audit recompiles and checks it so changes
in external XML/mesh assets cannot bypass the top-level file hash.

Run these commands from the project environment:

```powershell
python examples/recordings/check_recording.py freeze D:\study\draft.json D:\study\frozen.json
python examples/recordings/check_recording.py audit D:\study\frozen.json
python examples/recordings/check_recording.py run D:\study\frozen.json D:\study\regression01
```

Freeze resolves paths, hashes files with SHA256, and records installed software
provenance. It never copies participant data or overwrites a frozen manifest.
Audit checks hashes, model/config binding, explicit source/person identities,
clock presence, annotations, and optional baseline hashes before creating output.
Hashing establishes file identity; it does not establish authenticity or accuracy.

Run creates a new `session` with raw frames and quality decisions, plus
`regression.json`. Every accepted pose passes through the real IK/acquisition and
shadow pipeline. Inputs remain untouched. A report's `passed` is `null` when no
assertions/baseline were supplied, `false` for a failed assertion, or `true` when
the supplied software checks pass. The example command exits nonzero for failed
assertions. Independent measurement results are descriptive, not folded into an
arbitrary pass threshold.

`recorded_receive` reuses each frame's finite `receive_unix` as the acquisition
policy clock, retaining the original freshness/prediction/clock-uncertainty gates.
It does not recreate original CPU queuing delays or measure historical latency.
If receipt times were not recorded, choose `timing: "offline"` explicitly;
wall-clock freshness then cannot be checked. Neither mode opens a headset or
creates new capture timestamps. Session metadata identifies offline reprocessing.

## Annotated failures and registered changes

Quality `rows` are zero-based half-open raw-frame intervals, including rejected
frames. Select `accepted`, exact `reason`, and/or a truthy top-level frame metadata
`raw_field`, with declared count bounds. For example a missing camera should select
`raw_field: "missing_cameras"` as well as its rejection reason. A label such as
"occlusion" does not itself prove occlusion: retain the separate annotation source
as a hashed artifact. Pipeline events select their recorded `label` in
`expected_events`. These include `clock_mapping`, `registration` and `resumed`.

Recenter/skeleton revision changes still fail closed. To apply a genuinely measured
replacement registration, freeze its JSON and add an explicit replay control:

```json
{
  "controls": [
    {"before_row": 131, "kind": "registration", "file": "registration1"},
    {"before_row": 500, "kind": "resume"}
  ]
}
```

`registration1` must select a frozen artifact containing the existing Registration
record format. Controls occur before that raw row and preserve reset ordering.
`resume` closes, resumes, and resets the fit at that boundary. It exercises a
logical recovery boundary; it is not an invented hardware failure or a process
interruption test. Destructive-recovery fault tests remain in `tests/test_resume.py`.

## Reviewed software baselines and external loads

After reviewing an actual session, add `baseline_session: {"path": "session"}`
to the draft and freeze a new manifest. All session files used for replay are
hashed. Optionally add a `baseline_quality` artifact pointing to its original
`tracking_quality.jsonl`. Baselines require the same compiled subject model and
configuration. Quality decisions, accepted times, flags and reset boundaries
compare exactly; poses, forces, activations and fatigue use declared absolute
tolerances (`input_qpos`, `input_qfrc`, `activation`, `fatigue`, `torque`, `reserve`,
`contact`). Defaults are conservative numerical repeatability tolerances, not
scientific agreement thresholds. A self-generated baseline only tests repeatability.

The first reprocessing session drains each queued pose deterministically. When a
baseline exists, a separate `scheduled` session replays the new IK poses under its
original recorded skip/reset schedule and initial fatigue state. Physiology is
compared to that schedule, so original optimizer skips are not silently replaced
by solving every frame. Independent references use this scheduled session when
available. Original wall-clock timings are not replayed.

Loads default to the tracking configuration (including an explicitly known held
mass). To retain originally recorded time-varying external generalized forces,
declare `loads: "baseline_inputs"`. This uses the frozen baseline's `inputs.csv`
forces only at exactly matching accepted input times, validates each finite `nv`
vector before queuing it, and records its hash/time bounds. Missing accepted load
times fail the run; loads are not interpolated, inferred from outcomes, or added
again to configured held mass. Undeclared nonzero baseline loads without a
configured held mass block reprocessing. These journaled loads are part of the
model input and never count as an independent validation measurement.

## Independent force, dynamometer and EMG references

The standalone Python API is described in
[independent instrument measurements](independent_references.md).

Add each external instrument export to `files` and declare its measured units,
channel mapping, clock mapping, calibration identifier and calibration/validation
role in `references`. CSV channels select named columns; XDF channels select
numeric column indices and an unambiguous stream name via the existing XDF reader.
For example:

```json
{
  "references": [{
    "file": "dynamometer_holdout",
    "kind": "torque",
    "format": "csv",
    "role": "validation",
    "instrument": "actual_instrument_serial_and_calibration_id",
    "person_id": "pseudonymous_id",
    "calibration_file": "instrument_calibration",
    "clock": {
      "source_anchor_s": 50.0,
      "session_anchor_s": 0.0,
      "rate": 1.0,
      "uncertainty_s": 0.002,
      "clock_id": "measured_instrument_clock"
    },
    "channels": [{"name": "elbow", "column": "torque", "unit": "Nm",
                  "scale": 1.0, "offset": 0.0}],
    "max_gap_s": 0.05,
    "comparison": {"channel": "elbow", "prediction": "tau_0", "independent": true}
  }]
}
```

The selected artifact role must match `references.role`. `calibration_file`, when
present, selects a frozen calibration artifact. Scale/sign/offset come from the
instrument calibration, never a fit to held-out outcomes. Clock parameters must
come from measured synchronization; no automatic clock origin or lag is invented.
Use validity bounds on the existing ClockMapping when needed. Common session time
is required. Never label an input load reused by the model as independent.

CSV defaults to a `time` column. If a `role` column exists, only rows of the declared
role are used; excluded calibration rows create interpolation breaks. Use
`role_column` to name a different role column. Blank measurements remain missing.
Alignment uses existing `validation.sample_at`, never extrapolates or bridges
missing samples, excluded roles, or gaps exceeding `max_gap_s`.

Physical comparisons require `independent: true`, a validation role and matching
units: scalar hinge `tau_<dof>` in Nm, slide in N, or
`contact:<body>:<x|y|z>` in N. No force-to-torque lever arm or axis transform is
guessed. The report includes paired/excluded counts, bias, RMSE and existing
concordance; estimates marked missing, inferred, unsolved or poor quality are
excluded. Agreement thresholds and inference across participants remain a study
design decision.

For maximal dynamometer probes, use `kind: "mvc_probes"`, `independent: true`
and a frozen probe CSV with artifact role `validation`
with `time,joint,measured_torque,role,qpos_i`. The harness delegates to
`study.validate_probes`, which evaluates measured postures and excludes calibration
rows. Use the existing study calibration workflow for participant parameters;
this harness does not fit held-out results or replace participant calibration.

EMG uses `kind: "emg"`, units V/uV, measured `sampling_hz` on the session clock,
and `max_gap_s`. Existing envelope and median-frequency functions run separately
on finite uniform epochs; missing values, role breaks and timing deviations above
2% of the nominal sample interval split epochs. Short epochs retain missing
processed values. Outputs are saved as `<artifact_id>_emg.npz`. EMG remains a proxy:
the workflow does not infer force, MVC normalization, or measured model activation
from its amplitude. Original reference hashes and declared clock/calibration
metadata remain in the report.
