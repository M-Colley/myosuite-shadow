# Running a participant session

Install the frozen environment as described in [development.md](development.md).
The MyoSuite source commit is pinned; a separate model checkout is optional.
Keep source revisions, lockfile, model, acquisition configuration and protocol
fixed across the study. Record any explicit editable override and its local
changes. Hardware and human validation remain separate work.

## Prepare the subject model

Use a model scaled to participant segment geometry and muscle paths. Registration
changes coordinates, not body dimensions. `check_segment_lengths` in MyoSuite's
calibration module verifies measured anatomical landmark distances. Model sites
must represent the landmarks reported by the tracker.

Use participant/reference body mass ratios or measured represented-segment mass:

```python
from myosuite.biomechanics.calibration import scale_body_mass

# reference_mass is the nominal subject's documented whole-body mass.
model = scale_body_mass(model, participant_mass=71.5, reference_mass=75.0,
                        bodies=subject_body_names)
```

Exclude scenery/props from `subject_body_names`. Alternatively,
`represented_mass=...` sets those bodies' measured total. Do not set a partial arm
model to the participant's whole body mass. Mass updates preserve effective
muscle strength while refreshing derived constants.

Fit MVC strength using `strength_from_mvc` or `fit_strength_from_mvcs`, then scale
only the returned muscles. Use signed, gravity-corrected net torque including the
passive contribution, in N m for hinges or N for slides. Save measured postures,
instrument calibration and residuals. Fit fatigue on calibration holds and retain
the initial fatigue state if those holds have not been followed by recovery.
Save the final model and calibration records after all changes.
MVC helpers require force to be affine in activation and strength: force clamps,
multi-state actuators and artificial body gravity compensation are rejected.

## Configure and run

Run the [observability and protocol preflight](preflight.md) on the prepared
configuration and measured task postures before acquisition. Keep the report with
the session evidence.

The track command accepts a JSON object:

| Field | Meaning |
|---|---|
| `model` | Prepared `.xml` or `.mjb`, relative to config or absolute |
| `keypoints` | List of `{name, frame, frame_type}` mapping source to model |
| `registration` | Measured `Registration.record()`, targeting `model_world` |
| `ik` | `IKConfig`, including required landmarks and rejection thresholds |
| `acquisition` | `AcquisitionConfig`: inference, age, prediction, clock and load policy |
| `shadow` | `ShadowModelConfig`, including gaps and contacts |
| `metadata` | Participant code, condition and protocol/task revision |
| `calibration` | Strength, fatigue, anthropometry and measurement records |
| `quest` | Measured `clock`, expected `source_id` and `person_id` |

For continuous Quest recording, set `quest.clock_host` to the headset's IP.
`quest.clock_sync` accepts `QuestClockSynchronizer` options: `refresh_s`,
`session_epoch`, `count`, `interval_s`, `validity_s` and uncertainty limits. Keep
the same host session epoch. Refresh runs in the background and mappings are
retained with each source frame. A changed mapping resets differentiation so its
correction is not treated as body acceleration. Without a clock host, recording
is limited to the supplied mapping's validity interval.

Use the source names and clock/registration procedure in [tracking.md](tracking.md).
Review the anatomical mapping for every source/model version. All mapped landmarks
are required by default; `ik.required_keypoints` can define a reviewed subset.
Inferred inputs and positive prediction horizons are rejected by default.

Set IK speed limits for the actual task. `ik.max_joint_speed` bounds scalar
hinge/slide speeds and ball/free-joint angular speed (default 30 rad/s, or m/s
for slides). `ik.max_root_linear_speed` separately bounds floating-root
translation (default 5 m/s). Sudden coherent whole-body jumps can fit landmarks
well; they still require rejection before differentiation and muscle estimation.
Quality and solver thresholds must be finite, with positive speed/residual
limits.

```bash
myosuite-shadow-track --config study.json --quest-port 9763 --host 0.0.0.0 --log-dir logs/P07_B
```

`acquisition.held_mass_kg` plus `load_site` applies gravity through
`held_mass_qfrc` on every accepted pose. For moving objects, tools or MVC probes,
pass measured body/site forces through `external_qfrc`. Logging
`mark("set_start", load_kg=2)` alone does not apply a force.

Sapiens2 requires calibrated metric reconstruction before the same gate:

```bash
myosuite-shadow-track --config camera_study.json --frames camera_3d.jsonl --log-dir logs/P07_camera
```

Offline raw replay disables wall-clock age checks and drains the worker after
each frame for deterministic computation. It retains source timestamps and records
the offline policy. Accepted-pose `replay --session` retains the live decisions.
Use a separate output session when reprocessing its `tracking.jsonl`; the CLI
rejects reading and appending the same raw file, including filesystem aliases.

## Monitor and recover

Display `live.health` and `TrackedSession.health`: stale data, failed worker,
age, queue state, acceptance counts and drops. Check age when displaying
`live.latest`. Inspect `tracking.jsonl` and `tracking_quality.jsonl` for rejections.

Reference-space/skeleton changes require renewed registration using
`TrackedSession.set_registration`, which resets IK and differentiation priors.
The CLI cannot measure a new spatial calibration: stop, recalibrate and resume
with a clock mapping continuing the original session axis and including downtime.

```bash
myosuite-shadow-track --config renewed_study.json --quest-port 9763 --host 0.0.0.0 --log-dir logs/P07_B --resume
myosuite-shadow report --session logs/P07_B --thresholds protocol_thresholds.json --json logs/P07_B/quality.json
myosuite-shadow sensitivity --session logs/P07_B --out logs/P07_B/sensitivity
```

Resume preserves model, settings and fatigue and checks configuration before
changing logs. Legacy logs lacking their original schedule are marked as
recomputations. Sensitivity holds nominal optimizer weights fixed and returns
missing values outside coverage. Scenario bands are not confidence intervals.

Raw frames flush separately from model checkpoints and may include additional
frames after a crash. The accepted pose/decision journal determines live state;
retain both sets of evidence.

Resume keeps `.resume-recovery.zip` until replay and the recovered checkpoint
commit. If recovery is interrupted, retry `--resume`: the journal restores the
original pending inputs before replay. Reporting and sensitivity refuse a session
with pending recovery. Resume configuration is checked against the original
record before any restoration changes the session.

If checkpoint writing, file closure or a sink fails during `close()`, correct the
I/O problem and call `close()` again. The model rejects new inputs once closing
starts and continues remaining finalization without duplicating the live summary.

After reviewing a recording, use the [recording regression workflow](recording_regression.md)
to freeze its model, configuration, raw frames and annotations. Include synchronized
held-out [instrument references](independent_references.md) when available; model
input loads and repeatability baselines are not independent measurements.
