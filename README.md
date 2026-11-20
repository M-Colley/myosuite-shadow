# myosuite-shadow

Live muscle activation and fatigue estimates beside an experiment, built on
[MyoSuite](https://github.com/MyoHub/myosuite).

Tracked poses drive joint kinematics, static optimization and 3CC-r fatigue.
The optional fatigue cap limits each muscle to `1 - MF`; other muscles and
reserves can take more of the required load. Session tools record source
observations, accepted poses, estimates, processing decisions and events.

**Status: alpha, with no participant validation supplied.** Outputs are model
estimates under a minimum-effort assumption. Start with
[limitations](docs/limitations.md) and the
[human validation protocol](docs/validation_study.md).

## Install

Version 0.3 pins the published MyoSuite source commit
[`87c1896b12b99683aecc83d6aab00cc66243a50f`](https://github.com/MyoHub/myosuite/commit/87c1896b12b99683aecc83d6aab00cc66243a50f).
This supplies the study-support APIs, corrected fatigue censoring and MVC model
checks. It is not an official MyoSuite release or evidence of participant
accuracy. From this repository, install the frozen study environment:

```bash
uv sync --frozen --extra all --python 3.12
```

No sibling checkout is required. `python -m pip install ".[all]"` also resolves
the pinned MyoSuite commit, but does not freeze every other dependency. Archive
the shadow revision, model, configuration and lockfile for the study. See
[development.md](docs/development.md) for tests, optional editable model
development, CI configuration and current verification limits.

## Run a study session

Prepare the participant model, measured spatial registration, source clock
mapping and study policy using [study_session.md](docs/study_session.md).
[tracking.md](docs/tracking.md) covers the native Quest sender, receiver and
Sapiens2 imports with calibrated metric reconstruction.

```bash
# Live Quest body packets; study.json contains the prepared model and calibrations.
myosuite-shadow-track --config study.json --quest-port 9763 --host 0.0.0.0 --log-dir logs/P07_B
# Reprocess canonical metric 3D observations, preserving source timing and quality.
myosuite-shadow-track --config camera_study.json --frames camera_3d.jsonl --log-dir logs/P07_camera
# Apply the study's archived warning thresholds, then compare calibration scenarios.
myosuite-shadow report --session logs/P07_B --thresholds protocol_thresholds.json --json logs/P07_B/quality.json
myosuite-shadow sensitivity --session logs/P07_B --out logs/P07_B/sensitivity
# Evaluate independent, posture-matched dynamometer probes.
myosuite-shadow-validate --session logs/P07_B --probes P07_probes.csv --out logs/P07_B/validation.json
```

The source adapters retain source/person identity, tracked/inferred flags and
clock evidence. `TrackedSession` requires the reviewed landmarks, rejects
inferred/predicted observations by default, and applies the IK quality policy
before a pose can affect muscle or fatigue state. Recentring or a skeleton
revision requires renewed registration and a differentiation reset. Hardware
and participant accuracy still need to be established for the intended task.

## Calibrate the model and include actual loads

Registration aligns coordinate systems. It does not scale segment lengths,
muscle paths or strength. Use a prepared subject model with anatomical landmark
sites and verify its geometry before calibration. For a partial arm model,
scale represented body masses by the documented participant/reference mass
ratio, or provide the measured total mass of those represented segments.
Never replace an arm's total mass with the participant's whole-body mass.

This fragment assumes participant geometry is already prepared and
`model_reference.json` records the reference subject's whole-body mass and the
anatomical bodies to scale, excluding scenery and props:

```python
import json
from pathlib import Path

import mujoco
import numpy as np
from myosuite.biomechanics.calibration import (
    posture_at, scale_body_mass, scale_strength, strength_from_mvc,
)

model = mujoco.MjModel.from_xml_path("study/P07_subject.xml")
reference = json.loads(Path("study/model_reference.json").read_text())
model = scale_body_mass(
    model, participant_mass=71.5,
    reference_mass=reference["whole_body_mass_kg"],
    bodies=reference["anatomical_bodies"],
)
q = posture_at(model, {"elbow_flexion_r": np.radians(90)})
# Signed, gravity-corrected net MVC torque; includes the passive contribution.
strength = strength_from_mvc(model, q, "elbow_flexion_r", measured_torque=62.0)
model = scale_strength(model, strength.scale, strength.agonists)
mujoco.mj_saveModel(model, "study/P07_calibrated.mjb", None)
# Archive strength.record(), the measurements and geometry/mass calibration.
```

One MVC scales its returned agonist group; it does not identify each muscle's
strength or calibrate unrelated muscles. Multi-posture strength fits and fatigue
hold fits are available in `myosuite.biomechanics.calibration`; the shadow
`calibration` module re-exports those study-facing interfaces.

A load marker alone does not change the physics. In the Python path, after
registered IK has accepted `pose` for `frame`, a stationary 2 kg held load needs
an actual generalized force on every pose:

```python
from myosuite.biomechanics.external_loads import held_mass_qfrc

external = held_mass_qfrc(model, pose.qpos, 2.0, site="held_object_com")
live.submit(
    frame.time_s, pose.qpos, external, flags=pose.flags,
    capture_unix=frame.capture_unix,
    evaluation_unix=frame.evaluation_unix, receive_unix=frame.receive_unix,
)
```

The prepared model must contain that site at the object's centre of mass.
`TrackedSession` applies the same helper through `held_mass_kg` and `load_site`.
This is gravity only: moving objects and MVC probes need measured wrenches or
an appropriate object model. Avoid double-counting mass already in the model.
See [study_session.md](docs/study_session.md) for the complete acquisition path,
calibration records, recovery assumptions and initial fatigue handling.

## Capabilities

| Need | Support |
|---|---|
| Tracker sources | Quest body-API packets and Sapiens2 308-point exports; camera observations require synchronized calibrated views or registered metric camera-Z depth |
| Participant preparation | Geometry checks, explicit mass scaling, selective MVC strength fitting and fatigue hold fitting in MyoSuite |
| Quality | Required landmarks, confidence/residual/speed gates, inference/prediction policy, registration revisions, per-pose and estimate flags |
| Timing | Separate receive, measured capture and mapped evaluation timestamps; `frame_unix` prefers capture, then evaluation, then receive; LSL uses that frame time |
| Live operation | Worker thread, newest-pose optimization under backlog, counted pose drops, health/age/error state; `drain()` supports deterministic offline processing |
| Gaps and fatigue | `gap_load="rest"` or `"hold"` across long gaps; explicit tracking resets; initial fatigue recorded with the processing schedule |
| Recovery and replay | Atomic durable checkpoints, validated input/decision journals, resume and shared `Session.replay` |
| Study analysis | Protocol thresholds, tracking exclusions, per-joint reserves, root residuals, latency, sensitivity scenarios and independent MVC probe reports |

## Model math and study orchestration

| MyoSuite | myosuite-shadow |
|---|---|
| `myosuite.biomechanics`: inverse dynamics, static optimization, streaming kinematics | `ShadowModel` and `LiveShadowModel` session orchestration |
| Model calibration, segment checks, external-force mapping and camera reconstruction | Tracker decoding, identity/clock/registration records, study quality gates and mink IK wiring |
| `myosuite.core.muscle_conditions.CumulativeFatigue`: 3CC-r state | Participant records, protocol events, checkpoints, LSL and analysis commands |

The matching model interfaces are in the pinned MyoSuite source commit. Optional
co-development is described in [development.md](docs/development.md). Static
optimization and streaming imports now use `myosuite.biomechanics`.

## Accepted-pose replay and recovery

```bash
myosuite-shadow replay --log-dir logs/demo
myosuite-shadow replay --model subject.xml --log-dir logs/subject
myosuite-shadow replay --env myoMimicFullbody-v0 --clip walk.npz --contacts --log-dir logs/walk
myosuite-shadow replay --session logs/P07_B --window 21 --log-dir logs/P07_B_w21
```

`Session.replay(shadow)` and the CLI retain accepted inputs, timestamps,
optimization decisions, explicit resets and initial fatigue. Reproducing an
unchanged schedule requires the same model, settings, environment and solver
state; cross-platform numerical identity is not guaranteed. Changed windows
or calibration intentionally recompute estimates. Raw-frame replay through
`myosuite-shadow-track --frames` reruns registration/IK/gates and drains each
frame, so its processing schedule can differ from the original live run.

`ShadowModel.resume("logs/P07_B")` restores the checkpoint and replays complete
pending inputs with their decisions. It does not save the optimizer's warm
start, so resumed outputs are compared to uninterrupted outputs within solver
tolerance. The next pose must continue the original session clock, including
downtime; long downtime is treated under the configured gap assumption.

Version 0.2 logs remain readable and resumable. Their unrecorded live decisions
and resets cannot be reconstructed: fallback is a marked recomputation, not
exact recovery of the original live schedule. A corrupt or missing versioned
0.3 decision journal is rejected before recovery changes the files.

## Session evidence

| File | Content |
|---|---|
| `record.json` | Source commits/dirty state, dependency versions, model hash/layout, resolved optimizer settings, calibration, metadata, timestamp semantics and resumes |
| `tracking.jsonl` | Canonical source observations and source evidence, recorded before acceptance/IK |
| `tracking_quality.jsonl` | Accepted/rejected decisions, reasons, IK residuals and inference information |
| `inputs.csv` | Accepted poses, source/receive timestamps, flags and known generalized loads; optional evaluation time is appended after the original columns |
| `operations.jsonl` | Optimization decisions, explicit fit resets and initial fatigue; validated for ordering and complete input coverage |
| `estimates.csv` | Frame/completion times, compute cost, status, flags, gaps/skips, activations, fatigue, contacts, per-DOF reserves and required torque |
| `events.csv` | Experimenter and acquisition events with wall/session time and JSON data |
| `checkpoint.json` | Pipeline state and durable file offsets; `closed` marks a clean close |
| `model.mjb` | Compiled subject model, omitted only with `save_model=False` |

Raw tracking files exist when acquisition recording is enabled and flush
separately from model checkpoints. `read_session` reads model-session arrays;
`Session.load_model` checks the saved model hash. The report combines this
record with tracking-quality evidence when present. Sensitivity output marks
unobserved endpoints and gaps as missing; scenario bounds are not confidence
intervals. See [validation_study.md](docs/validation_study.md) for the evidence
needed to support claims about people.

Run [study preflight](docs/preflight.md) before acquisition. Preserve reviewed
recordings with the [recording regression workflow](docs/recording_regression.md)
and compare held-out force, dynamometer or EMG exports using
[independent references](docs/independent_references.md). These tools retain
provenance and missing samples; study accuracy requires actual measurements.

## Historical benchmarks (0.2)


These are historical 0.2 measurements using MyoSuite snapshot `13b70270`,
Windows, Python 3.10, MuJoCo 3.17 and OSQP 1.1.3. They are reference data,
not performance or accuracy claims for 0.3. The current optimizer uses
different model-scaled reserve defaults. Rerun both scripts in the archived
study environment before choosing settings.

### Delay against accuracy

`python benchmarks/delay_accuracy.py`: myoArm minimum-jerk reaches (0.5 to
0.9 s, 0.3 s holds) tracked at 90 Hz. The reference is static optimization on
the noise-free kinematics of the 1.8 kHz motion. Activation RMS over all 50
muscles and frames:

| Delay | Window | 0.5 deg noise | 1.0 deg noise |
|---|---|---|---|
| 0 ms | 7 | 0.40 | 0.47 |
| 0 ms | 11 | 0.32 | 0.38 |
| 33 ms | 7 | 0.25 | 0.33 |
| 56 ms | 11 | 0.12 | 0.19 |
| 78 ms | 15 (default) | 0.074 | 0.12 |
| 111 ms | 21 | 0.050 | 0.073 |
| 167 ms | 31 | 0.055 | 0.061 |
| 0 ms, `qacc = 0` | 7 | 0.16 | 0.19 |

Noise in the accelerations dominates the error, so use the longest delay the
study can tolerate. If there can be no delay at all, dropping the
accelerations (quasi-static) works better than a short fit. One update costs
about 1 ms on myoArm. The script also prints the torque errors.

### Fatigue

`python benchmarks/fatigue_hold.py`: MyoSuite's elbow (six muscles) holds the
forearm at 69 deg for two minutes, estimated at 20 Hz:

| Load | Cap | Long biceps, start / 120 s | Brachialis, start / 120 s | Peak MF | Max excess over `1 - MF` | Reserve at 120 s |
|---|---|---|---|---|---|---|
| 20 N m | yes | 0.51 / 0.31 | 0.33 / 0.42 | 0.69 | 0.00 | 1.8 N m |
| 20 N m | no | 0.51 / 0.51 | 0.33 / 0.33 | 0.69 | 0.19 | 0 |
| 40 N m | yes | 0.92 / 0.28 | 0.60 / 0.29 | 0.72 | 0.00 | 27.4 N m |

With the cap, the fatigued long head drops to what it has left and the
brachialis and short head take over. At 40 N m every flexor saturates and the
reserve grows: in model terms the load cannot be held any longer. Without the
cap, muscles exceed their remaining capacity by up to 0.19 (0.64 at 40 N m).

### Full body

On a retargeted KIT walk (MyoFullBody at 100 Hz, with ground reactions; the
clip is not included here) an update takes 15 ms (median), with a p99 of 56 ms.
With `LiveShadowModel` the kinematics see every pose and the optimization
should run at about 60 Hz on the newest one (from the update time; not yet
measured live). Ground reactions are only
estimated against world-fixed bodies (floor, chair, table).

## Citing

See [`CITATION.cff`](CITATION.cff). Please also cite MyoSuite (Caggiano et al.,
L4DC 2022), the 3CC-r fatigue model (Looft, Herkert and Frey-Law, J. Biomech.
2018), static optimization (Crowninshield and Brand, J. Biomech. 1981;
Anderson and Pandy, J. Biomech. 2001), and mink and LabStreamingLayer if you
use the IK or LSL parts. Source-specific references are linked in
[tracking.md](docs/tracking.md).

## License

Apache 2.0, like MyoSuite. External tracker/model dependencies retain their own
licenses and terms.
