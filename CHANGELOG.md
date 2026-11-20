# Changelog

## 0.3.0 (unreleased)

- **Study preflight.** Checks required landmark observability on declared task
  DOFs and postures, allowing free nuisance motion unless explicitly fixed.
  Reports model/configuration hashes, resolved settings, registration, declared
  calibrations and optional measured Quest clock continuity before acquisition.

Pins the published MyoSuite source commit
[`87c1896b12b99683aecc83d6aab00cc66243a50f`](https://github.com/MyoHub/myosuite/commit/87c1896b12b99683aecc83d6aab00cc66243a50f),
with optional editable development described in [development.md](docs/development.md).
Reusable model calculations live in `myosuite.biomechanics`; shadow owns
acquisition and study orchestration. This remains alpha software, not an official
MyoSuite release or a completed participant validation.

- **MyoSuite calibration fixes.** The source pin includes corrected endurance
  censoring at `max_time` and rejects stateless muscle control clamps that would
  make capped MVC torque predictions incorrect.
- **Participant acquisition.** Canonical metric 3D frames retain source/person
  identity, tracked/inferred states, source revisions and clock evidence. Quest
  body-API decoding, native sender/clock examples and Sapiens2 308-point imports
  feed measured registration and calibrated camera reconstruction. Hardware
  capture and participant validity remain to be established.
- **Acceptance before fatigue.** `TrackedSession` records raw frames and
  decisions, requires reviewed landmarks, applies inference/prediction/timing
  policy and gates IK quality. Registration/skeleton changes require renewed
  registration and reset the fit. Rejected IK solutions do not become priors.
  Speed checks include floating-root translation/rotation and ball joints;
  nonfinite thresholds are rejected. Invalid pose clocks do not advance
  acquisition ordering, allowing valid subsequent frames to recover.
- **Calibration and actual loads.** Shadow re-exports MyoSuite's calibration
  interfaces: selective MVC strength scaling, multi-posture fits, geometry
  checks, explicit represented/reference mass semantics and fatigue hold fits.
  External-load helpers map measured forces or held-object gravity into joint
  forces; a load marker alone does not apply a physical load.
  Steady MVC probes handle early activation updates and reject force clamps,
  multi-state actuators and gravity compensation. Fatigue fitting initializes
  finite failure predictions when the starting fatigue rate is too slow.
- **Timestamp meanings.** `capture_unix` remains a measured capture timestamp.
  Separate `evaluation_unix` describes mapped tracked/predicted state time;
  receive time remains distinct. Frame/LSL timestamps prefer capture, then
  evaluation, then receive. New input columns are appended to preserve the old
  input prefix.
  Quest clock exchanges bound offset/rate uncertainty and support automatic
  background refresh on the same session axis. Mapping corrections reset the fit.
- **Replay and recovery.** `operations.jsonl` records optimization decisions,
  explicit resets and initial fatigue. Replay, resume and sensitivity share a
  schedule interpreter, with ordering/coverage checks before recovery changes
  logs. Legacy 0.2 logs remain readable/resumable through marked recomputation;
  their unrecorded scheduling cannot be recovered. Solver warm starts remain
  unsaved, so crash recovery is checked within tolerance.
  Appending raw/quality JSONL after a torn write preserves the exact partial bytes
  in recovery sidecars before continuing complete records.
  Raw-frame reprocessing refuses to read the session's own acquisition output,
  including filesystem aliases, before changing session files.
  Sessions from models without actuators retain empty activation arrays through
  reading, replay, reporting and sensitivity analysis.
  Resume preserves a durable recovery journal until replay and its checkpoint
  commit. Interrupted recovery can be retried without losing pending poses;
  configuration preflight reads original metadata without changing the logs.
- **Worker lifecycle.** Timed close leaves resources open while the worker is
  active; completed close is idempotent. Health reports age, stale/error status,
  queue and drops. Queued resets preserve order. `drain()` waits explicitly for
  offline processing without stopping the worker.
  Failed checkpoint, file or sink finalization can be retried. Closing rejects
  new inputs and retains event progress so the live summary is recorded once.
- **Study quality.** Per-DOF reserves and required torques identify the joint
  behind a reserve warning; root residuals remain separate. Reports include raw
  tracking acceptance/reasons and protocol-specific JSON thresholds. Recorded
  optimizer settings include resolved model-dependent weights.
- **Sensitivity and validation.** Strength scenarios retain nominal solver
  weights. Gap/reset-aware interpolation returns missing values outside actual
  coverage. Empty sessions and disabled fatigue are supported. Independent MVC
  probes predict posture-specific joint torque under fatigue caps, with
  calibration trials excluded and descriptive agreement metrics. EMG windows
  with no valid spectral power return missing values.
- **Installation and checks.** Standard package metadata pins the exact published
  MyoSuite Git commit, so both uv and pip resolve the prerequisite without a
  sibling checkout. `uv.lock` freezes the remaining dependencies. CI defaults to
  that same source commit and checks the installed revision; editable local
  development is an explicit override. Existing benchmark tables are labelled
  as the historical 0.2 snapshot, since current optimizer defaults differ.

## 0.2.0

Changes the logs and fatigue across tracking gaps; sessions recorded with
0.1.0 cannot be read or resumed.

- **Fatigue across tracking gaps.** Untracked time is now integrated, as rest
  (`gap_load="rest"`, the default) or as the last load held (`"hold"`). In
  0.1.0 fatigue froze over gaps, so it stayed too high after every break
  between sets. The first estimate after a gap carries `Flag.AFTER_GAP` and
  the gap length.
- **Quality flags** on every pose and estimate: low confidence, missing
  keypoints, IK residual, joint limit, glitch, after gap, capacity exceeded,
  solver failure.
- **Wall-clock times and events.** Inputs log receive and capture times, and
  estimates log their frame's time and completion time, so end-to-end latency
  is measurable. `ShadowModel.mark(label, **data)` logs experimenter events
  to `events.csv`.
- **Checkpoints and resume.** `checkpoint.json` is written atomically every
  `checkpoint_every` seconds. `ShadowModel.resume(log_dir)` continues a
  crashed session.
- **Real time.** `LiveShadowModel` runs the pipeline on a worker thread. It
  optimizes only the newest pose when it falls behind and drops (and counts)
  only poses on overflow. `estimate_every` decimates optimizations;
  `push(estimate=False)` updates kinematics only.
- **Calibration.** Strength from an MVC (`strength_from_mvc`,
  `scale_strength`), body mass (`scale_body_mass`), and 3CC-r F (and R) from
  endurance times (`endurance_time`, `fit_fatigue_parameters`,
  `ShadowModelConfig.fatigue_params`). The record stores the calibration.
- **Tracker IK** (`myosuite_shadow.ik`, extra `ik`): keypoints onto model
  frames with mink, joint couplings as hard constraints, quality flags.
- **LabStreamingLayer** (`myosuite_shadow.lsl`, extra `lsl`): estimates and
  markers on the LSL clock, stamped at their frame's capture time.
- **After the session.** `myosuite-shadow report` checks a session's quality.
  `myosuite-shadow sensitivity` replays it under calibration errors and gives
  uncertainty bands. `myosuite_shadow.validation` holds the EMG, XDF and
  agreement statistics.
- `uv.lock` pins every dependency, including the MyoSuite commit.
- The record stores the model's mass and every resume. Package provenance
  now covers mink and pylsl.

## 0.1.0

- `ShadowModel`, moved from MyoHub/myosuite#640, with session logs, replay
  CLI and benchmarks.
