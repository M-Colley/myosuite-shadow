# Changelog

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
