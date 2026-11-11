# Human validation protocol

This is a protocol and analysis workflow, not a completed validation study.
No participant measurements or accuracy claims are supplied with the repository.

## Define claims and holdouts

Choose primary outputs before collection: joint loading, activation timing or
posture-specific residual MVC torque. MF is a model state; EMG does not directly
measure force or MF. Individual-muscle estimates need stronger evidence than
joint-level trends because minimum-effort optimization does not identify
co-contraction.

Separate calibration MVC/hold trials from validation trials. Freeze parameters,
then evaluate independent trials/postures/tasks. For generalization claims,
hold participants out of parameter/threshold selection too. Count people
separately from repeated probes and frames. Choose recruitment, sample size,
exertion and stopping rules through the study's normal review process.

## Independent measurements

Collect synchronized dynamometer/force measurements, relevant EMG, raw tracking,
quality flags and markers. Include measured rest/recovery, loaded/unloaded tasks
and another session/day for repeatability. Establish initial fatigue or the
protocol's rested condition. Log probe start, interval, posture and calibration.

Preserve Quest tracking flags and actual fidelity; assess inferred legs separately.
Validate Sapiens2 reconstruction against a reference across placement, occlusion,
motion and body regions. Test identity changes and missing views. Full-body
loading needs validated floor/root registration and ground reactions; force plates
provide independent comparisons where available.

Check timing with a recorded common event, including after refreshes and restarts.
Distinguish capture, evaluation/prediction and receive time. Select quality
thresholds for actual study requirements; defaults do not establish accuracy.

**MVC probes cause exertion.** Supply measured reaction forces/torques during
probes. A static pose and marker omit this loading. Predict capacity at the
predefined probe time, such as immediately before exertion, then account for probe
loading in subsequent fatigue. Prespecify probe/recovery handling instead of
fitting it away using validation measurements.

## Exclusions and analysis

Archive a JSON `ReportThresholds` file: `unsolved_fraction`, `capacity_fraction`,
`flagged_fraction`, `latency_s`, `reserve_rms_fraction`, `required_calibrations`.
Reserve warnings concern model adequacy; they can reflect geometry, strength or
tracking errors and do not establish participant failure. Prespecify landmarks,
inference, IK/reprojection error, timing, dropout and usable coverage. Report
exclusions by participant/condition. Assess root residuals separately from joint
reserves. Exclude probes in gaps and across tracking resets.

Probe CSV columns:

```text
time,joint,measured_torque,role,qpos_0,...,qpos_<nq-1>
```

`time` uses the session axis. `qpos_i` is the measured probe posture in the
subject model. `measured_torque` is signed **gravity-corrected net joint torque**
(N m for hinges, N for slides), matching joint signs/passive contributions.
Document the instrument correction. `role` must be `calibration` or `validation`;
calibration rows are excluded from agreement metrics.

```bash
myosuite-shadow-validate --session logs/P07_B --probes P07_probes.csv --out logs/P07_B/validation.json
```

The command checks model identity, computes maximal torque under each muscle's
fatigue cap at the measured posture, and excludes unsupported times and
tracking/solver/inferred flags. It reports predictions, exclusions and
joint/direction-specific MAE, RMSE, bias and concordance. Limits of agreement are
descriptive within-session summaries, without population uncertainty or adjustment
for repeated probes.

Across people, use subject-aware intervals or hierarchical analysis.
`repeated_measures_correlation` supports within-participant association;
correlation alone does not establish agreement. Report participant/task variability
and processing/calibration sensitivity. `residual_capacity` remains a descriptive
unweighted fatigue index; use `mvc_capacity` for predicted joint MVC torque.

Use `load_xdf`, `emg_envelope`, `emg_median_frequency` and `sample_at` for EMG
comparisons. Normalize with an independent reference, inspect artifacts/electrodes,
and compare timing/shape before asserting amplitude agreement. Missing/zero-power
frequency windows return NaN. Analyse dynamic tasks separately from holds and
report latency/accuracy tradeoffs. Avoid tuning on validation outcomes.

The calibration/validation separation follows
[Hicks et al., musculoskeletal verification and validation guidance](https://pmc.ncbi.nlm.nih.gov/articles/PMC4321112/).
