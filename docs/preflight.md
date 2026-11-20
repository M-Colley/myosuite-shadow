# Check a study configuration before collection

Run preflight with the same configuration used by `myosuite-shadow-track`:

```bash
myosuite-shadow-preflight --config study.json --out preflight.json
```

The command loads the prepared model, IK mapping, acquisition policy and optimizer
settings. It opens no recording or tracker and writes a new report without
overwriting an existing file. `blocked` exits with code 1; malformed configuration
exits with code 2. Warnings give `needs_attention` and exit 0. Archive the report
with the model, configuration and source revisions.

## Declare the scope and task postures

Add a `preflight` object to the tracking JSON:

```json
{"preflight": {
  "target_joints": ["r_elbow_flex"],
  "fixed_joints": [],
  "postures": [[1.2], [0.6]],
  "require_full_rank": true,
  "translation_scale_m": 0.1,
  "rotation_scale_rad": 1.0,
  "landmark_scale_m": 1.0,
  "max_condition_number": 1000000,
  "required_calibrations": ["geometry", "strength"]
}}
```

This abbreviated posture example fits a one-coordinate elbow model. Replace it
with measured, coupling-consistent `nq`-length postures from the actual task and
the calibration names used by your protocol. Joint selections expand to their
independent tangent DOFs. `target_dofs` and `fixed_dofs` can instead contain full
model DOF indices or the exact names returned in the report, such as `root/rx`.
Select one syntax per scope. Dependent/locked coordinates cannot be selected as
independent DOFs.

The analysis uses the required mapped position landmarks, with unit weights.
Other independent DOFs remain free nuisance motion: if they can mimic target
motion, the target is ambiguous. `fixed_joints`/`fixed_dofs` declare an explicit
known-coordinate assumption. Unmapped fingers do not invalidate an observable
arm scope merely by existing in the model.

Each posture reports rank, scaled singular values/condition, ambiguous DOFs and
groups. `--include-matrices` adds the Jacobian and nullspace arrays. The input and
analyzed (projected/normalized) poses are retained. Scales are characteristic
physical increments, not measurement uncertainty; choose them and numerical
thresholds for the protocol. `dof_scales` can override them by independent DOF
index. With no target scope, all independent DOFs are diagnostic and rank
deficiency warns. With an explicit scope, `require_full_rank` makes deficiency
blocking. With no postures, only the initial model posture is analyzed and warns.

These are local position-geometry checks. Joint limits, posture priors, temporal
information, source confidence and orientation measurements are not extra
position evidence. Check multiple measured postures and use recording regressions
for missing/occluded landmarks. Declared calibration records are checked for
presence; their physical correctness still needs independent measurements.

## Check live clocks explicitly

```bash
myosuite-shadow-preflight --config quest_study.json --live --out live_preflight.json
myosuite-shadow-preflight --config quest_study.json --live --probe-clock --out clock_preflight.json
```

`--live` checks the configured Quest mapping against the current host time, its
validity interval and the acquisition uncertainty budget. `--probe-clock` uses
`quest.clock_host` and the existing bounded exchange options to measure a refresh.
It requires the same source identity, session axis and Unix reference. The
measured mapping is saved in the report; the command does not replace the study
configuration. Spatial registration must target `model_world`.

Live acquisition requires a `quest` object. Declare its `source_id`, `person_id`
and `joint_set` (`upper_body` or `full_body`). Required landmark names must belong
to that layout; the receiver rejects packets reporting a different declared
layout. Registration must start in `quest_stage`, `quest_local` or
`quest_local_floor`, matching the measured source space. Device marketing names
do not establish capability. Clock refresh options are checked before any probe,
including finite limits and a refresh interval shorter than mapping validity.

Preflight does not establish on-device support, camera synchronization, identity
accuracy, ground reactions or participant validity. Preserve actual source
recordings and synchronized reference measurements for those checks.
