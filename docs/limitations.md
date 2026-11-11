# Limitations

The outputs are **model estimates**. This repository supplies software tests and
a validation workflow, not recordings or validation results from participants.
Calibration and quality flags do not establish accuracy for a person or task.
Use the [validation protocol](validation_study.md) to define and evaluate the
study's claims.

## Model and participant

- **Recruitment is underdetermined.** Static optimization minimizes squared
  activations subject to model forces, bounds and reserve penalties. Load sharing
  changes with posture, moment arms, strength, external loading and fatigue.
  Pose alone does not identify voluntary co-contraction or an individual's
  recruitment strategy. Individual-muscle estimates need independent evidence.
- **Generic geometry and passive forces.** Registration only changes coordinates.
  `scale_body_mass` only scales selected masses and inertias; segment lengths,
  muscle paths and moment arms require a prepared subject model.
  `check_segment_lengths` checks distances, not every anatomical property.
  Generic passive forces can generate predicted activation even during a posture
  intended as rest. Inspect passive torque and range of motion for the task.
- **Mass semantics.** A partial model represents only some of the participant.
  Use a documented whole-body participant/reference mass ratio on anatomical
  bodies, or the measured represented-segment total. Exclude scenery and props.
  Neither choice establishes correct segment-by-segment mass distribution.
- **MVC calibration is effective strength.** A single MVC scales the returned
  agonist group; it cannot separate synergists or inform unrelated muscles.
  Multi-posture fits provide more constraints but do not establish each
  muscle's true strength. Torque sign, gravity correction, passive contributions,
  posture, dynamometer geometry and effort affect the fit. Keep calibration
  measurements separate from validation outcomes.
- **Reserves are model discrepancies.** A reserve can reflect model geometry,
  strength, omitted loading or tracking error. `CAPACITY_EXCEEDED` is not a
  measurement of participant exhaustion. Inspect per-joint reserves relative
  to required loading and free-root force/moment residuals separately.
  The historical 0.2 benchmark numbers do not establish current model adequacy.

## Tracking, reconstruction and time

- **Observability.** Required-landmark and residual gates reject selected failures;
  they do not guarantee that every modeled degree of freedom is constrained.
  Axial rotations need appropriate landmarks, and body origins may not coincide
  with tracked anatomical landmarks. Check the mapping and held-out poses for
  each model/source configuration.
- **Quest body estimates.** API-valid positions can be inferred. Full-body output
  includes estimated legs, and actual fidelity can differ from requested fidelity.
  Keep source flags and evaluate accuracy by body region and task. Enabling
  inferred inputs or prediction is an explicit protocol choice, not extra
  measured information.
- **Sapiens2 observations.** Detector pixels and scores are not metric 3D
  measurements. Reconstruction requires calibrated synchronized views or
  registered metric camera-Z depth. Occlusion, identity errors, camera geometry,
  image timestamps, depth error and reprojection error propagate into IK.
  Calibration and reconstruction tests do not establish real-camera accuracy.
- **Timing meanings.** Quest poses describe a requested evaluation time, usually
  not a measured sensor capture time. Keep `capture_unix`, `evaluation_unix`
  and `receive_unix` distinct. Frame/LSL timestamps prefer them in that order.
  Measured clock mappings still have uncertainty and a validity interval;
  network arrival does not reveal sensor capture time. Check alignment against a
  common recorded event in the actual measurement setup.
- **Recentring and identity.** Source-space or skeleton changes require renewed
  registration and a reset of IK/differentiation priors. An unhandled jump creates
  artificial velocities and accelerations. Stable source/person identifiers and
  recorded revision changes are necessary but do not replace identity review.
- **Hardware status.** Software and loopback tests do not test an on-device Quest
  session, Android permissions, camera synchronization or Sapiens2 GPU inference.
  See [tracking.md](tracking.md) for the implemented paths and remaining checks.

## Fatigue (3CC-r)

- **Parameters and initial state.** Defaults use functional-group population
  parameters. Hold fits estimate F and, when requested, R under that protocol;
  they do not identify the rest multiplier r. Sustained holds alone provide
  limited information about recovery. Start from a protocol-defined rested
  condition or an explicitly initialized fatigue state, including fatigue
  induced by calibration. The initial state is recorded with the schedule.
- **Driver and feedback.** Estimated activations drive fatigue, so geometry,
  loading, tracking and recruitment errors propagate into MF. The `1 - MF`
  activation cap feeds that state back into optimization. This is a modeling
  assumption, not an observation of remaining physiological capacity.
- **Interpretation.** MF is a compartment state for the modeled task. It does not
  separate central/peripheral mechanisms, pain, motivation or metabolic state.
  An unweighted group mean of `1 - MF` is descriptive; posture-specific joint
  MVC prediction must also use muscle strength, moment arms and passive torque.

## Loading and live processing

- **External loads must reach the solver.** An event saying “2 kg” or an MVC marker
  does not apply a force. Supply synchronized generalized loads or measured
  body/site wrenches. `held_mass_qfrc` includes gravity only; it omits object
  acceleration, rotational inertia and contacts. Do not double-count an object
  already represented in model masses.
- **Contacts.** Estimated ground reactions act against world-fixed bodies, such
  as a floor, chair or table. They do not identify all human/object or intersegment
  forces. Floor/root registration and contact assumptions need validation,
  particularly for full-body loading.
- **Noise and delay.** Differentiating poses amplifies tracking noise. Window/lag
  selection trades delay against derivative accuracy; measure the tradeoff in
  the actual acquisition setup. The README's 0.2 table is a historical example
  with different optimizer defaults, not a current error bound.
- **Tracking gaps.** Long gaps integrate zero load (`gap_load="rest"`) or the last
  load (`"hold"`). Occlusion does not imply rest. Integration occurs when the
  next accepted pose arrives; the previous displayed estimate remains old in
  the meantime. Check health/age before using `latest` for feedback.
- **Backlog and overflow.** The worker skips optimizations when behind and drops
  counted poses on overflow. Fatigue still covers elapsed time under the sampled
  load assumption; this does not reconstruct unobserved activation. Every skipped
  frame does not receive an independent estimate. Offline raw replay drains each
  frame and can therefore differ from the original live schedule.
- **Recovery and reproducibility.** New logs retain the processing decisions and
  resets, but not the optimizer's warm start. Resumed results are checked within
  solver tolerance. The clock must continue across downtime. Legacy 0.2 logs
  lack their original schedule and use marked recomputation. Archive both paired
  source revisions and their edits as well as the lockfile; editable sources are
  not frozen by the lock.

## Study analysis

Report thresholds are study policy, not validated universal acceptance limits.
Sensitivity scenarios show selected perturbations, not calibrated confidence
intervals or all sources of uncertainty. Missing coverage stays missing across
gaps, resets and changed fit windows.

The independent-probe command checks model identity and separates rows labelled
calibration/validation; the investigator must ensure those labels reflect the
actual data use. MVC probes themselves cause exertion: include their measured
loading in subsequent fatigue and prespecify the prediction time.

Within-session agreement metrics and limits of agreement are descriptive.
Repeated probes are not independent participants, and correlation does not
establish agreement. EMG also needs artifact review, independent normalization
and timing checks; it is not a direct measurement of muscle force or MF.
See [study_session.md](study_session.md) and
[validation_study.md](validation_study.md) for the operational protocol.
