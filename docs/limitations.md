# Limitations

The shadow model's outputs are **model estimates**, not measurements. State the
following in any study that uses them; none is fixed by calibration alone.

## Muscle model and recruitment

- **Minimum effort.** Static optimization minimises the sum of squared
  activations. It predicts no co-contraction and shares load among synergists
  in fixed proportions. Real recruitment varies between people, tasks and
  fatigue states, so estimates of individual muscles are less reliable than
  joint-level loading.
- **Passive forces.** MyoSuite's muscle parameters are generic. In the elbow
  model the passive flexor torque exceeds the forearm's gravity torque near
  full extension, with a passive equilibrium at about 25-30 degrees. Holding
  the arm straighter needs estimated extensor activation, so extensor fatigue
  can rise during rests with the arm hanging straight. Treat extensor
  estimates near full extension as unreliable.
- **Anthropometry.** `scale_body_mass` scales masses and inertias only.
  Segment lengths, moment arms and muscle paths stay generic unless the
  model is scaled to the participant first (MyoSuite's subject scaling).
- **Strength.** One MVC gives one scale for the muscles that produced it
  (`strength_from_mvc(...).agonists`); scale only those. Without it, absolute
  activation and fatigue rates can be off by tens of percent.
- **Model capacity.** Generic models cannot produce every posture's torques.
  On random myoArm reaches, without noise or scaling, the shoulder
  (`elv_angle`, `shoulder_rot`) needs more than 1 N m of reserve in 16 % of
  frames. `CAPACITY_EXCEEDED` there reflects the model, not the participant;
  check which joints need reserves before interpreting the flag.

## Fatigue (3CC-r)

- **Parameters.** By default every muscle uses MyoSuite's functional-group
  parameters, fitted to population endurance-time curves.
  `fit_fatigue_parameters` fits F (and R) to a participant's sustained holds.
  It cannot inform the rest multiplier r.
- **Driver.** Fatigue is driven by the estimated activations, so every error
  in loading propagates into fatigue. The cap at `1 - MF` feeds fatigue back
  into the optimization; it is a modelling choice, not an observation.
- **Scope.** MF is the loss of available motor units for the modelled task.
  Central and peripheral components, metabolic state, pain and motivation are
  not separated.
- **Start.** Fatigue starts at the first estimate of a session. A session
  should start rested, or the initial state should be set explicitly.

## Pipeline assumptions

- **Tracking gaps.** Untracked time longer than `max_gap` counts as rest
  (`gap_load="rest"`), or as the last load held (`"hold"`). Pick the one that
  matches the protocol. The first estimate after a gap is flagged
  `AFTER_GAP`.
- **Delay against accuracy.** Accelerations from a live pose stream are noisy.
  The kinematics lag trades delay against accuracy (see the README's
  benchmark). Estimates at zero delay are poor; dropping accelerations
  (quasi-static) is the better zero-delay option.
- **Real time.** When the optimizer falls behind, the live runner skips
  optimizations, not poses, so fatigue still covers all time. The skipped
  frames have no estimate of their own.
- **Ground reactions** are estimated only against world-fixed bodies (floor,
  chair, table), not between body segments or with hand-held objects. Known
  hand loads must be passed as external forces.
- **Resume after a crash.** Resumed sessions match uninterrupted ones to the
  optimizer's tolerance, not bit for bit, because the solver's warm start is
  not saved. The downtime counts as a tracking gap.

## Validation

None of the estimates has been validated against people in this repository.
The `validation` module provides the tools (EMG envelopes and median
frequency, residual capacity at MVC probes, concordance and repeated-measures
correlation). The headset prototype's `docs/validation_study.md` describes a
study design.
