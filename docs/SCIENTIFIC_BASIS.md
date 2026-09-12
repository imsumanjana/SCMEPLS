# Scientific basis

## Scope

The software supports article-level feasibility analysis for a hybrid maglev–mechanical rocket ground-transport concept. It does not certify a full-scale launchpad. The application separates five evidence classes: illustrative, literature-derived, simulation, experimental, and AI-calibrated.

## Vibration

For each acceleration history, the root-mean-square acceleration is

`a_RMS = sqrt((1/T) integral_0^T a(t)^2 dt)`.

The article comparison plot uses a normalized index, normally referenced to the largest RMS value or a selected baseline. The software also reports peak, peak-to-peak, crest factor, and dominant frequency obtained from Welch power spectral density. Welch analysis assumes approximately uniform sampling; the implementation rejects imported series whose sample-interval jitter exceeds the documented tolerance. Generated default signals contain harmonics, broadband noise, and transient events only to demonstrate the workflow. They are not measurements.

## Closed-loop response

Three model classes are available:

1. First-order equivalent: `G(s)=K/(tau*s+1)`.
2. Standard second-order: `G(s)=K*wn^2/(s^2+2*zeta*wn*s+wn^2)`.
3. Mass–damper plant with unity-feedback PID. For `G_p(s)=K_a/(M*s^2+B*s)` and `C(s)=Kp+Ki/s+Kd*s`, the closed-loop denominator is `M*s^3+(B+K_a*Kd)*s^2+K_a*Kp*s+K_a*Ki`.

The software reports rise time, 2% settling time, overshoot, steady-state error, IAE, and ITAE. Equivalent models should be replaced by identified or validated subsystem models before quantitative claims.

## Energy

The preferred metric is net specific energy consumption:

`E_net = E_traction + E_levitation + E_auxiliary - E_recovered`

`SEC = E_net / (mass_tonnes * distance_km)`.

The current accounting model requires `E_recovered <= E_gross`; exported or externally supplied energy would require an extended balance. A normalized energy-performance index is also provided for visual comparison, but the physical SEC should remain in the paper or supplementary material.

## Radar/MCDA

Every radar axis is written so that a higher score is better. The default 1–5 scores are illustrative expert scores. Criterion weights must be finite and non-negative and are normalized before calculating the weighted aggregate. Publication use requires a documented scoring rubric and traceable evidence for each value. Scores, weights, and ranking are retained in export metadata.

## SC-MEPLS reduced-order simulation

The eight-module model represents distributed electromagnetic lift, longitudinal propulsion, rigid-body gap/attitude dynamics, center-of-gravity shift, wind disturbance, coil degradation, sensor bias with fault-tolerant estimation, pneumatic leakage, first-order current/pressure actuator dynamics, load transfer, seating, and reduced-order mechanical hard locking.

Three gap quantities are kept distinct:

- `gap_i_m`: true physical module gap used by the electromagnetic plant equation and physical safety checks;
- `sensor_gap_i_m`: raw sensor channel including injected bias when active;
- `estimated_gap_i_m`: controller/fault-tolerant estimate used by the electromagnetic allocator/controller.

This separation prevents sensor reconstruction from changing the physical plant force law.

During docking, mechanical support is enabled only after the transporter enters a position/lateral/yaw docking envelope. The scheduled handover then reduces the levitation reference toward the seated gap while pneumatic/mechanical support increases. Reduced-order lock forces restrain X/Y/Z translation and roll/pitch/yaw rotation. This remains an engineering abstraction rather than a detailed contact model.

Module-level outputs expose true/measured/estimated gap, current, electromagnetic force, pressure, pneumatic force and health/effectiveness for independent MATLAB/Python comparison.

Full-scale design requires electromagnetic FEA, structural FEA, identified actuator dynamics, environmental qualification, hardware testing, and independent safety review.

## External 3D geometry

GLB/STL files are visualization assets only. The viewer validates basic mesh/topology properties, scale, naming, and optional component manifests. Geometry does not change numerical mass, inertia, actuator coordinates, controller gains, or force laws. A future explicitly validated geometry-derived-physics mode would be a separate development stage.

## AI calibration

The AI module fits Random Forest or Gradient Boosting regressors only to imported numeric datasets. Users select the target and feature columns explicitly. KFold is available for independent runs, chronological splitting for ordered data, and GroupKFold when a run/group column is supplied. Reported R² and MAE are out-of-sample cross-validation metrics for the selected split strategy. Saved models retain the feature list, validation mode, package/library versions, and a SHA-256 fingerprint of the calibration table. Cross-validation is not a replacement for a final external validation set, and AI cannot convert illustrative assumptions into verified physics.
