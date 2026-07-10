# Scientific basis

## Scope

The software supports article-level feasibility analysis for a hybrid maglev–mechanical rocket ground-transport concept. It does not certify a full-scale launchpad. The application separates five evidence classes: illustrative, literature-derived, simulation, experimental, and AI-calibrated.

## Vibration

For each acceleration history, the root-mean-square acceleration is

`a_RMS = sqrt((1/T) integral_0^T a(t)^2 dt)`.

The article comparison plot uses a normalized index, normally referenced to the largest RMS value or a selected baseline. The software also reports peak, peak-to-peak, crest factor, and dominant frequency obtained from Welch power spectral density. Generated default signals contain harmonics, broadband noise, and transient events only to demonstrate the workflow. They are not measurements.

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

This avoids the ambiguous use of “efficiency” for a level trip that starts and ends at rest. A normalized energy-performance index is also provided for visual comparison, but the physical SEC should remain in the paper or supplementary material.

## Radar/MCDA

Every radar axis is written so that a higher score is better. The default 1–5 scores are illustrative expert scores. Weights are normalized before calculating the weighted aggregate. Publication use requires a documented scoring rubric and traceable evidence for each value.

## SC-MEPLS reduced-order simulation

The eight-module model represents distributed electromagnetic lift, longitudinal propulsion, gap/attitude dynamics, center-of-gravity shift, wind disturbance, coil degradation, sensor bias replacement by the median of healthy sensors, pneumatic leakage, first-order current/pressure actuator dynamics, and force-balanced load handover. During transfer, total support approximately satisfies

`sum(F_EM) + sum(F_pneumatic) + F_lock = M*g`.

The default parameters reproduce a scaled feasibility scenario. Full-scale design requires electromagnetic FEA, structural FEA, identified actuator dynamics, environmental qualification, hardware testing, and independent safety review.

## AI calibration

The AI module is deliberately constrained. It fits Random Forest or Gradient Boosting regressors only to imported numeric datasets. It reports cross-validated R² and MAE. AI cannot convert illustrative assumptions into verified physics; it can only interpolate patterns present in traceable calibration data.
