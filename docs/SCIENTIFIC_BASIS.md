# Scientific basis

## Scope

SC-MEPLS Analysis Studio v1.4.0 supports research-level feasibility analysis of a swarm-coordinated modular electromagnetic–pneumatic roll-out and locking system for rocket ground transportation. It does not certify a full-scale launchpad. Evidence provenance should be identified as illustrative, literature-derived, simulation, experimental or AI-calibrated.

## Coordinate system and rigid-body dynamics

The model and 3D viewer use the same right-handed frame: X = roll-out, Y = lateral, Z = vertical; positive roll/pitch/yaw are rotations about +X/+Y/+Z. The body-to-world attitude is ZYX. For small roll/pitch angles, the vertical offset of a module at body coordinate `(x,y)` is approximately `roll*y - pitch*x`. This sign convention is shared by gap calculation, viewer transforms and structural body-frame conversion.

M1–M8 support coordinates are parameters. Manual reduced-order runs use configured coordinates; validated geometry-coupled runs replace them with the explicit module points in the structural manifest.

The electromagnetic gap channels remain distinct:

- true physical gap: plant force law and physical-truth safety check;
- sensor gap: measurement including an injected sensor bias;
- estimated gap: controller input. A failed channel is reconstructed from a rigid-plane fit to the seven healthy sensor locations.

The lock is an interlocked state rather than a pure time schedule. Engagement is inhibited by physical truth violations or detected estimated-state violations. Hard-lock control suppression and low residual electromagnetic support are enabled only after confirmed lock engagement inside the docking envelope.

## Structural coupling

A quantitative structural run requires a closed structural surface and `.fea.json` manifest. Gmsh generates first-order tetrahedra. Tetrahedral connectivity orientation is normalized before analysis; a negative input node ordering is not by itself reported as a physically inverted element.

Volume integration provides mass, centroid and full centroidal inertia. For `moving_assembly`, those properties drive the reduced-order model directly. For `platform_only`, explicit payload mass, centroid and inertia are required and combined with the platform by the parallel-axis theorem. Payload loads enter through explicitly mapped support patches.

Module, propulsion, wind, lock and payload forces are distributed over boundary-node patches. Generalized moments are transferred as nodal wrenches rather than being silently absorbed by artificial constraints. World-frame forces/accelerations are rotated into the instantaneous structural body frame. Force and moment resultants are calculated before the FEA solve as a load-model audit; the post-solve reaction residual is retained separately as a numerical equilibrium check.

The current structural solver uses homogeneous isotropic linear elasticity, first-order tetrahedral elements, small strain and quasi-static frame solutions. A 3-2-1 kinematic constraint removes rigid modes and must be interpreted as a numerical stabilization, not as the physical maglev support condition. An optional coarse/base/fine mesh study compares displacement and von Mises stress before local stress values are treated quantitatively.

For heterogeneous assemblies, one-material FEA can only be an explicitly documented equivalent-homogeneous screening model. Joint/contact, weld/bolt, local multi-material, nonlinear contact, buckling, fatigue, plasticity and transient flexible-body claims require a validated external structural workflow.

## Imported 3D geometry

GLB/STL surface geometry can be used solely for visualization or, after a successful structural-manifest workflow, as the source for structural mass properties and actuator locations. These modes are deliberately separated so importing a mesh can never silently change the numerical plant.

Geometry-manifest dynamic groups distinguish stationary world geometry, moving platform/module geometry and world-side lock hardware. Optional axis/stroke metadata drives actual lock/actuator translation from lock fraction; engagement is not represented by opacity.

## Vibration

Acceleration RMS is `sqrt(mean(a_processed^2))`. Preprocessing is explicit: raw, mean removal, linear detrend or zero-phase high-pass filtering. The same processed series is used for RMS/peak/crest-factor and Welch PSD. Welch analysis requires approximately uniform sampling; histories exceeding the configured sample-interval jitter limit are rejected.

## Closed-loop response

The supporting control-comparison module includes first-order and second-order equivalents plus a mass–damper plant with PID derivative roll-off. The filtered derivative avoids the infinite high-frequency gain of an ideal derivative. It remains a linear comparison and does not model saturation/rate limits; those effects belong in the nonlinear SC-MEPLS simulation or a validated detailed plant model.

## Energy

`E_net = E_traction + E_levitation + E_auxiliary - E_recovered`

`SEC = E_net / (mass_tonnes * distance_km)`

SEC is the physical reporting metric. The optional relative efficiency is `best_SEC/system_SEC`, so its meaning is transparent and bounded by 1 for the best system. It is not an independent energy quantity.

## Radar / MCDA

All displayed criteria are oriented so higher is better. Weights are finite/non-negative and normalized for the weighted aggregate. The 1–5 scores remain evidence/expert inputs and require a documented rubric; they do not emerge from the physics model automatically.

## AI calibration

Random Forest or Gradient Boosting calibration is fit only to imported numeric data. Input features are explicit. K-fold, chronological and grouped validation produce out-of-sample R², MAE and RMSE. Near-perfect feature/target correlation is flagged as a possible leakage warning. A separately imported untouched external dataset can be evaluated after training; external metrics and a dataset fingerprint are reported. AI performance does not validate the underlying physics.
