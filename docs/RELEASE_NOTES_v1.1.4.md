# SC-MEPLS v1.1.4 — validation-output expansion

This release exposes the internal signals needed for MATLAB/Python comparison and future 3D animation without changing the reduced-order governing equations.

- Adds mode names alongside numeric mission-state IDs.
- Adds lateral/vertical velocities, lateral acceleration, and angular rates.
- Exposes CG-shift states used by the scenario.
- Exposes all eight true physical gaps, raw sensor gaps, and controller-estimated gaps.
- Exposes M1–M8 coil health/effectiveness, coil current, electromagnetic force, pneumatic pressure, and pneumatic force.
- Keeps aggregate force/support outputs for existing plots and metrics.
- Adds a regression test that checks the complete module-level validation signal set.

The new fields are intended for independent MATLAB comparison, diagnostic plots, and subsequent binding to imported GLB/STL actors.
