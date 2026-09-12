# SC-MEPLS v1.1.2 — physics-path correction

This release corrects the most important reduced-order simulation inconsistencies found in the repository audit.

- Separates true physical air gaps, raw sensor gaps, and controller-estimated gaps.
- Uses true physical gap in the electromagnetic plant force equation.
- Bases the physical unsafe flag on true gap/attitude rather than reconstructed controller measurements.
- Removes the previous double derating of coil faults by compensating desired force conversion with the estimated actuator effectiveness.
- Removes perfect advance compensation of pneumatic leaks; leaks now reduce actual pressure/force unless a future observer/controller is added.
- Uses exact first-order actuator discretization for current and pressure dynamics.
- Makes the vibration RMS proxy time-step consistent using a physical averaging time constant.
- Restricts the explicit rigid-body solver to dt <= 0.02 s.
- Gates mechanical lock engagement on a docking-ready condition so lock force cannot act while the transporter is away from the launch datum.
- Removes the unused rollout random seed parameter.
