# SC-MEPLS v1.1.3 — seating and hard-lock correction

This release makes the hybrid handover sequence physically closer to the intended roll-out/alignment/seating/locking process.

- Adds an explicit seating phase that reduces the levitation gap from the transport target back toward the seated/initial gap.
- Prevents seating from starting until the transporter is within the docking-ready position/lateral/yaw envelope.
- Adds mechanical lock restoring forces in longitudinal and lateral translation.
- Adds mechanical yaw restraint in addition to the existing vertical/roll/pitch lock terms.
- Disables active longitudinal/lateral/yaw control after hard lock so the mechanical restraint carries the locked-state role.
- Adds final lateral, yaw, and mean-gap validation metrics.
- Requires the seated gap to be smaller than the levitation target gap.
- Adds regression tests for final seating and locked-state alignment.
