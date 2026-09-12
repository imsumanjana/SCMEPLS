# SC-MEPLS v1.1.1 — correctness hotfix

This release corrects configuration, provenance, and input-validation defects identified during the v1.1.0 repository audit.

- Synchronizes package/app version metadata to 1.1.1.
- Retains NumPy 1.26 compatibility in control metrics.
- Rejects invalid rollout fault indices and impossible gap/end-time configurations.
- Requires a complete rollout sequence before transfer/hard-lock metrics are accepted.
- Aligns GUI gap/end-time ranges with the numerical solver.
- Keeps imported vibration provenance explicit instead of forcing Experimental.
- Marks imported vibration uncertainty as unspecified instead of zero.
- Rejects recovered energy greater than gross supplied energy.
- Adds regression tests for the corrected validation paths.
