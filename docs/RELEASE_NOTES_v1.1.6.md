# SC-MEPLS v1.1.6 — QA and reproducibility baseline

This release closes the repository-audit cycle and establishes the next validation baseline.

- Introduces a single-source software version module and dynamic package versioning.
- Packages the default project resource correctly for installed-package use and removes the duplicate top-level default project definition.
- Makes runtime output paths independent of the source-tree layout.
- Adds GitHub Actions CI for Python 3.10 and 3.12 with package-install and pytest checks.
- Rejects irregularly sampled vibration histories before Welch PSD analysis.
- Rejects negative/non-finite MCDA weights and records criterion weights in export metadata.
- Strengthens AI calibration with explicit feature selection, K-fold/chronological/grouped validation, reproducibility metadata, library versions, and a SHA-256 data fingerprint.
- Wires the Overview tab into the main application.
- Adds provenance validation and software-version metadata to figure exports.
- Expands batch-export metadata to retain models, parameters, scores, weights, and computed metrics.
- Synchronizes README, user guide, data formats, scientific basis, and limitations with the corrected v1.1.x architecture.
- Adds QA regression tests for packaged resources, version consistency, irregular sampling, MCDA validation, and AI split modes.

v1.1.6 is intended to be the stable baseline before connecting SC-MEPLS time-history signals to imported GLB/STL actors and before formal MATLAB/Python parity studies.
