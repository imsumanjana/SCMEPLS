# Limitations and required validation

- Default values are illustrative and must not be described as measured results.
- Relative vibration and radar scores do not establish full-scale performance.
- Welch PSD/dominant-frequency analysis assumes approximately uniform sampling; irregular time histories must be resampled before analysis.
- The control-response models are reduced-order and do not include flexible-body modes, detailed actuator saturation, switching delay, sensor quantization, or full flexible six-degree-of-freedom coupling.
- Energy results depend entirely on the entered energy budget and do not estimate magnetic, thermal, converter, cryogenic, or infrastructure losses automatically. Recovered energy is constrained to the gross supplied energy in the present accounting model.
- The SC-MEPLS rollout simulation is a reduced-order rigid-body model. It separates true, measured and estimated air gaps and models distributed electromagnetic/pneumatic support, but it is not a substitute for Simscape/Multibody, electromagnetic FEA, CFD, structural FEA, or test data.
- The mechanical lock is represented by reduced-order translational/rotational restoring elements rather than detailed pin/socket contact, friction, backlash, impact, or structural compliance.
- Fault detection/health knowledge remains simplified. The model does not yet represent a complete diagnostic observer, communication latency, or probabilistic fault-detection performance.
- Imported GLB/STL geometry is visualization-only. Mesh validation warnings do not certify a geometry for FEA, manufacturing, collision/contact analysis, or safety-critical clearance verification.
- The geometry manifest validates component naming/binding only; dynamic actor binding to the SC-MEPLS time history is a later stage.
- The AI calibration module is valid only inside the domain represented by its training data. KFold, chronological, or grouped cross-validation is not a replacement for a final external validation dataset.
- Feature selection remains the user's responsibility; target leakage or physically meaningless inputs can still produce misleading model scores.
- Export resolution does not increase scientific validity; high DPI only improves publication rendering.
