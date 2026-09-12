# SC-MEPLS v1.2.6 — Wired 3D digital twin

This release completes the sequential wiring requested after the v1.1.6 audit baseline.

- **v1.2.0 Geometry:** semantic component bindings connect GLB/STL nodes to world/platform/module/lock groups, with manifest-defined pivots and conservative name inference.
- **v1.2.1 Mesh:** imported triangle surfaces are registered as immutable PyVista meshes with component identity, result-safe display modes, and no feedback into the numerical model.
- **v1.2.2 Physics:** a read-only `SimulationTimeline` maps SC-MEPLS rigid-body and M1–M8 signals to digital-twin frames and reference-relative rigid transforms.
- **v1.2.3 Animation:** deterministic play/pause/reset/seek/speed/loop playback and per-component transforms are provided without changing solver state.
- **v1.2.4 Results:** module gap, current, electromagnetic force, pressure, pneumatic force, health, force vectors, CG and lock state are available for visualization.
- **v1.2.5 Plotting:** the 3D twin gains linked module/rigid-body time-history plots with a synchronized time cursor.
- **v1.2.6 UI:** the PyQt6 3D Digital Twin tab now integrates imported geometry, mesh modes, manifest bindings, live SC-MEPLS history, playback controls, module results, force/CG overlays, linked plots and reproducible screenshot/plot export.

The `SC-MEPLS Simulation` tab emits each completed run to the 3D Digital Twin automatically. A time-history CSV can also be loaded directly into the twin. Geometry remains visualization-only and cannot alter the independent SC-MEPLS physics model.
