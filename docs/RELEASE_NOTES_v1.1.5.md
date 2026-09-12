# SC-MEPLS v1.1.5 — geometry hardening

This release strengthens external GLB/STL geometry handling before simulation animation is connected to the viewer.

- Adds a versioned JSON geometry-manifest schema with component roles, dynamic groups, required components, and M1–M8 simulation-module bindings.
- Automatically looks for `<geometry-name>.manifest.json` and validates it against imported component IDs.
- Replaces fragile Python-object-id mesh picking with persistent component indices stored in VTK/PyVista field data.
- Adds checks/warnings for invalid face indices, degenerate triangles, zero-area triangles, duplicate faces, inconsistent winding, non-watertight meshes, disconnected bodies, duplicate component IDs, extreme scale, and excessive triangle count.
- Distinguishes the standard GLB metre convention from explicit nonstandard unit overrides.
- Moves GLB/STL parsing to a Qt worker thread to reduce GUI blocking during file loading.
- Adds 3D screenshot export with a JSON sidecar containing geometry source, units, axis mode, component IDs, warnings, manifest status, and selected component.
- Adds regression tests for GLB axis conversion and manifest validation.

Imported geometry remains visualization-only and cannot change SC-MEPLS numerical physics.
