# Renderer selection and verified scope

| Candidate | Fit for this experiment | Decision |
|---|---|---|
| VisPy | GPU scene/point rendering; additional mesh import, picking and camera integration | Not selected |
| PyQtGraph OpenGL | Lightweight Qt points, lines and meshes; additional anatomy and export integration | Not selected |
| PyVista/VTK + PyVistaQt | PLY import, transparent surfaces, skeleton lines, picking, camera and screenshot export | Selected |

Official references: https://vispy.org/,
https://pyqtgraph.readthedocs.io/en/latest/api_reference/3dgraphics/index.html,
https://qtdocs.pyvista.org/, https://docs.pyvista.org/,
https://doc.qt.io/qt-6/qthread.html.

Selection follows the required features and working implementation, not a target
GPU comparison benchmark. VTK 9.5 is required for the verified EGL route. Normal
desktop rendering uses QtInteractor. With `QT_QPA_PLATFORM=offscreen`, the same
real meshes, coordinates and colors are rendered by a pure offscreen VTK Plotter
and displayed in a QLabel, avoiding the missing embedded display context here.
This is actual geometry rendering, but it does not validate native NVIDIA camera
interaction or target frame rates.

Both original PLY files and all 16 skeletons are retained without synthesis.
Camera movement never repositions coordinates. Updates touch a 512-row RGBA
array and 16 skeleton actor properties. Alpha is
`round(255 * clip(abs(state) / display_scale, 0, 1))`; zero has no activity glow.
Gold is an explicit selection marker independent of activity.

Inference and the sparse CPU reservoir run on a dedicated QThread. Queued signals
send compact snapshots at at most roughly 20 Hz during generation. Idle updates
run every 250 ms when the worker is free. Simulation time is separate from wall
time. Rendering does not alter simulator state, and view pause suppresses only
visual updates. The genuine CPU worker validation passed with GUI heartbeat
callbacks during inference, actual render/export and normal exit.
