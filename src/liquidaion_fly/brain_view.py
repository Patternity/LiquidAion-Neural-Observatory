"""VTK rendering of registered FlyWire geometry and measured reservoir state."""
from __future__ import annotations

from pathlib import Path
import os
import numpy as np
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QImage, QPixmap
import pyvista as pv
from pyvistaqt import QtInteractor


class BrainView(QWidget):
    neuron_selected = Signal(int)

    def __init__(self, anatomy, graph=None, parent=None):
        super().__init__(parent)
        self.anatomy = anatomy
        self.positions = np.asarray(anatomy.positions, dtype=np.float64)
        if self.positions.shape != (len(anatomy.neuron_ids), 3) or not np.isfinite(self.positions).all():
            raise ValueError("Anatomy must contain registered, finite positions for every graph neuron")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.headless = os.environ.get("QT_QPA_PLATFORM") == "offscreen"
        if self.headless:
            self.plotter = pv.Plotter(off_screen=True, window_size=(1100, 600))
            self.image_label = QLabel()
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setMinimumSize(350, 200)
            layout.addWidget(self.image_label)
        else:
            self.plotter = QtInteractor(self, auto_update=False)
            layout.addWidget(self.plotter.interactor)
        self.plotter.set_background("#080e19")
        self.plotter.add_text("FLYWIRE / FAFB14.1\nReal anatomical coordinates · nm", position="upper_left", font_size=9, color="#9bacc9", name="provenance")
        palette = ["#8394b2", "#596b90", "#9eabc7", "#728baa"]
        mesh_paths = list(getattr(anatomy, "mesh_paths", ()) or anatomy.meshes.values())
        if not mesh_paths:
            raise ValueError("No real anatomy meshes are available; synthetic geometry is forbidden")
        self.mesh_actors = []
        self.geometry = []
        for index, path in enumerate(mesh_paths):
            mesh = pv.read(str(path))
            if not mesh.n_points:
                raise ValueError(f"Empty anatomical mesh: {path}")
            self.geometry.append(mesh)
            self.mesh_actors.append(self.plotter.add_mesh(mesh, color=palette[index % len(palette)], opacity=0.20,
                smooth_shading=True, name=f"anatomy_{index}", pickable=False))
        self.points = pv.PolyData(self.positions.copy())
        self.activity_scale = 0.3
        self.current_state = np.zeros(len(self.positions), dtype=float)
        self.points["activity_rgba"] = np.zeros((len(self.positions), 4), dtype=np.uint8)
        self.neuron_actor = self.plotter.add_mesh(self.points, scalars="activity_rgba", rgb=True,
            point_size=9, render_points_as_spheres=True, name="neurons", show_scalar_bar=False,
            lighting=False, pickable=True)
        self.edge_actor = None
        if graph is not None:
            rows, cols = graph.nonzero()
            lines = np.column_stack((np.full(len(rows), 2), rows, cols)).ravel()
            edge_mesh = pv.PolyData(self.positions.copy(), lines=lines)
            self.edge_actor = self.plotter.add_mesh(edge_mesh, color="#496282", opacity=0.12, line_width=1,
                name="connections", pickable=False)
            self.edge_actor.SetVisibility(False)
        self.skeleton_actors = {}
        for root_id in getattr(anatomy, "skeletons", {}):
            vertices, edges = anatomy.read_skeleton(root_id)
            skeleton = pv.PolyData(np.asarray(vertices, dtype=np.float64),
                lines=np.column_stack((np.full(len(edges), 2), edges)).ravel())
            self.skeleton_actors[str(root_id)] = self.plotter.add_mesh(skeleton, color="#91c9d8", opacity=0.0,
                line_width=1, pickable=False)
        self.selection_actor = None
        if self.plotter.iren is not None:
            self.plotter.enable_point_picking(callback=self._picked, show_message=False, show_point=False,
                use_picker=True, picker="point", left_clicking=True, pickable_window=False)
        self.plotter.view_xy()
        self.plotter.reset_camera()
        self.plotter.camera.zoom(1.5)
        self._render()

    def _render(self):
        if getattr(self.plotter, "_closed", False):
            # A deferred resize render can arrive after the window has closed the plotter.
            return
        if self.headless:
            self.plotter.window_size = (max(500, self.image_label.width()), max(300, self.image_label.height()))
        self.plotter.render()
        if self.headless:
            pixels = np.ascontiguousarray(self.plotter.screenshot(return_img=True))
            image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QImage.Format_RGB888).copy()
            self.image_label.setPixmap(QPixmap.fromImage(image).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        if getattr(self, "headless", False) and hasattr(self, "points"):
            QTimer.singleShot(0, self._render)
        super().resizeEvent(event)

    def _picked(self, point, picker=None):
        if picker is not None and picker.GetDataSet() is self.points:
            index = int(picker.GetPointId())
        else:
            distance = np.linalg.norm(self.positions - np.asarray(point), axis=1)
            index = int(distance.argmin())
        if 0 <= index < len(self.positions):
            self.select_neuron(index)
            self.neuron_selected.emit(index)

    def select_neuron(self, index):
        if self.selection_actor is not None:
            self.plotter.remove_actor(self.selection_actor, render=False)
        self.selection_actor = self.plotter.add_mesh(pv.PolyData(self.positions[index:index + 1]), color="#f9e09c",
            point_size=17, render_points_as_spheres=True, lighting=False, pickable=False, name="selected")
        self._render()

    def update_state(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(self.positions),) or not np.isfinite(values).all():
            raise ValueError("Measured state does not match the anatomical neuron order")
        self.current_state = values.copy()
        amplitude = np.clip(np.abs(values) / self.activity_scale, 0, 1)
        rgba = np.zeros((len(values), 4), dtype=np.uint8)
        rgba[:, :3] = np.where((values >= 0)[:, None], np.array([255, 101, 94]), np.array([93, 164, 255]))
        rgba[:, 3] = np.rint(amplitude * 255).astype(np.uint8)
        self.points["activity_rgba"] = rgba
        for root_id, actor in self.skeleton_actors.items():
            index = self.anatomy.neuron_ids.index(root_id)
            actor.GetProperty().SetColor(*(rgba[index, :3] / 255))
            actor.GetProperty().SetOpacity(float(amplitude[index]) * 0.65)
        self.points.Modified()
        self._render()

    def set_activity_scale(self, value):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Activity display scale must be finite and positive")
        self.activity_scale = float(value)
        self.update_state(self.current_state)

    def set_connections_visible(self, visible):
        if self.edge_actor is not None:
            self.edge_actor.SetVisibility(bool(visible))
            self._render()

    def set_anatomy_visible(self, visible):
        for actor in self.mesh_actors:
            actor.SetVisibility(bool(visible))
        self._render()

    def reset_camera(self):
        self.plotter.reset_camera()
        self._render()

    def focus_neurons(self):
        self.plotter.reset_camera(bounds=self.points.bounds)
        self._render()

    def export_image(self, path: Path):
        self.plotter.screenshot(str(path))

    def closeEvent(self, event):
        self.plotter.close()
        super().closeEvent(event)
