"""
app.py
======

UX layer for the 3DGS alignment engine (Milestone 1).

This is a PyQt5 single-window application that ties the three other modules
together:

    File ▸ Open .ply        — invokes `loader.load_point_cloud(...)`
    Auto Align              — invokes `AlignmentEngine().align(...)`
    Align Up Only           — keeps the auto up-axis, zeroes yaw
    Rotate 90° / Flip       — manual nudges on top of the current transform
    Save Alignment          — writes the 4×4 matrix + metadata as JSON next
                              to the .ply
    Reset                   — drops the transform back to identity

The viewport is driven by `VulkanRenderer`, which prefers the integrated
GPU and gracefully falls back to a NumPy software rasteriser when Vulkan
init fails.  Mouse drag orbits the camera, wheel zooms.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from typing import Optional

import numpy as np

# Qt is imported lazily here so that a failure in PyQt5 doesn't break
# loader/alignment imports for headless callers.
from PyQt5 import QtCore, QtGui, QtWidgets

from loader import PointCloud, load_point_cloud, DEFAULT_POINT_CAP
from alignment import (AlignmentEngine, AlignmentResult, AlignmentSession,
                       UP_AXIS_WORLD)
from renderer import VulkanRenderer


VIEW_W, VIEW_H = 900, 700
GIZMO_PX = 90                                # axis-gizmo footprint in pixels


# ---------------------------------------------------------------------------
# Viewport widget — owns the VulkanRenderer
# ---------------------------------------------------------------------------

class Viewport(QtWidgets.QWidget):
    """
    Repaints from a VulkanRenderer-produced RGBA buffer.

    Mouse drag orbits, right-drag pans, wheel zooms.  An axis gizmo is
    painted over the bottom-right corner.
    """

    transformChanged = QtCore.pyqtSignal()       # forwarded to status bar

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 320)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(True)

        self._renderer = VulkanRenderer(VIEW_W, VIEW_H)
        self._cloud: Optional[PointCloud] = None
        self._last_mouse: Optional[QtCore.QPoint] = None
        self._dragging = False
        self._needs_redraw = True
        self._fps_t = time.time()
        self._fps_frames = 0
        self._fps_value = 0.0
        self._preview_active = False             # draw a PREVIEW badge when True

        # Drive repaints from a timer so the viewport stays interactive
        # without blocking the GUI thread.
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)              # ~30 Hz cap
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ----- public API used by MainWindow -------------------------------------

    def set_point_cloud(self, pc: PointCloud) -> None:
        self._cloud = pc
        self._renderer.set_points(pc.positions, pc.colors)
        # The renderer auto-frames; nudge the camera to a slight elevation.
        self._renderer.set_view(yaw_deg=35.0, pitch_deg=22.0)
        self.request_redraw()

    def set_scene_transform(self, m4: np.ndarray) -> None:
        self._renderer.set_scene_root_transform(m4)
        self.request_redraw()
        self.transformChanged.emit()

    def scene_transform(self) -> np.ndarray:
        return self._renderer.scene_root.copy()

    def renderer_info(self) -> dict:
        return self._renderer.info()

    def fps(self) -> float:
        return self._fps_value

    def set_preview_active(self, on: bool) -> None:
        self._preview_active = bool(on)
        self.request_redraw()

    def probe_fps(self, frames: int = 60) -> float:
        """Render `frames` back-to-back and return the measured FPS.

        Used to quantify the load-time FPS drop ("no meaningful FPS drop after
        load"): call it with an empty renderer for the baseline, and again
        after `set_point_cloud` for the loaded figure.
        """
        r = self._renderer
        try:
            r.render()                           # warm up / prime pipelines
            t0 = time.perf_counter()
            for _ in range(max(1, frames)):
                r.render()
            dt = time.perf_counter() - t0
            return float(frames / dt) if dt > 0 else 0.0
        except Exception:
            return 0.0

    def request_redraw(self) -> None:
        self._needs_redraw = True
        self.update()

    def close_renderer(self) -> None:
        try:
            self._renderer.close()
        except Exception:
            pass

    # ----- input handling ----------------------------------------------------

    def mousePressEvent(self, ev):
        if ev.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            self._dragging = True
            self._last_mouse = ev.pos()

    def mouseReleaseEvent(self, ev):
        self._dragging = False
        self._last_mouse = None

    def mouseMoveEvent(self, ev):
        if not self._dragging or self._last_mouse is None:
            return
        dx = ev.x() - self._last_mouse.x()
        dy = ev.y() - self._last_mouse.y()
        self._last_mouse = ev.pos()
        if ev.buttons() & QtCore.Qt.LeftButton:
            cam = self._renderer.camera
            cam.yaw_deg = (cam.yaw_deg - dx * 0.4) % 360.0
            cam.pitch_deg = max(-85.0, min(85.0, cam.pitch_deg - dy * 0.3))
            self.request_redraw()

    def wheelEvent(self, ev):
        cam = self._renderer.camera
        steps = ev.angleDelta().y() / 120.0
        cam.distance = max(0.05, cam.distance * (0.9 ** steps))
        self.request_redraw()

    # ----- rendering ---------------------------------------------------------

    def resizeEvent(self, ev):
        w = max(1, self.width())
        h = max(1, self.height())
        self._renderer.resize(w, h)
        self.request_redraw()
        super().resizeEvent(ev)

    def _tick(self) -> None:
        if self._needs_redraw:
            self.update()

    def paintEvent(self, ev):
        self._needs_redraw = False
        try:
            img = self._renderer.render()
        except Exception:
            # Render error - paint a black background and a short message.
            painter = QtGui.QPainter(self)
            painter.fillRect(self.rect(), QtCore.Qt.black)
            painter.setPen(QtGui.QPen(QtCore.Qt.red))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter,
                             "render error — see console")
            painter.end()
            traceback.print_exc()
            return

        h, w, _ = img.shape
        qimg = QtGui.QImage(img.tobytes(), w, h, w * 4,
                            QtGui.QImage.Format_RGBA8888)
        painter = QtGui.QPainter(self)
        painter.drawImage(self.rect(), qimg)
        self._paint_axis_gizmo(painter)
        self._paint_hud(painter)
        painter.end()

        # FPS — measured over a rolling 1-second window.
        self._fps_frames += 1
        now = time.time()
        dt = now - self._fps_t
        if dt >= 1.0:
            self._fps_value = self._fps_frames / dt
            self._fps_frames = 0
            self._fps_t = now

    # ----- HUD + axis gizmo --------------------------------------------------

    def _paint_axis_gizmo(self, painter: QtGui.QPainter) -> None:
        """
        Draw a tiny 3-axis indicator that respects the current scene-root
        transform.  Helps the user see when the auto-align engine has
        rotated the cloud.

        Axes are projected through (view ⋅ sceneRoot) only — perspective is
        deliberately skipped so the gizmo stays a constant on-screen size.
        """
        cx = self.width() - GIZMO_PX // 2 - 14
        cy = self.height() - GIZMO_PX // 2 - 14

        # Background disc.
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(0, 0, 0, 110))
        painter.drawEllipse(QtCore.QPoint(cx, cy), GIZMO_PX // 2, GIZMO_PX // 2)

        # View matrix without perspective — just (view · sceneRoot) for axes.
        VR = self._renderer.camera.view() @ self._renderer.scene_root
        axes = np.eye(3, dtype=np.float32) * (GIZMO_PX * 0.42)
        # Project as if camera at origin (skip projection, just rotate).
        rotated = (VR[:3, :3] @ axes.T).T          # (3,3)

        labels = ["X", "Y", "Z"]
        colors = [QtGui.QColor(220, 80, 80),
                  QtGui.QColor(80, 200, 80),
                  QtGui.QColor(80, 150, 230)]
        # Painter Y axis is flipped — invert Y to match world up = +Y.
        for i, (vec, col, lab) in enumerate(zip(rotated, colors, labels)):
            ex = cx + vec[0]
            ey = cy - vec[1]
            painter.setPen(QtGui.QPen(col, 2.4))
            painter.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(ex, ey))
            painter.setPen(QtGui.QPen(col, 1))
            painter.drawText(QtCore.QPointF(ex + 4, ey - 4), lab)

    def _paint_hud(self, painter: QtGui.QPainter) -> None:
        info = self._renderer.info()
        n = info.get("points", 0)
        device = info.get("device_name", "?")
        device_type = info.get("device_type", "?")
        fps = self._fps_value
        line1 = f"device: {device}  ({device_type})"
        line2 = f"points: {n:,}    {fps:5.1f} fps"
        painter.setPen(QtGui.QColor(220, 220, 220))
        painter.fillRect(QtCore.QRect(0, 0, 360, 38),
                         QtGui.QColor(0, 0, 0, 130))
        painter.drawText(QtCore.QPoint(10, 16), line1)
        painter.drawText(QtCore.QPoint(10, 32), line2)

        # PREVIEW badge — makes it unmistakable that the shown alignment is
        # pending and has not been committed yet.
        if self._preview_active:
            badge = QtCore.QRect(self.width() // 2 - 70, 10, 140, 26)
            painter.fillRect(badge, QtGui.QColor(190, 130, 20, 200))
            painter.setPen(QtGui.QColor(20, 20, 20))
            painter.drawText(badge, QtCore.Qt.AlignCenter, "PREVIEW — not applied")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("3DGS Alignment — Milestone 1")
        self.resize(1200, 760)

        self._session = AlignmentSession()
        self._cloud: Optional[PointCloud] = None
        self._loaded_path: Optional[str] = None
        # FPS-drop tracking: baseline is the empty-scene framerate, measured
        # once at startup; per-load FPS is probed after each cloud loads.
        self._baseline_fps: float = 0.0
        self._loaded_fps: float = 0.0

        # central layout: viewport | side panel
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        self.viewport = Viewport(self)
        self.viewport.transformChanged.connect(self._refresh_status)
        h.addWidget(self.viewport, stretch=1)

        h.addWidget(self._build_side_panel())

        self._build_menu()
        self._status = self.statusBar()

        # Baseline FPS on the empty scene — the reference for the "no meaningful
        # FPS drop after load" criterion.  Probed once, before any cloud.
        self._baseline_fps = self.viewport.probe_fps(frames=40)
        self._refresh_status()

    # ----- side panel --------------------------------------------------------

    def _build_side_panel(self) -> QtWidgets.QWidget:
        side = QtWidgets.QWidget(self)
        side.setFixedWidth(260)
        side.setStyleSheet("QWidget { background-color: #1c1f25; color: #ddd; }"
                           "QPushButton { padding: 8px; font-weight: 500; }"
                           "QLabel#section { font-weight: bold; color: #bcd; }")
        v = QtWidgets.QVBoxLayout(side)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(8)

        sec1 = QtWidgets.QLabel("Cloud"); sec1.setObjectName("section")
        v.addWidget(sec1)

        btn_open = QtWidgets.QPushButton("Open .ply…")
        btn_open.clicked.connect(self.action_open_ply)
        v.addWidget(btn_open)

        self.cloud_label = QtWidgets.QLabel("no cloud loaded")
        self.cloud_label.setWordWrap(True)
        v.addWidget(self.cloud_label)

        v.addSpacing(8)
        sec2 = QtWidgets.QLabel("Alignment"); sec2.setObjectName("section")
        v.addWidget(sec2)

        btn_preview = QtWidgets.QPushButton("Preview Align")
        btn_preview.clicked.connect(self.action_preview_align)
        v.addWidget(btn_preview)

        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.clicked.connect(self.action_apply)
        v.addWidget(btn_apply)

        btn_up = QtWidgets.QPushButton("Align Up Only")
        btn_up.clicked.connect(self.action_align_up_only)
        v.addWidget(btn_up)

        btn_rot = QtWidgets.QPushButton("Rotate 90°")
        btn_rot.clicked.connect(self.action_rotate_90)
        v.addWidget(btn_rot)

        btn_flip = QtWidgets.QPushButton("Flip Front/Back")
        btn_flip.clicked.connect(self.action_flip_front_back)
        v.addWidget(btn_flip)

        btn_reset = QtWidgets.QPushButton("Reset")
        btn_reset.clicked.connect(self.action_reset)
        v.addWidget(btn_reset)

        v.addSpacing(8)
        sec3 = QtWidgets.QLabel("Persistence"); sec3.setObjectName("section")
        v.addWidget(sec3)

        btn_save = QtWidgets.QPushButton("Save Alignment + History")
        btn_save.clicked.connect(self.action_save_alignment)
        v.addWidget(btn_save)

        self.history_label = QtWidgets.QLabel("changes: 0")
        self.history_label.setStyleSheet("color: #8a9;")
        v.addWidget(self.history_label)

        v.addStretch(1)

        self.diagnostics_label = QtWidgets.QLabel("")
        self.diagnostics_label.setWordWrap(True)
        self.diagnostics_label.setStyleSheet("color: #9ab;")
        v.addWidget(self.diagnostics_label)

        return side

    def _build_menu(self) -> None:
        m_file = self.menuBar().addMenu("&File")
        a_open = m_file.addAction("&Open .ply…")
        a_open.setShortcut("Ctrl+O")
        a_open.triggered.connect(self.action_open_ply)
        a_save = m_file.addAction("&Save alignment")
        a_save.setShortcut("Ctrl+S")
        a_save.triggered.connect(self.action_save_alignment)
        m_file.addSeparator()
        a_quit = m_file.addAction("&Quit")
        a_quit.setShortcut("Ctrl+Q")
        a_quit.triggered.connect(self.close)

    # ----- actions -----------------------------------------------------------

    def action_open_ply(self) -> None:
        start_dir = os.path.dirname(self._loaded_path or "") or os.getcwd()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open PLY point cloud", start_dir,
            "PLY point cloud (*.ply);;All files (*)"
        )
        if not path:
            return
        try:
            pc = load_point_cloud(path, point_cap=DEFAULT_POINT_CAP)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Load failed", f"Could not load {path}\n\n{exc}"
            )
            return
        self._cloud = pc
        self._loaded_path = path
        self._session = AlignmentSession()       # fresh history per cloud
        self.viewport.set_point_cloud(pc)
        self._sync_viewport()
        # Probe the loaded-scene FPS so we can report the load-time drop.
        self._loaded_fps = self.viewport.probe_fps(frames=40)
        self._refresh_cloud_label()
        self._refresh_history()
        self._refresh_status(f"loaded {os.path.basename(path)} — "
                             f"{pc.n:,} pts (raw {pc.raw_point_count:,})  ·  "
                             f"{self._fps_drop_note()}")
        self.diagnostics_label.setText("")

    # ----- preview / apply ---------------------------------------------------

    def action_preview_align(self) -> None:
        """Compute an auto-alignment and *stage* it (preview, not committed)."""
        if self._cloud is None:
            self._refresh_status("load a .ply first")
            return
        try:
            result = self._session.preview_auto(self._cloud.positions)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Alignment failed", f"{exc}")
            return
        self._sync_viewport()
        print(f"[App] transform:\n{result.transform}")
        print(f"[App] diagnostics: {result.diagnostics}")
        flags = f"  flags: {','.join(result.ambiguity_flags)}" if result.ambiguity_flags else ""
        self._refresh_status(
            f"PREVIEW: mode={result.mode}  conf={result.confidence:.2f}{flags}"
            f"  — press Apply to commit"
        )
        self._refresh_diagnostics()
        self._refresh_history()

    # Back-compat alias: headless tests / CLI may still call action_auto_align.
    action_auto_align = action_preview_align

    def action_apply(self) -> None:
        if not self._session.has_preview():
            self._refresh_status("nothing to apply — Preview Align first")
            return
        self._session.apply()
        self._sync_viewport()
        self._refresh_status("applied — alignment committed")
        self._refresh_history()

    def action_align_up_only(self) -> None:
        if self._cloud is None:
            self._refresh_status("load a .ply first")
            return
        # If nothing staged or committed yet, compute a preview first.
        if self._session.is_identity() and not self._session.has_preview():
            self._session.preview_auto(self._cloud.positions)
        self._session.nudge("up_only")
        self._sync_viewport()
        self._refresh_status("kept up-axis, dropped yaw")
        self._refresh_history()

    def action_rotate_90(self) -> None:
        if self._cloud is None:
            self._refresh_status("load a .ply first")
            return
        self._session.nudge("rotate_90")
        self._sync_viewport()
        self._refresh_status("rotated 90° about up")
        self._refresh_history()

    def action_flip_front_back(self) -> None:
        if self._cloud is None:
            self._refresh_status("load a .ply first")
            return
        self._session.nudge("flip")
        self._sync_viewport()
        self._refresh_status("flipped front/back")
        self._refresh_history()

    def action_reset(self) -> None:
        self._session.reset()
        self._sync_viewport()
        self._refresh_status("transform reset")
        self._refresh_history()

    def action_save_alignment(self) -> None:
        if self._cloud is None or self._loaded_path is None:
            self._refresh_status("load a .ply first")
            return
        default = os.path.splitext(self._loaded_path)[0] + "_alignment.json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save alignment", default, "JSON (*.json);;All files (*)"
        )
        if not path:
            return
        source = {
            "source_ply": self._loaded_path,
            "raw_points": self._cloud.raw_point_count,
            "kept_points": self._cloud.n,
            "baseline_fps": self._baseline_fps,
            "loaded_fps": self._loaded_fps,
        }
        try:
            self._session.save(path, source=source)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", f"{exc}")
            return
        self._refresh_status(f"saved alignment + {len(self._session.history)} "
                             f"change(s) to {path}")

    # ----- helpers -----------------------------------------------------------

    def _refresh_cloud_label(self) -> None:
        if self._cloud is None:
            self.cloud_label.setText("no cloud loaded")
            return
        s = self._cloud.stats
        gs = " (3DGS)" if self._cloud.is_gaussian_splat else ""
        vox = f", voxel={self._cloud.used_voxel_size:.4f}" if self._cloud.used_voxel_size else ""
        self.cloud_label.setText(
            f"{os.path.basename(self._cloud.source_path)}{gs}\n"
            f"points: {self._cloud.n:,} / {self._cloud.raw_point_count:,}{vox}\n"
            f"extent: {s.extent[0]:.2f} × {s.extent[1]:.2f} × {s.extent[2]:.2f}\n"
            f"scale: {s.scale:.3f}"
        )

    def _sync_viewport(self) -> None:
        """Push the session's current transform + preview state to the viewport."""
        self.viewport.set_scene_transform(self._session.current())
        self.viewport.set_preview_active(self._session.has_preview())

    def _fps_drop_note(self) -> str:
        base = self._baseline_fps
        loaded = self._loaded_fps
        if base <= 0 or loaded <= 0:
            return "fps drop: n/a"
        drop = max(0.0, (base - loaded) / base) * 100.0
        verdict = "OK" if loaded >= 30.0 else "LOW"
        return (f"fps {loaded:.0f} vs {base:.0f} empty "
                f"(drop {drop:.0f}%, ≥30 {verdict})")

    def _refresh_history(self) -> None:
        n = len(self._session.history)
        last = self._session.history[-1].action if n else "—"
        self.history_label.setText(f"changes: {n}   (last: {last})")

    def _refresh_diagnostics(self) -> None:
        res = self._session.last_result
        if res is None:
            self.diagnostics_label.setText("")
            return
        d = res.diagnostics
        bits = [f"mode: {res.mode}",
                f"confidence: {res.confidence:.2f}"]
        if res.ambiguity_flags:
            bits.append("flags: " + ", ".join(res.ambiguity_flags))
        if "plane_inlier_ratio" in d:
            bits.append(f"plane inliers: {d['plane_inlier_ratio']*100:.0f}%")
        if d.get("wall_yawed", 0.0):
            bits.append("heading: squared to wall")
        bits.append(self._fps_drop_note())
        self.diagnostics_label.setText("\n".join(bits))

    def _refresh_status(self, msg: Optional[str] = None) -> None:
        info = self.viewport.renderer_info()
        engine = "Vulkan" if info.get("is_vulkan") else "CPU"
        device = info.get("device_name", "?")
        base = f"{engine} ({device})"
        if msg:
            self._status.showMessage(f"{msg}    |    {base}")
        else:
            self._status.showMessage(base)

    def closeEvent(self, ev):
        self.viewport.close_renderer()
        ev.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("3dgs-aligner-m1")
    w = MainWindow()
    w.show()

    # Open a file passed on the command line.
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        try:
            print(f"\n{'='*60}")
            pc = load_point_cloud(sys.argv[1])
            print(f"{'='*60}\n")
            w._cloud = pc
            w._loaded_path = sys.argv[1]
            w._session = AlignmentSession()
            w.viewport.set_point_cloud(pc)
            w._sync_viewport()
            w._loaded_fps = w.viewport.probe_fps(frames=40)
            w._refresh_cloud_label()
            w._refresh_history()
            w._refresh_status(f"loaded {os.path.basename(sys.argv[1])}  ·  "
                              f"{w._fps_drop_note()}")
        except Exception:
            traceback.print_exc()

    rc = app.exec_()
    # The Python `vulkan` cffi wrapper can segfault during interpreter
    # finalisation (driver/ICD teardown race); calling os._exit avoids it.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
