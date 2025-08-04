#!/usr/bin/env python
"""
Smoke-test that VTK really uses the OSMesa software back-end.
Writes headless_test.png next to the script.
"""
import os, ctypes.util, numpy as np

# --- 1. tell VTK to pick the OSMesa window class ---------------------------
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkOSOpenGLRenderWindow"
# ---------------------------------------------------------------------------

import pyvista as pv, vtk

print("VTK version        :", vtk.vtkVersion().GetVTKVersion())
print("RenderWindow class :", vtk.vtkRenderWindow().GetClassName())
print("libOSMesa found at :", ctypes.util.find_library("OSMesa"))

# build a tiny 3-D numpy cube
cube = np.zeros((64, 64, 64), dtype=np.uint8)
cube[16:48, 16:48, 16:48] = 1

plotter = pv.Plotter(off_screen=True, window_size=[600, 500])
plotter.add_volume(cube, cmap="gray", opacity="sigmoid")
plotter.camera_position = "xy"
plotter.show(screenshot="headless_test.png")

print("✓ wrote headless_test.png")

