import depthai as dai
import cv2
import numpy as np
from PIL import Image, ImageTk
from vision.depth_stream import create_pipeline, filter_depth_range
from gui.plot_utils import render_profile_plot


def start_camera_stream(gui):
    pipeline = create_pipeline()
    gui.device = dai.Device(pipeline)
    gui.rgb_queue = gui.device.getOutputQueue(
        name="rgb", maxSize=4, blocking=False)
    gui.depth_queue = gui.device.getOutputQueue(
        name="depth", maxSize=4, blocking=False)

    update_camera_frames(gui)


def update_camera_frames(gui):
    # RGB
    in_rgb = gui.rgb_queue.tryGet()
    if in_rgb:
        rgb_frame = in_rgb.getCvFrame()
        img = Image.fromarray(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img.resize((440, 350)))
        gui.rgb_canvas.configure(image=imgtk, text="")
        gui.rgb_canvas.image = imgtk

    # Depth
    in_depth = gui.depth_queue.tryGet()
    if in_depth:
        depth_frame = in_depth.getFrame()
        depth_frame = filter_depth_range(depth_frame)
        depth_frame = depth_frame.astype(np.uint16)

        render_profile_plot(depth_frame, gui.normals_canvas, gui)

        # Normaliza para exibição
        depth_clip = np.clip(depth_frame, gui.min_depth, gui.max_depth)
        depth_norm = ((depth_clip - gui.min_depth) /
                      (gui.max_depth - gui.min_depth) * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        img = Image.fromarray(depth_colormap)
        imgtk = ImageTk.PhotoImage(image=img.resize((440, 350)))
        gui.depth_canvas.configure(image=imgtk, text="")
        gui.depth_canvas.image = imgtk

    gui.after(30, lambda: update_camera_frames(gui))
