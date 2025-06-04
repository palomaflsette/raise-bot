import cv2
import numpy as np
from PIL import Image, ImageTk
from gui.plot_utils import render_profile_plot
from customtkinter import CTkImage


def start_simulated_stream(gui):
    print("Modo simulado iniciado")
    gui.rgb_queue = None
    gui.depth_queue = None
    update_simulated_frames(gui)


def update_simulated_frames(gui):
    # Gera imagem RGB fake (gradiente colorido)
    rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(3):
        rgb_frame[..., i] = np.linspace(0, 255, 640, dtype=np.uint8)

    img_rgb = Image.fromarray(rgb_frame)
    #imgtk_rgb = ImageTk.PhotoImage(image=img_rgb.resize((440, 350)))
    imgtk_rgb = CTkImage(light_image=img_rgb, size=(440, 350))
    gui.rgb_canvas.configure(image=imgtk_rgb, text="")
    gui.rgb_canvas.image = imgtk_rgb

    # Gera mapa de profundidade fake (senoides)
    x = np.linspace(0, 2 * np.pi, 640)
    z = 3000 + 1000 * np.sin(3 * x)
    depth_frame = np.tile(z, (480, 1)).astype(np.uint16)

    render_profile_plot(depth_frame, gui.normals_canvas, gui)

    depth_clip = np.clip(depth_frame, gui.min_depth, gui.max_depth)
    depth_norm = ((depth_clip - gui.min_depth) /
                  (gui.max_depth - gui.min_depth) * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    img_depth = Image.fromarray(depth_colormap)
    #imgtk_depth = ImageTk.PhotoImage(image=img_depth.resize((440, 350)))
    imgtk_depth = CTkImage(light_image=img_depth, size=(440, 350))
    gui.depth_canvas.configure(image=imgtk_depth, text="")
    gui.depth_canvas.image = imgtk_depth

    gui.after(30, lambda: update_simulated_frames(gui))
