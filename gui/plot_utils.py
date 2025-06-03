from io import BytesIO
from PIL import Image
import numpy as np
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use("Agg")



def render_profile_plot(depth_frame, target_widget, parent_gui):
    depth_line = depth_frame[240, :]  # linha central

    fig, ax = plt.subplots(figsize=(5.5, 2.5), dpi=100)
    ax.plot(depth_line, color="cyan")
    ax.set_title("Perfil da Superfície (Profundidade)")
    ax.set_xlabel("Pixel (X)")
    ax.set_ylabel("Distância da Câmera (Z)")
    ax.grid(True)

    # Cálculo ---> normais
    dx = np.gradient(depth_line)
    normals = np.stack([-dx, np.ones_like(dx)], axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    ax.quiver(
        np.arange(len(depth_line)),
        depth_line,
        normals[:, 0],
        -normals[:, 1],
        color="red", scale=20, width=0.002
    )

    if hasattr(parent_gui, 'profile_canvas'):
        parent_gui.profile_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    # parent_gui.profile_canvas = FigureCanvasTkAgg(fig, master=target_widget)
    # parent_gui.profile_canvas.draw()
    # parent_gui.profile_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)

    imgtk = ctk.CTkImage(light_image=img, size=(440, 200))
    target_widget.configure(image=imgtk, text="")
    target_widget.image = imgtk

    plt.close(fig)
