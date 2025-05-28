"""
script principal da interface
"""
import cv2
import sys
import os
sys.path.append(os.path.abspath(".."))
import webbrowser
import threading
from tkinter import Menu, messagebox
import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import depthai as dai
from vision.depth_stream import create_pipeline, filter_depth_range

import numpy as np
from PIL import Image, ImageTk
import numpy as np

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class RaiseGui(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.geometry("1325x800")
        self.minsize(1000, 600)
        self.title("RAISE - Robotic Acoustic Inspection with Surface Estimation")
        self.grid_columnconfigure((0, 1, 2), weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        
        # parametros de profundidade
        self.min_depth = 100 # mm
        self.max_depth = 10000 # mm
        
        # scroller
        container = ctk.CTkFrame(self)
        container.pack(fill="both", expand=True)
        canvas = tk.Canvas(container, bg="#2b2b2b", highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ctk.CTkFrame(canvas)
        scrollable_frame.bind(
          "<Configure>",
          lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
          )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


        # ---------- MENU SUPERIOR ----------
        self.menu_bar = Menu(self)
        self.config(menu=self.menu_bar)

        file_menu = Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.quit)
        self.menu_bar.add_cascade(label="File", menu=file_menu)

        help_menu = Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="Ver README", command=self.open_readme)
        help_menu.add_command(label="Sobre", command=self.show_about)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)

        # ---------- PARTE SUPERIOR: ROBÓTICA ----------
        self.frame_top = ctk.CTkFrame(scrollable_frame, height=300)
        self.frame_top.grid(row=0, column=0, columnspan=3,
                            sticky="nsew", padx=10, pady=5)
        self.frame_top.grid_columnconfigure((0, 1, 2), weight=1)

        self.depth_frame, self.depth_canvas = self.create_section(
            self.frame_top, "Depth View", "(Imagem depth aqui)")


        self.depth_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.rgb_frame, self.rgb_canvas = self.create_section(
            self.frame_top, "RGB View", "(Imagem RGB aqui)")
        self.rgb_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")


        self.normals_frame, self.normals_canvas = self.create_section(
            self.frame_top, "Normals / Profile", "(Imagem Normals aqui)")
        self.normals_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        # ---------- PARTE DO MEIO: SINAIS DINÂMICOS ----------
        self.frame_middle = ctk.CTkFrame(scrollable_frame, height=250)
        self.frame_middle.grid(row=1, column=0, columnspan=3,
                               sticky="nsew", padx=10, pady=5)
        self.frame_middle.grid_columnconfigure((0, 1, 2), weight=1)

        self.winding_frame, self.winding_canvas = self.create_section(
            self.frame_middle, "Winding", "(Imagem Winding aqui)")
        self.winding_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.embedding_frame, self.embedding_canvas = self.create_section(
            self.frame_middle, "Embedding", "(Imagem Embedding aqui)")
        self.embedding_frame.grid(
            row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.rqa_frame, self.rqa_canvas = self.create_section(
            self.frame_middle, "RQA + Entropy", "(Imagem RQA aqui)")
        self.rqa_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        # ---------- PARTE INFERIOR: CONTROLES ----------
        self.frame_bottom = ctk.CTkFrame(scrollable_frame, height=100)
        self.frame_bottom.grid(row=2, column=0, columnspan=3,
                               sticky="nsew", padx=10, pady=5)
        self.frame_bottom.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.start_button = ctk.CTkButton(
            self.frame_bottom, text="Iniciar Sistema", command=self.start_system)
        self.start_button.grid(row=0, column=0, padx=10, pady=10)

        self.reset_button = ctk.CTkButton(
            self.frame_bottom, text="Resetar Robô", command=self.reset_robot)
        self.reset_button.grid(row=0, column=1, padx=10, pady=10)

        self.save_button = ctk.CTkButton(
            self.frame_bottom, text="Salvar Captura", command=self.save_capture)
        self.save_button.grid(row=0, column=2, padx=10, pady=10)

        self.debug_button = ctk.CTkButton(
            self.frame_bottom, text="Modo Debug", command=self.toggle_debug)
        self.debug_button.grid(row=0, column=3, padx=10, pady=10)
        

        # Sliders de profundidade
        self.slider_min = ctk.CTkSlider(self.frame_bottom, from_=100, to=10000, number_of_steps=10,
                                        command=self.update_min_depth, width=200)
        self.slider_min.set(self.min_depth)
        self.slider_min.grid(row=1, column=0, columnspan=2, padx=10, pady=5)

        self.label_min = ctk.CTkLabel(
            self.frame_bottom, text=f"Min Depth: {self.min_depth} mm")
        self.label_min.grid(row=1, column=2, padx=5)

        self.slider_max = ctk.CTkSlider(self.frame_bottom, from_=100, to=10000, number_of_steps=10,
                                        command=self.update_max_depth, width=200)
        self.slider_max.set(self.max_depth)
        self.slider_max.grid(row=2, column=0, columnspan=2, padx=10, pady=5)

        self.label_max = ctk.CTkLabel(
            self.frame_bottom, text=f"Max Depth: {self.max_depth} mm")
        self.label_max.grid(row=2, column=2, padx=5)
        
        
    def update_min_depth(self, value):
        self.min_depth = int(value)
        self.label_min.configure(text=f"Min Depth: {self.min_depth} mm")

    def update_max_depth(self, value):
        self.max_depth = int(value)
        self.label_max.configure(text=f"Max Depth: {self.max_depth} mm")


    def render_profile_plot(self, depth_frame):
        depth_line = depth_frame[240, :]  # linha central
        
        fig, ax = plt.subplots(figsize=(4.2, 2.5), dpi=100)
        ax.plot(depth_line, color="blue")
        ax.set_title("Perfil da Superfície")
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Profundidade (u)")
        ax.grid(True)

        if hasattr(self, 'profile_canvas'):
            self.profile_canvas.get_tk_widget().grid_forget()
            self.profile_canvas.get_tk_widget().destroy()

        self.profile_canvas = FigureCanvasTkAgg(fig, master=self.normals_canvas)
        self.profile_canvas.draw()
        self.profile_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")



    def create_section(self, parent, title_text, placeholder_text):
        section_frame = ctk.CTkFrame(parent)
        title = ctk.CTkLabel(section_frame, 
                             text=title_text, 
                             font=("Helvetica", 14, "bold"), 
                             anchor="w",
                             justify="left")
        title.pack(pady=(5, 2))
        canvas = ctk.CTkLabel(
            section_frame, text=placeholder_text, width=420, height=240, text_color="gray")
        canvas.pack(pady=30)
        return section_frame, canvas

    def open_readme(self):
        webbrowser.open(
            "https://github.com/palomaflsette/raise-bot/blob/main/README.md")

    def show_about(self):
        messagebox.showinfo(
            "Sobre o projeto RAISE Bot",
            "RAISE: Robotic Acoustic Inspection with Surface Estimation\n\nDesenvolvido por Paloma Sette sob orientação de Wouter Caarls na PUC-Rio\n2025"
        )

    def start_system(self):
        print("Sistema iniciado")
        threading.Thread(target=self.start_camera_stream, daemon=True).start()

    def updateArg(self, arg_name, arg_value, shouldUpdate=True):
        setattr(self.confManager.args, arg_name, arg_value)
        if shouldUpdate:
            self.worker.signals.setDataSignal.emit(
                    ["restartRequired", True])
            print("CONFMANAGER ------->" + str(self.confManager.args))

    def start_camera_stream(self):
        pipeline = create_pipeline()
        self.device = dai.Device(pipeline)
        self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        self.update_camera_frames()

    def update_camera_frames(self):
        # RGB
        in_rgb = self.rgb_queue.tryGet()
        if in_rgb:
            rgb_frame = in_rgb.getCvFrame()
            img = Image.fromarray(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img.resize((440, 350)))
            #imgtk = ctk.CTkImage(Image.fromarray(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)), size=(390, 300))
            self.rgb_canvas.configure(image=imgtk, text="")
            self.rgb_canvas.image = imgtk

        # Depth
        in_depth = self.depth_queue.tryGet()
        if in_depth:
            depth_frame = in_depth.getFrame()
            depth_frame = filter_depth_range(depth_frame)
            depth_frame = depth_frame.astype(np.uint16)
            
            # self.render_profile_plot(depth_frame) # na label de Profile/Normals, vai atualizando o perfil
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.1), cv2.COLORMAP_JET)

            # Clipa e normaliza manualmente para 0–255
            depth_clip = np.clip(depth_frame, self.min_depth, self.max_depth)
            depth_norm = ((depth_clip - self.min_depth) /
                          (self.max_depth - self.min_depth) * 255).astype(np.uint8)

            # Aplica o colormap
            depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

            img = Image.fromarray(depth_colormap)
            imgtk = ImageTk.PhotoImage(image=img.resize((440, 350)))
            #imgtk = ctk.CTkImage(Image.fromarray(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)), size=(390, 300))
            self.depth_canvas.configure(image=imgtk, text="")
            self.depth_canvas.image = imgtk
            



        self.after(30, self.update_camera_frames)  # Atualiza a cada 30ms (~30FPS)


    def reset_robot(self):
        print("Robô resetado (placeholder)")

    def save_capture(self):
        print("Captura salva (placeholder)")

    def toggle_debug(self):
        print("Modo debug ativado/desativado (placeholder)")


if __name__ == "__main__":
    app = RaiseGui()
    app.mainloop()
