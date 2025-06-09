import sys
import os
sys.path.append(os.path.abspath(".."))

import webbrowser
from tkinter import Menu, messagebox
import customtkinter as ctk
import tkinter as tk
from tkinter import *
from gui.layout import create_section
from gui.widgets import create_depth_slider
from gui.controllers import start_system, save_capture, start_debug_mode, reset_robot
from gui.assets import README_URL, ABOUT_TEXT, TITLE


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class RaiseGui(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.geometry("1325x800")
        self.minsize(1000, 600)
        self.title(TITLE)
        try:
            icon_path = "assets/icon-bot.ico"
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
            else:
                print(f"Ícone não encontrado: {icon_path}")

        except Exception as e:
            print(f"Erro ao definir ícone: {e}")
        self.grid_columnconfigure((0, 1, 2), weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        
        self.min_depth = 100 # mm
        self.max_depth = 1000 # mm
        
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

        self.depth_frame, self.depth_canvas = create_section(
            self.frame_top, "Depth View", "(Imagem depth aqui)")


        self.depth_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.rgb_frame, self.rgb_canvas = create_section(
            self.frame_top, "RGB View", "(Imagem RGB aqui)")
        self.rgb_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")


        self.normals_frame, self.normals_canvas = create_section(
            self.frame_top, "Normals / Profile", "(Imagem Normals aqui)")
        self.normals_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        self.normals_canvas.grid_rowconfigure(0, weight=1)
        self.normals_canvas.grid_columnconfigure(0, weight=1)

        # ---------- PARTE DO MEIO: SINAIS DINÂMICOS ----------
        self.frame_middle = ctk.CTkFrame(scrollable_frame, height=250)
        self.frame_middle.grid(row=1, column=0, columnspan=3,
                               sticky="nsew", padx=10, pady=5)
        self.frame_middle.grid_columnconfigure((0, 1, 2), weight=1)

        self.winding_frame, self.winding_canvas = create_section(
            self.frame_middle, "Winding", "(Imagem Winding aqui)")
        self.winding_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.embedding_frame, self.embedding_canvas = create_section(
            self.frame_middle, "Embedding", "(Imagem Embedding aqui)")
        self.embedding_frame.grid(
            row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.rqa_frame, self.rqa_canvas = create_section(
            self.frame_middle, "RQA + Entropy", "(Imagem RQA aqui)")
        self.rqa_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        # ---------- PARTE INFERIOR: CONTROLES ----------
        self.frame_bottom = ctk.CTkFrame(scrollable_frame, height=100)
        self.frame_bottom.grid(row=2, column=0, columnspan=3,
                               sticky="nsew", padx=10, pady=5)
        self.frame_bottom.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.start_button = ctk.CTkButton(
            self.frame_bottom, text="Iniciar Sistema", command=lambda: start_system(self))
        self.start_button.grid(row=0, column=0, padx=10, pady=10)

        self.reset_button = ctk.CTkButton(
            self.frame_bottom, text="Resetar Robô", command=lambda: reset_robot(self))
        self.reset_button.grid(row=0, column=1, padx=10, pady=10)

        self.save_button = ctk.CTkButton(
            self.frame_bottom, text="Salvar Captura", command=lambda: save_capture(self))
        self.save_button.grid(row=0, column=2, padx=10, pady=10)

        self.debug_button = ctk.CTkButton(
            self.frame_bottom, text="Modo Debug", command=lambda: start_debug_mode(self))
        self.debug_button.grid(row=0, column=3, padx=10, pady=10)
        

    def open_readme(self):
        webbrowser.open(README_URL)

    def show_about(self):
        messagebox.showinfo(
            "Sobre o projeto",
            ABOUT_TEXT
        )



if __name__ == "__main__":
    app = RaiseGui()
    app.mainloop()
