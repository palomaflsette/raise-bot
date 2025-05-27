"""
script principal da interface
"""

import webbrowser
from tkinter import Menu, messagebox
import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class RaiseGui(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # <- Tamanho inicial da janela (largura x altura)
        self.geometry("1325x800")
        self.minsize(1000, 600)
        
        self.title("RAISE - Robotic Acoustic Inspection with Surface Estimation")
        self.grid_columnconfigure((0, 1, 2), weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        

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

        self.depth_frame = self.create_section(
            self.frame_top, "Depth View", "(Imagem depth aqui)")
        self.depth_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.rgb_frame = self.create_section(
            self.frame_top, "RGB View", "(Imagem RGB aqui)")
        self.rgb_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.normals_frame = self.create_section(
            self.frame_top, "Normals / Profile", "(Imagem Normals aqui)")
        self.normals_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        # ---------- PARTE DO MEIO: SINAIS DINÂMICOS ----------
        self.frame_middle = ctk.CTkFrame(scrollable_frame, height=250)
        self.frame_middle.grid(row=1, column=0, columnspan=3,
                               sticky="nsew", padx=10, pady=5)
        self.frame_middle.grid_columnconfigure((0, 1, 2), weight=1)

        self.winding_frame = self.create_section(
            self.frame_middle, "Winding", "(Imagem Winding aqui)")
        self.winding_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.embedding_frame = self.create_section(
            self.frame_middle, "Embedding", "(Imagem Embedding aqui)")
        self.embedding_frame.grid(
            row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.rqa_frame = self.create_section(
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
        return section_frame

    def open_readme(self):
        webbrowser.open(
            "https://github.com/palomaflsette/raise-bot/blob/main/README.md")

    def show_about(self):
        messagebox.showinfo(
            "Sobre o projeto RAISE Bot",
            "RAISE: Robotic Acoustic Inspection with Surface Estimation\n\nDesenvolvido por Paloma Sette sob orientação de Wouter Caarls na PUC-Rio\n2025"
        )

    def start_system(self):
        print("Sistema iniciado (placeholder)")

    def reset_robot(self):
        print("Robô resetado (placeholder)")

    def save_capture(self):
        print("Captura salva (placeholder)")

    def toggle_debug(self):
        print("Modo debug ativado/desativado (placeholder)")


if __name__ == "__main__":
    app = RaiseGui()
    app.mainloop()
