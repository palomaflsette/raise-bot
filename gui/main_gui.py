import sys
import os
sys.path.append(os.path.abspath(".."))
from gui.assets import README_URL, ABOUT_TEXT, TITLE
from gui.controllers import start_system, save_capture, start_debug_mode, reset_robot, move_to_target
from gui.widgets import create_depth_slider
from gui.layout import create_section
from tkinter import *
import tkinter as tk
import customtkinter as ctk
from tkinter import Menu, messagebox
import webbrowser


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class RaiseGui(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.geometry("800x700")
        self.minsize(800, 600)
        self.title(TITLE)
        try:
            icon_path = "assets/icon-bot.ico"
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
            else:
                print(f"Ícone não encontrado: {icon_path}")
        except Exception as e:
            print(f"Erro ao definir ícone: {e}")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        container = ctk.CTkFrame(self)
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)

        canvas = tk.Canvas(container, bg="#2b2b2b", highlightthickness=0)
        scrollbar = tk.Scrollbar(
            container, orient="vertical", command=canvas.yview)
        scrollable_frame = ctk.CTkFrame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )


        scrollable_frame.grid_rowconfigure(0, weight=4, minsize=300)
        scrollable_frame.grid_rowconfigure(1, weight=3, minsize=250)
        scrollable_frame.grid_rowconfigure(
            2, weight=0)               # Separador
        scrollable_frame.grid_rowconfigure(3, weight=0, minsize=80)
        scrollable_frame.grid_columnconfigure(0, weight=1)

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)

        def configure_canvas(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas_width = event.width
            canvas.itemconfig(canvas.create_window(
                (0, 0), window=scrollable_frame, anchor="nw"), width=canvas_width)

        canvas.bind('<Configure>', configure_canvas)

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        canvas.bind("<MouseWheel>", on_mousewheel)

        self.menu_bar = Menu(self)
        self.config(menu=self.menu_bar)

        file_menu = Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.quit)
        self.menu_bar.add_cascade(label="File", menu=file_menu)

        help_menu = Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="Show README", command=self.open_readme)
        help_menu.add_command(label="About", command=self.show_about)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)

        self.frame_top = ctk.CTkFrame(scrollable_frame)
        self.frame_top.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)

        self.frame_top.grid_columnconfigure(0, weight=1)  # Coluna Depth
        self.frame_top.grid_columnconfigure(1, weight=1)  # Coluna RGB
        self.frame_top.grid_rowconfigure(0, weight=1)

        self.depth_frame, self.depth_canvas = create_section(
            self.frame_top, "Depth View", "(Depth image here)")
        self.depth_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.depth_frame.grid_columnconfigure(0, weight=1)
        self.depth_frame.grid_rowconfigure(
            1, weight=1)  # Canvas flexível verticalmente

        self.rgb_frame, self.rgb_canvas = create_section(
            self.frame_top, "RGB View", "(RGB image here)")
        self.rgb_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.rgb_frame.grid_columnconfigure(0, weight=1)
        self.rgb_frame.grid_rowconfigure(1, weight=1)

        self.frame_middle = ctk.CTkFrame(scrollable_frame)
        self.frame_middle.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        self.frame_middle.grid_columnconfigure(0, weight=1)
        self.frame_middle.grid_rowconfigure(
            0, weight=1)  

        self.normals_frame, self.normals_canvas = create_section(
            self.frame_middle, "Normals / Profile", "(Normals and graph info here)")
        self.normals_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.normals_frame.grid_columnconfigure(0, weight=1)
        self.normals_frame.grid_rowconfigure(1, weight=1)

        self.frame_bottom = ctk.CTkFrame(scrollable_frame)
        self.frame_bottom.grid(row=3, column=0, sticky="ew", padx=10, pady=5)

        self.frame_bottom.grid_columnconfigure(0, weight=1)
        self.frame_bottom.grid_columnconfigure(1, weight=1)
        self.frame_bottom.grid_columnconfigure(2, weight=1)
        self.frame_bottom.grid_columnconfigure(3, weight=1)
        self.frame_bottom.grid_rowconfigure(
            0, weight=0)  

        button_options = {
            "height": 40,
            "font": ("Arial", 12)
        }

        self.start_button = ctk.CTkButton(
            self.frame_bottom, text="Iniciar Sistema",
            command=lambda: start_system(self), **button_options)
        self.start_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.reset_button = ctk.CTkButton(
            self.frame_bottom, text="Resetar Robô",
            command=lambda: reset_robot(self), **button_options)
        self.reset_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.save_button = ctk.CTkButton(
            self.frame_bottom, text="Salvar Captura",
            command=lambda: save_capture(self), **button_options)
        self.save_button.grid(row=0, column=2, padx=10, pady=10, sticky="ew")

        self.debug_button = ctk.CTkButton(
            self.frame_bottom, text="Modo Debug",
            command=lambda: start_debug_mode(self), **button_options)
        self.debug_button.grid(row=0, column=3, padx=10, pady=10, sticky="ew")
        
        self.debug_button.grid(row=0, column=3, padx=10, pady=10, sticky="ew")

        # --- NOVO BOTÃO ---
        self.move_button = ctk.CTkButton(
            self.frame_bottom, text="MOVER ROBÔ PARA O ALVO",
            command=lambda: move_to_target(self),
            height=40,
            font=("Arial", 12, "bold"),
            fg_color="#2E8B57",  
            hover_color="#3CB371"
        )
        self.move_button.grid(row=1, column=0, columnspan=4,
                              padx=10, pady=10, sticky="ew")


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
