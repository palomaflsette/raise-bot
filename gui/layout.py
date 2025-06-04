""" 
organização dos frames e widgets 
"""
import customtkinter as ctk


def create_section(parent, title_text, placeholder_text):
    section_frame = ctk.CTkFrame(parent)

    title = ctk.CTkLabel(
        section_frame,
        text=title_text,
        font=("Helvetica", 14, "bold"),
        anchor="w",
        justify="left"
    )
    title.pack(pady=(5, 2))

    canvas = ctk.CTkLabel(
        section_frame,
        text=placeholder_text,
        width=420,
        height=240,
        text_color="gray"
    )
    canvas.pack(pady=30)

    return section_frame, canvas
