import customtkinter as ctk


def create_depth_slider(parent, label_text, initial_value, command):
    slider = ctk.CTkSlider(parent, from_=100, to=10000, number_of_steps=10,
                           command=command, width=200)
    slider.set(initial_value)
    label = ctk.CTkLabel(parent, text=f"{label_text}: {initial_value} mm")
    return slider, label
