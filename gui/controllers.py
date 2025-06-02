""" 
botões e eventos da interface 
"""
import sys
import os
sys.path.append(os.path.abspath(".."))

import threading
from vision.camera_stream import start_camera_stream
from vision.simulate_stream import start_simulated_stream


def start_debug_mode(gui):
    print("Modo debug (simulado)")
    threading.Thread(target=lambda: start_simulated_stream(
        gui), daemon=True).start()

def start_system(gui):
    print("Sistema iniciado")
    threading.Thread(target=lambda: start_camera_stream(gui),
                     daemon=True).start()

def reset_robot(gui): pass
def save_capture(gui): pass
def toggle_debug(gui): pass
