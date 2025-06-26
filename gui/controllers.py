""" 
botões e eventos da interface 
"""
import sys
import os
sys.path.append(os.path.abspath(".."))

from robot.pincher import Pincher 

import threading
from vision.simulate_stream import start_simulated_stream
from vision.camera_stream import start_camera_stream, start_candidate_thread

def start_system(gui):
    gui.pincher = Pincher()

    # Inicializa ponto salvo
    if not hasattr(gui, "last_valid_point"):
        gui.last_valid_point = None

    # Inicia thread de candidatos, com last_valid_point passado
    start_candidate_thread(gui)

    print("Sistema iniciado")
    threading.Thread(target=lambda: start_camera_stream(gui), daemon=True).start()


def start_debug_mode(gui):
    print("Modo debug (simulado)")
    threading.Thread(target=lambda: start_simulated_stream(
        gui), daemon=True).start()


def reset_robot(gui): pass
def save_capture(gui): pass
def toggle_debug(gui): pass
