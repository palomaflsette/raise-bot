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
import depthai as dai

def start_system(gui):
    """
    Versão corrigida: Apenas prepara os objetos e delega a conexão da
    câmera para a thread do camera_stream.
    """
    gui.pincher = Pincher()

    if not hasattr(gui, "last_valid_point"):
        gui.last_valid_point = None

    start_candidate_thread(gui)

    print("Sistema iniciado. Aguardando conexão da câmera...")
    
    threading.Thread(target=lambda: start_camera_stream(gui), daemon=True).start()


def start_debug_mode(gui):
    print("Modo debug (simulado)")
    threading.Thread(target=lambda: start_simulated_stream(
        gui), daemon=True).start()


def reset_robot(gui): pass
def save_capture(gui): pass
def toggle_debug(gui): pass
