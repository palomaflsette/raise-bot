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


# Em gui/controllers.py

def move_to_target(gui):
    """
    Pega o último alvo válido encontrado e envia o comando de movimento
    para o robô Pincher usando a função correta 'move'.
    """
    # Verifica se existe um alvo válido guardado na "memória"
    if not hasattr(gui, 'last_known_point') or gui.last_known_point is None:
        print(
            "[ERRO] Nenhum alvo estável encontrado ainda. Tente posicionar melhor o objeto.")
        return

    print("--- COMANDO DE MOVIMENTO INICIADO ---")

    # Pega a solução dos motores (q) que foi guardada
    try:
        # A estrutura que definimos antes é um dicionário
        q_solution = gui.last_known_point["q_solution"]
    except (KeyError, TypeError):
        print("[ERRO] O alvo guardado não contém a solução dos motores ('q_solution').")
        return

    # Pega os IDs dos motores que definimos no Pincher
    motor_ids = gui.pincher.arm_ids

    # Define uma velocidade para o movimento (em rad/s) para não ser muito brusco
    velocidade = 0.5

    print(
        f"Movendo os motores {motor_ids} para os ângulos (rad): {q_solution} com velocidade {velocidade} rad/s")

    try:
        # --- CHAMADA CORRETA DA FUNÇÃO DE MOVIMENTO ---
        gui.pincher.move(motor_ids, q_solution, speed=velocidade)
        print("✅ Movimento concluído!")
    except Exception as e:
        print(f"[ERRO] Falha ao enviar comando para o robô: {e}")
        
        
def reset_robot(gui): pass
def save_capture(gui): pass
def toggle_debug(gui): pass
