""" 
botões e eventos da interface 
"""
import sys
import os
sys.path.append(os.path.abspath(".."))
from robot.pincher import Pincher
import depthai as dai
from config import ROBOT_SERIAL_PORT, CALIBRATION_MATRIX_FILE
from vision.camera_stream import start_camera_stream, start_candidate_thread
from vision.simulate_stream import start_simulated_stream
import threading
import numpy as np  
from time import sleep


def start_debug_mode(gui):
    print("Modo debug (simulado)")
    threading.Thread(target=lambda: start_simulated_stream(
        gui), daemon=True).start()



def start_system(gui):
    """
    Versão final que conecta ao robô e o prepara para a ação.
    """
    try:
        gui.T_cam_to_robo = np.loadtxt(CALIBRATION_MATRIX_FILE)
        print(f"Matriz de calibração '{CALIBRATION_MATRIX_FILE}' carregada.")
        print(f"Matriz T_cam_to_robot ==> {gui.T_cam_to_robo}")
    except IOError:
        print(
            f"ERRO FATAL: Arquivo '{CALIBRATION_MATRIX_FILE}' não encontrado!")
        return
    
    try:
        gui.pincher = Pincher(port=ROBOT_SERIAL_PORT)
        print(f"Conexão com o robô estabelecida na porta {ROBOT_SERIAL_PORT}.")

        sleep(2)

        gui.pincher.enable(gui.pincher.arm_ids)
        print("Motores do robô habilitados e prontos.")
    except Exception as e:
        print(
            f"[ERRO FATAL] Não foi possível conectar/habilitar o robô na {ROBOT_SERIAL_PORT}: {e}")
        return

    start_candidate_thread(gui)

    print("Sistema de visão iniciado. Procurando alvos...")
    threading.Thread(target=lambda: start_camera_stream(gui),
                     daemon=True).start()



def move_to_target(gui):
    """
    Versão final que lê a tupla do alvo na memória e move o robô.
    """
    if not hasattr(gui, 'last_known_point') or gui.last_known_point is None:
        print("[AVISO] Nenhum alvo na memória. Aguarde o alvo verde aparecer e estabilizar.")
        return

    print("--- COMANDO DE MOVIMENTO RECEBIDO ---")
    

    q_solution = gui.last_known_point[5]
    motor_ids = gui.pincher.arm_ids
    velocidade = 0.4

    print(f"Alvo em graus: {np.degrees(q_solution).round(1)}°")
    
    try:
        gui.pincher.enable(motor_ids)
        gui.pincher.move(motor_ids, q_solution, speed=velocidade)
        print(" Movimento CONCLUÍDO!")
    except Exception as e:
        print(f"[ERRO] Falha durante a execução do movimento: {e}")

def reset_robot(gui):
    """
    Reseta o robô para a posição home (todos os ângulos = 0).
    """
    if not hasattr(gui, 'pincher') or gui.pincher is None:
        print(
            "[ERRO] Sistema não foi inicializado. Clique em 'Iniciar Sistema' primeiro.")
        return

    print("--- RESETANDO ROBÔ PARA POSIÇÃO HOME ---")

    try:
        motor_ids = gui.pincher.arm_ids
        home_position = [0.0] * len(motor_ids)  

        gui.pincher.enable(motor_ids)
        gui.pincher.move(motor_ids, home_position, speed=0.2, margin=0.1)

        print("Robô resetado para posição home.")

    except Exception as e:
        print(f"[ERRO] Falha ao resetar robô: {e}")


def save_capture(gui):
         print("Função de salvar captura não implementada ainda.")


def toggle_debug(gui):
    print("Função de toggle debug não implementada ainda.")
