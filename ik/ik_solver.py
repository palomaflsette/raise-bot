""" 
Envia pose para IK e obtém ângulos das juntas
"""
import numpy as np


def solve_ik(pose_4x4):
    """
    Dummy solver — depois substituir com a função real de IK.
    """
    # Exemplo: extração direta de posição
    position = pose_4x4[:3, 3]
    print(f"Posição alvo: {position}")
    return np.array([0.0, 0.0, 0.0])  # ângulos fictícios
