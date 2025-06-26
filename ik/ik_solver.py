"""
Envia pose para IK e obtém ângulos das juntas
"""

import numpy as np
from math import pi, sin, cos, atan2, sqrt


def calculate_ik(p):
    """A função recebe um ponto p = [x, y, z] no espaço 3D e retorna os valores das juntas [theta1, d2, d3] necessários para alcançar esse ponto"""
    x, y, z = p
    t1 = atan2(y, x)  
    d2 = z            
    d3 = sqrt(x**2 + y**2)
    return [t1, d2, d3]


def solve_ik(pose_4x4):
    """
    Usa a posição extraída da pose para calcular IK com modelo do Lab 2
    """
    position = pose_4x4[:3, 3]
    q = calculate_ik(position)
    print(f"Posição alvo (m): {position}")
    print(f"Ângulos calculados: {q}")
    return q
