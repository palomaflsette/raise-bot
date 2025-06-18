""" 
Cinemática direta/inversa
"""
import numpy as np

from transforms.transformations import HT, HX, HZ


def pincher_frames(q):
    """
    Retorna lista de transformações homogêneas para o braço tipo Pincher
    """
    L1, L2, L3 = 0.1, 0.1, 0.1  
    frames = []

    # Base fixa
    frames.append(HZ(q[0]) @ HT([0, 0, L1]))     # Junta 1 (rotação Z) + subida
    # Junta 2 (rotação X) + braço 1
    frames.append(HX(q[1]) @ HT([0, 0, L2]))
    # Junta 3 (rotação X) + braço 2
    frames.append(HX(q[2]) @ HT([0, 0, L3]))
    # Junta 4 (rotação X, fim do braço)
    frames.append(HX(q[3]))

    return frames



def get_ee(frames):
    return frames[-1]


def get_frame(frames, i):
    return frames[i]


def origin(T):
    return T[:3, 3]


def jacobian(q, delta=1e-6):
    """
    Jacobiana numérica por diferenças finitas
    """
    J = np.zeros((3, 4))
    def f(q): return origin(get_ee(pincher_frames(q)))

    f0 = f(q)
    for i in range(4):
        dq = np.copy(q)
        dq[i] += delta
        J[:, i] = (f(dq) - f0) / delta

    return J
