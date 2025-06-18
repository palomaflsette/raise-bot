""" 
Aplicando T_cameta+robot
"""

import numpy as np

def apply_transform(point, T):
    """Transforma ponto [x, y, z] com matriz homogênea 4x4"""
    ponto_homogeneo = np.append(point, 1)
    transformed = T @ ponto_homogeneo
    return transformed[:3]