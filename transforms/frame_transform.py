""" 
Aplicando T_cameta+robot
"""

import numpy as np

def apply_transform(point, T):
    """Transforma ponto [x, y, z] com matriz homogênea 4x4"""
    point_hom = np.append(point, 1)
    transformed = T @ point_hom
    return transformed[:3]
