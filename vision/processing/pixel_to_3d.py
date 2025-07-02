"""  
convertendo (u, v, z) em (X, Y ,Z) da camera
"""

import numpy as np

def pixel_to_camera_coords(u, v, Z, K):
    """ 
    Aqui estamos retornando p_cam (ver anotações)
    Implementação da formula de back-projection
    
    (u,v): coordenadas de pixels
    Z: profundidade -> distância da câmera até um dado ponto no mundo real
    K: matriz intrínseca da câmera
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * Z / fx
    y = (v - cy) * Z / fy
    return np.array([x, y, Z])
