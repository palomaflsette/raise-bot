"""  
convertendo (u, v, z) em (X, Y ,Z) da camera
"""

import numpy as np

def pixel_to_camera_coords(u, v, Z, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * Z / fx
    y = (v - cy) * Z / fy
    return np.array([x, y, Z])
