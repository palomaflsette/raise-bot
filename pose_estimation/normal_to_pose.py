""" 
Gera pose 4x4 com orientação alinhada à normal
"""

import numpy as np

def build_pose_matrix(position, normal):
    """Cria uma matriz de pose 4x4 onde z = normal"""
    z = normal / np.linalg.norm(normal)
    up = np.array([0, 1, 0]) if abs(z[1]) < 0.99 else np.array([1, 0, 0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    return T