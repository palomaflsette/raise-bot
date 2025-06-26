import numpy as np


def corrigir_perfil_com_transformacao(depth_frame, T_cam_to_robo, line_y=240, K=None):
    """
    Converte uma linha do depth_frame para o sistema do robô usando T_cam_to_robo.
    Retorna lista de pontos (x, z) no robô para traçar o perfil.
    """
    if K is None:
        # Parâmetros aproximados para câmera 640x480
        fx, fy = 615, 615
        cx, cy = 320, 240
    else:
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

    h, w = depth_frame.shape
    perfil_robo = []

    for u in range(w):
        d = depth_frame[line_y, u]
        if d == 0:
            continue
        z = d / 1000.0  # converter mm → metros
        x = (u - cx) * z / fx
        y = (line_y - cy) * z / fy
        p_cam = np.array([x, y, z, 1.0])
        p_robo = T_cam_to_robo @ p_cam
        perfil_robo.append((p_robo[0], p_robo[2]))  # x, z no robô

    return perfil_robo
