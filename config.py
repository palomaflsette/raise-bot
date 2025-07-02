# --- CONFIGURAÇÕES DE HARDWARE ---
ROBOT_SERIAL_PORT = 'COM7'
ROBOT_IDS = [1, 2, 3, 4]
GRIPPER_ID = 5

DEPTH_MIN = 100   # mm
DEPTH_MAX = 1000  # mm
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 480
LR_CHECK_THRESHOLD = 5


""" 
T_{cam->robo}, agpra que conseguimos alinhar fisicamente  o sistem câmera-robô
"""
T_CAM_TO_ROBOT = np.array([
    [-1.,  0.,  0.,   0.],   # Rotação em Y (π) + Translação Z (500mm)
    [0.,  1.,  0.,   0.],
    [0.,  0., -1., 500.],
    [0.,  0.,  0.,   1.]
])
