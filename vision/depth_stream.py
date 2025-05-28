"""
Captura do depth map com controle de parâmetros.
"""

import depthai as dai
import numpy as np
from depthai_sdk import Record


DEPTH_MIN = 100   
DEPTH_MAX = 10000 


"""
Captura do depth map com parâmetros refinados
"""


def create_pipeline():
    pipeline = dai.Pipeline()

    # Câmera RGB
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setFps(30)

    # Câmeras mono
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_left.setResolution(
        dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(
        dai.MonoCameraProperties.SensorResolution.THE_400_P)

    # Ligações
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # --- Parâmetros adicionais para melhorar visualização da profundidade ---
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(True)
    stereo.setSubpixel(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    stereo.initialConfig.setConfidenceThreshold(240)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.initialConfig.setBilateralFilterSigma(0)
    stereo.initialConfig.setLeftRightCheckThreshold(10)

    # Saídas
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.setOutputDepth(True)

    stereo.depth.link(xout_depth.input)

    return pipeline



# ---------- UTILIDADE OPCIONAL PARA APLICAR THRESHOLD ----------
def filter_depth_range(depth_frame, min_depth=DEPTH_MIN, max_depth=DEPTH_MAX):
    """
    Filtra a imagem de profundidade para manter apenas valores dentro da faixa desejada.
    Os pixels fora da faixa são zerados.
    """
    filtered = np.copy(depth_frame)
    filtered[(filtered < min_depth) | (filtered > max_depth)] = 0
    return filtered
