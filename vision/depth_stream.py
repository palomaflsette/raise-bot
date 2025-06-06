"""
Configuração otimizada para captura de profundidade com OAK-D-Lite.

Este pipeline foi cuidadosamente ajustado com base nas configurações padrão da biblioteca DepthAI,
com modificações específicas para melhorar a estabilidade e qualidade em objetos próximos (aprox. 100–500mm),
incluindo filtros espaciais, temporais e controle de disparidade estendida.

As referências principais para esses parâmetros são:
- Documentação oficial da DepthAI: https://docs.luxonis.com
- Exemplos do repositório GitHub oficial: https://github.com/luxonis/depthai-experiments
- API Reference: https://docs.luxonis.com/projects/api/en/latest/components/nodes/stereo_depth/

Ajustes adicionais foram aplicados empiricamente com base na prática em laboratório e inspeção visual
dos mapas de profundidade.
"""

import depthai as dai
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d, uniform_filter


DEPTH_MIN = 100   # mm 
DEPTH_MAX = 430   # mm 
CONFIDENCE_THRESHOLD = 255  # Mais permissivo
LR_CHECK_THRESHOLD = 4      # Mais tolerante

def extract_stable_profile_line(depth_frame, line_y=240, window_size=5):
    height, width = depth_frame.shape
    start_y = max(0, line_y - window_size // 2)
    end_y = min(height, line_y + window_size // 2 + 1)
    lines = depth_frame[start_y:end_y, :].astype(np.float32)
    lines[lines == 0] = np.nan
    profile_line = np.nanmedian(lines, axis=0)
    valid_mask = ~np.isnan(profile_line)
    if np.sum(valid_mask) > 20:
        x_coords = np.arange(len(profile_line))
        valid_coords = x_coords[valid_mask]
        valid_values = profile_line[valid_mask]
        if len(valid_values) > 10:
            profile_line = np.interp(x_coords, valid_coords, valid_values)
            profile_line = gaussian_filter1d(profile_line, sigma=1.5)
    return profile_line


def extract_vertical_profile(depth_frame, col_x=320, window_size=5):
    height, width = depth_frame.shape
    start_x = max(0, col_x - window_size // 2)
    end_x = min(width, col_x + window_size // 2 + 1)
    columns = depth_frame[:, start_x:end_x].astype(np.float32)
    columns[columns == 0] = np.nan
    profile_column = np.nanmedian(columns, axis=1)
    valid_mask = ~np.isnan(profile_column)
    if np.sum(valid_mask) > 20:
        y_coords = np.arange(len(profile_column))
        valid_coords = y_coords[valid_mask]
        valid_values = profile_column[valid_mask]
        profile_column = np.interp(y_coords, valid_coords, valid_values)
        profile_column = gaussian_filter1d(profile_column, sigma=1.5)
    return profile_column


def local_surface_analysis(depth_frame, window_size=21):
    depth = depth_frame.astype(np.float32)
    depth[depth == 0] = np.nan
    mean = uniform_filter(depth, size=window_size, mode='constant')
    sq_mean = uniform_filter(depth**2, size=window_size, mode='constant')
    var = sq_mean - mean**2
    rugosity = np.sqrt(var)
    grad_x = np.gradient(depth, axis=1)
    grad_y = np.gradient(depth, axis=0)
    curvature = np.hypot(grad_x, grad_y)
    return rugosity, curvature


def create_pipeline():
    """
    Pipeline otimizado especificamente para detecção estável
    de objetos próximos com OAK-D-Lite
    """
    pipeline = dai.Pipeline()

    # RGB Camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setFps(30)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    # Mono Cameras - Configuração crítica para objetos próximos
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)

    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    #  400P ao invés de 720P para melhor performance em objetos próximos
    mono_left.setResolution(
        dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(
        dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setFps(30)
    mono_right.setFps(30)

    # StereoDepth - Configuração otimizada
    stereo = pipeline.create(dai.node.StereoDepth)
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # Configurações fundamentais
    # stereo.setDefaultProfilePreset(
    #     dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    # CRÍTICO: ExtendedDisparity para objetos próximos
    stereo.setExtendedDisparity(True)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)  # Desabilitar para reduzir ruído

    # Configuração avançada - A CHAVE PARA ESTABILIDADE
    config = stereo.initialConfig.get()

    # Filtros de pós-processamento
    try:
        # Filtro de threshold - usando a API correta
        config.postProcessing.thresholdFilter.minRange = DEPTH_MIN
        config.postProcessing.thresholdFilter.maxRange = DEPTH_MAX

        # Filtro espacial - CRÍTICO para suavização
        config.postProcessing.spatialFilter.enable = True
        config.postProcessing.spatialFilter.holeFillingRadius = 2
        config.postProcessing.spatialFilter.numIterations = 1
        config.postProcessing.spatialFilter.alpha = 0.5
        config.postProcessing.spatialFilter.delta = 20

        # Filtro temporal - estabiliza entre frames
        config.postProcessing.temporalFilter.enable = True
        config.postProcessing.temporalFilter.alpha = 0.4
        config.postProcessing.temporalFilter.delta = 20
        config.postProcessing.temporalFilter.persistencyMode = dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_8_OUT_OF_8

        # Filtro de speckle
        config.postProcessing.speckleFilter.enable = True
        config.postProcessing.speckleFilter.speckleRange = 50

    except AttributeError as e:
        print(
            f"[WARNING] Alguns filtros não estão disponíveis nesta versão do DepthAI: {e}")

    try:
        config.algorithmic.enableExtended = True
        config.algorithmic.enableLeftRightCheck = True
        config.algorithmic.leftRightCheckThreshold = LR_CHECK_THRESHOLD
    except AttributeError as e:
        print(f"[WARNING] Configurações algorítmicas não disponíveis: {e}")

    # Otimizando censo para objetos próximos com KERNEL_7X9
    try:
        config.censusTransform.enableMeanMode = True
        config.censusTransform.kernelSize = dai.RawStereoDepthConfig.CensusTransform.KernelSize.KERNEL_7x9
    except AttributeError as e:
        print(f"[WARNING] Configurações de censo não disponíveis: {e}")

    try:
        config.postProcessing.median = dai.MedianFilter.KERNEL_7x7
    except AttributeError as e:
        print(f"[WARNING] Filtro mediano não disponível: {e}")

    stereo.initialConfig.set(config)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline


def create_simple_pipeline():
    """
    Pipeline simplificado para máxima compatibilidade
    """
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setFps(30)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)

    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_left.setResolution(
        dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(
        dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setFps(30)
    mono_right.setFps(30)

    # StereoDepth - Configuração básica
    stereo = pipeline.create(dai.node.StereoDepth)
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # Configurações básicas mais compatíveis
    # stereo.setDefaultProfilePreset(
    #     dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    stereo.setExtendedDisparity(True)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline


def filter_depth_range(depth_frame, min_depth=DEPTH_MIN, max_depth=DEPTH_MAX):
    """
    Filtragem avançada para estabilizar profundidade em objetos próximos.
    Corrige erro do bilateralFilter usando float32.
    """
    filtered = depth_frame.astype(np.float32)

    #  máscara de validade
    mask = (filtered >= min_depth) & (filtered <= max_depth)
    filtered[~mask] = np.nan

    if np.count_nonzero(mask) > 100:
        #  filtro bilateral com float32 (ok para OpenCV)
        valid = np.copy(filtered)
        valid[np.isnan(valid)] = 0  # zera os NaNs para filtrar
        valid = cv2.bilateralFilter(valid, d=9, sigmaColor=75, sigmaSpace=75)
        valid[np.isnan(filtered)] = 0  # Restaura os nulos originais
        return valid.astype(np.uint16)
    else:
        filtered[np.isnan(filtered)] = 0
        return filtered.astype(np.uint16)



def extract_stable_profile_line(depth_frame, line_y=240, window_size=5):
    """
    Extrai linha de profundidade estável usando média de múltiplas linhas
    Para usar no plot_utils.py
    """
    height, width = depth_frame.shape

    start_y = max(0, line_y - window_size // 2)
    end_y = min(height, line_y + window_size // 2 + 1)

    # extraindo múltiplas linhas
    lines = depth_frame[start_y:end_y, :].astype(np.float32)
    lines[lines == 0] = np.nan

    #  mediana ao longo do eixo Y
    profile_line = np.nanmedian(lines, axis=0)

    #  filtros para suavização
    valid_mask = ~np.isnan(profile_line)
    if np.sum(valid_mask) > 20:
        # Interpolação de pontos faltantes
        x_coords = np.arange(len(profile_line))
        valid_coords = x_coords[valid_mask]
        valid_values = profile_line[valid_mask]

        if len(valid_values) > 10:
            # Interpolando valores faltantes
            profile_line = np.interp(x_coords, valid_coords, valid_values)

            #  filtro gaussiano para suavização final
            profile_line = gaussian_filter1d(profile_line, sigma=1.5)

    return profile_line


def optimize_device_settings(device):
    """
    Configurações adicionais do dispositivo para melhorar qualidade stereo
    """
    try:
        # nem todos os dispositivos têm essas configurações
        device.setIrLaserDotProjectorBrightness(0)
        device.setIrFloodLightBrightness(0)
        print("[INFO] Configurações IR otimizadas")
    except Exception as e:
        print(f"[INFO] Configurações IR não disponíveis: {e}")

def analyze_depth_quality(depth_frame, line_y=240):
    """
    Analisa qualidade da detecção de profundidade
    """
    profile = depth_frame[line_y, :].astype(np.float32)
    profile[profile == 0] = np.nan

    valid_points = np.sum(~np.isnan(profile))
    close_points = np.sum((profile > 200) & (profile < 600))

    stats = {
        'total_pixels': len(profile),
        'valid_points': valid_points,
        'close_points': close_points,
        'validity_ratio': valid_points / len(profile),
        'close_ratio': close_points / len(profile) if valid_points > 0 else 0,
        'mean_depth': np.nanmean(profile),
        'depth_std': np.nanstd(profile)
    }

    return stats
