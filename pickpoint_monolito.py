"""
Sistema interativo para câmera 3D OAK-D e Robô com motores Dynamixel
"""
import depthai as dai
import numpy as np
import cv2
from datetime import datetime
import json
from config import (T_CAM_TO_ROBOT, 
                    DEPTH_MAX, 
                    DEPTH_MIN, 
                    WINDOW_HEIGHT, 
                    WINDOW_WIDTH)


clicked_point = None
depth_value_at_click = None
p_cam_at_click = None
mouse_hover = None
recording_points = []
show_crosshair = True
camera_intrinsics = None
p_robot_at_click = None


def create_optimized_pipeline():
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
    pipeline = dai.Pipeline()

    # RGB Camera - Configuração consistente de resolução
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480)  # Mantendo consistência com depth
    cam_rgb.setResolution(
        dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setFps(30)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    # Configurações de exposição
    exposure_anotado = 7857
    iso_anotado = 729
    wb_anotado = 5388

    cam_rgb.initialControl.setManualExposure(exposure_anotado, iso_anotado)
    cam_rgb.initialControl.setManualWhiteBalance(wb_anotado)

    # Mono Cameras - Configuração crítica para objetos próximos
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)

    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    # 400P ao invés de 720P para melhor performance em objetos próximos
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
    stereo.setDefaultProfilePreset(
        dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    # CRÍTICO: ExtendedDisparity para objetos próximos
    stereo.setExtendedDisparity(True)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)  # Desabilitar para reduzir ruído

    # Configuração avançada - A CHAVE PARA ESTABILIDADE
    config = stereo.initialConfig.get()

    try:
        # Filtro de threshold
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

    # Saídas do pipeline
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)  # Usa preview que já está em 640x480

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline


def transform_cam_to_robot(p_cam):
     """
     Transforma coordenadas da câmera para coordenadas do robô,
     retornando assim a matriz homogênea de p_robo
     """
     if p_cam is None:
          return None
     
     p_cam_homogeneous = np.append(p_cam, 1)
     
     p_robot_homogeneous = T_CAM_TO_ROBOT @ p_cam_homogeneous
     
     return p_robot_homogeneous[:3]

def get_camera_intrinsics(device):
    """
    Obtém a matriz intrínseca K da câmera RGB
    """
    try:
        calibData = device.readCalibration()

        intrinsics = calibData.getCameraIntrinsics(
            dai.CameraBoardSocket.RGB, 640, 480)

        K = np.array([
            [intrinsics[0][0], 0, intrinsics[0][2]],
            [0, intrinsics[1][1], intrinsics[1][2]],
            [0, 0, 1]
        ])

        print(f"Matriz intrínseca K:")
        print(f"fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
        print(f"cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")

        return K

    except Exception as e:
        print(f"Erro ao obter matriz intrínseca: {e}")
        '''valores aproximados para OAK-D-Lite em 640x480 como fallback'''
        K_fallback = np.array([
            [525.0, 0, 320.0],
            [0, 525.0, 240.0],
            [0, 0, 1]
        ])
        print("Usando matriz intrínseca aproximada como fallback")
        return K_fallback


def pixel_to_camera_coordinates(u, v, depth_mm, K):
    """
    Converte coordenadas de pixel + profundidade para coordenadas 3D da câmera
    p_cam = d * K^(-1) * [u v 1]^T
    
    Args:
        u, v: coordenadas do pixel
        depth_mm: profundidade em milímetros
        K: matriz intrínseca 3x3
    
    Returns:
        p_cam: ponto 3D em coordenadas da câmera [X, Y, Z] em mm
    """
    depth = depth_mm 

    pixel_homogeneous = np.array([u, v, 1.0])

    K_inv = np.linalg.inv(K)

    p_cam = depth * (K_inv @ pixel_homogeneous)

    return p_cam


def mouse_callback(event, x, y, flags, param):
    """
    Callback para eventos do mouse
    """
    global clicked_point, mouse_hover

    if event == cv2.EVENT_LBUTTONDOWN:
        if x < 640:
            clicked_point = (x, y)
        else:  
            clicked_point = (x - 640, y)

    mouse_hover = (x, y)


def get_depth_at_point(depth_frame, x, y, window_size=5):
    """
    Obtém valor de profundidade em um ponto com média local para estabilidade - CORRIGIDO
    """
    h, w = depth_frame.shape

    x = max(window_size//2, min(x, w - window_size//2 - 1))
    y = max(window_size//2, min(y, h - window_size//2 - 1))

    """janela ao redor do ponto"""
    x_start = x - window_size//2
    x_end = x + window_size//2 + 1
    y_start = y - window_size//2
    y_end = y + window_size//2 + 1

    window = depth_frame[y_start:y_end, x_start:x_end].astype(np.float32)

    valid_mask = (window > DEPTH_MIN) & (window < DEPTH_MAX)
    valid_values = window[valid_mask]

    if len(valid_values) > 0:
        depth_value = np.median(valid_values)
        confidence = len(valid_values) / window.size
        return depth_value, confidence
    else:
        return None, 0


def create_depth_colormap(depth_frame):
    """
    Cria mapa de cores otimizado para visualização de profundidade
    """
    # Normaliza parao range esperado
    depth_normalized = np.clip(depth_frame, DEPTH_MIN, DEPTH_MAX)
    depth_normalized = ((depth_normalized - DEPTH_MIN) /
                        (DEPTH_MAX - DEPTH_MIN) * 255).astype(np.uint8)

    #  colormap (TURBO é bom para profundidade)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)

    #  áreas inválidas marcadas
    invalid_mask = (depth_frame < DEPTH_MIN) | (depth_frame > DEPTH_MAX)
    depth_colormap[invalid_mask] = [0, 0, 0]  

    return depth_colormap


def draw_interface(rgb_frame, depth_colormap, depth_frame):
    """
    Desenha interface com informações e marcadores
    FIXADO: Verificação e redimensionamento de frames + cálculo p_cam
    """
    global clicked_point, depth_value_at_click, p_cam_at_click, mouse_hover, recording_points, camera_intrinsics

    target_height, target_width = 480, 640

    if rgb_frame.shape[:2] != (target_height, target_width):
        rgb_frame = cv2.resize(rgb_frame, (target_width, target_height))

    if depth_colormap.shape[:2] != (target_height, target_width):
        depth_colormap = cv2.resize(
            depth_colormap, (target_width, target_height))

    if depth_frame.shape[:2] != (target_height, target_width):
        depth_frame = cv2.resize(depth_frame, (target_width, target_height))

    canvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

    canvas[:target_height, :target_width] = rgb_frame
    canvas[:target_height, target_width:WINDOW_WIDTH] = depth_colormap

    cv2.line(canvas, (640, 0), (640, 480), (255, 255, 255), 2)

    cv2.putText(canvas, "RGB", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(canvas, "DEPTH", (650, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if show_crosshair and mouse_hover:
        mx, my = mouse_hover
        cv2.line(canvas, (mx, 0), (mx, WINDOW_HEIGHT), (0, 255, 0), 1)
        cv2.line(canvas, (0, my), (WINDOW_WIDTH, my), (0, 255, 0), 1)

    if clicked_point:
        x, y = clicked_point

        depth_value, confidence = get_depth_at_point(depth_frame, x, y)

        if depth_value and camera_intrinsics is not None:
            depth_value_at_click = depth_value

            p_cam_at_click = pixel_to_camera_coordinates(
                x, y, depth_value, camera_intrinsics)
            
            global p_robot_at_click
            p_robot_at_click = transform_cam_to_robot(p_cam_at_click)

            cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(canvas, (x, y), 8, (0, 255, 0), 2)

            cv2.circle(canvas, (x + 640, y), 5, (0, 255, 0), -1)
            cv2.circle(canvas, (x + 640, y), 8, (0, 255, 0), 2)

            info_text = [
                f"Pixel: ({x}, {y})",
                f"Profundidade: {depth_value:.1f} mm",
                f"Confianca: {confidence*100:.0f}%",
                "--- COORDENADAS CAMERA ---",
                f"p_cam X: {p_cam_at_click[0]:.1f} mm",
                f"p_cam Y: {p_cam_at_click[1]:.1f} mm",
                f"p_cam Z: {p_cam_at_click[2]:.1f} mm",
                "--- COORDENADAS ROBO ---",
                f"p_robot X: {p_robot_at_click[0]:.1f} mm",
                f"p_robot Y: {p_robot_at_click[1]:.1f} mm",
                f"p_robot Z: {p_robot_at_click[2]:.1f} mm"
            ]

            y_offset = 60
            for i, text in enumerate(info_text):
                y_pos = y_offset + i * 25
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(canvas, (10, y_pos - text_height - 5),
                              (15 + text_width, y_pos + 5), (0, 0, 0), -1)
                
                if "CAMERA" in text or "ROBO" in text:
                    color = (255, 255, 255)  # Branco paraos headers
                elif i >= 4 and i <= 6:  # p_cam
                    color = (0, 255, 255)   # Ciano para câmera
                elif i >= 8 and i <= 10: # p_robot  
                    color = (255, 0, 255)   # Magenta para robô
                else:
                    color = (0, 255, 0)     # Verde para resto
                
                cv2.putText(canvas, text, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

            if len(recording_points) >= 0:  
                recording_points.append({
                    'timestamp': datetime.now().isoformat(),
                    'pixel': (x, y),
                    'depth_mm': float(depth_value),
                    'confidence': float(confidence),
                    'p_cam_mm': {
                        'X': float(p_cam_at_click[0]),
                        'Y': float(p_cam_at_click[1]),
                        'Z': float(p_cam_at_click[2])
                    }
                })

    instructions = [
        "Click: Medir profundidade + p_cam",
        "C: Limpar marcacao",
        "S: Salvar medicao",
        "R: Gravar serie",
        "H: Toggle crosshair",
        "Q/ESC: Sair"
    ]

    for i, instruction in enumerate(instructions):
        y_pos = WINDOW_HEIGHT - 20 - (len(instructions) - i - 1) * 20
        cv2.putText(canvas, instruction, (WINDOW_WIDTH - 250, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    valid_depth = depth_frame[(depth_frame > DEPTH_MIN)
                              & (depth_frame < DEPTH_MAX)]
    if len(valid_depth) > 0:
        stats_text = f"Prof. media: {np.mean(valid_depth):.1f}mm | Min: {np.min(valid_depth):.1f}mm | Max: {np.max(valid_depth):.1f}mm"
        cv2.putText(canvas, stats_text, (10, WINDOW_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return canvas


def save_measurement(rgb_frame, depth_frame, clicked_point, depth_value, p_cam_point):
    """
    Salva medição atual em arquivo incluindo coordenadas p_cam
    """
    if clicked_point and depth_value and p_cam_point is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        data = {
            'timestamp': timestamp,
            'pixel_coordinates': clicked_point,
            'depth_value_mm': float(depth_value),
            'camera_coordinates_mm': {
                'X': float(p_cam_point[0]),
                'Y': float(p_cam_point[1]),
                'Z': float(p_cam_point[2])
            },
            'robot_coordinates_mm': {  # NOVO!
                'X': float(p_robot_at_click[0]),
                'Y': float(p_robot_at_click[1]),
                'Z': float(p_robot_at_click[2])
            },
            'depth_stats': {
                'min': float(np.min(depth_frame[depth_frame > 0])),
                'max': float(np.max(depth_frame[depth_frame > 0])),
                'mean': float(np.mean(depth_frame[depth_frame > 0]))
            }
        }

        with open(f'measurement_{timestamp}.json', 'w') as f:
            json.dump(data, f, indent=2)

        cv2.imwrite(f'rgb_{timestamp}.png', cv2.cvtColor(
            rgb_frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'depth_{timestamp}.png', depth_frame)

        print(f"Medição salva: measurement_{timestamp}.json")
        print(
            f"p_cam: X={p_cam_point[0]:.1f}, Y={p_cam_point[1]:.1f}, Z={p_cam_point[2]:.1f} mm")
        print(
            f"p_robot: X={p_robot_at_click[0]:.1f}, Y={p_robot_at_click[1]:.1f}, Z={p_robot_at_click[2]:.1f} mm")



def main():
    global clicked_point, show_crosshair, recording_points, camera_intrinsics

    pipeline = create_optimized_pipeline()

    try:
        with dai.Device(pipeline) as device:
            print(f"Dispositivo conectado: {device.getDeviceName()}")

            camera_intrinsics = get_camera_intrinsics(device)

            q_rgb = device.getOutputQueue(
                name="rgb", maxSize=4, blocking=False)
            q_depth = device.getOutputQueue(
                name="depth", maxSize=4, blocking=False)

            # Configurar janela
            cv2.namedWindow("OAK-D Depth Measurement", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("OAK-D Depth Measurement",
                             WINDOW_WIDTH, WINDOW_HEIGHT)
            cv2.setMouseCallback("OAK-D Depth Measurement", mouse_callback)

            print("Sistema de medição de profundidade iniciado!")
            print("Clique em qualquer ponto para medir a profundidade e obter p_cam")

            while True:
                in_rgb = q_rgb.get()
                in_depth = q_depth.get()

                if in_rgb and in_depth:
                    rgb_frame = in_rgb.getCvFrame()
                    depth_frame = in_depth.getFrame()

                    # print(f"RGB shape: {rgb_frame.shape}, Depth shape: {depth_frame.shape}")

                    depth_colormap = create_depth_colormap(depth_frame)

                    display_frame = draw_interface(
                        rgb_frame, depth_colormap, depth_frame)

                    cv2.imshow("OAK-D Depth Measurement", display_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  
                    break
                elif key == ord('c'):
                    clicked_point = None
                    depth_value_at_click = None
                    
                    p_cam_at_click = None
                    p_robot_at_click = None
                    print("Marcação limpa")
                elif key == ord('s'):  
                    if clicked_point and depth_value_at_click and p_cam_at_click is not None:
                        save_measurement(rgb_frame, depth_frame, clicked_point,
                                         depth_value_at_click, p_cam_at_click)
                    else:
                        print("Nenhum ponto selecionado para salvar")
                elif key == ord('h'):  
                    show_crosshair = not show_crosshair
                    print(f"Crosshair: {'ON' if show_crosshair else 'OFF'}")
                elif key == ord('r'):  
                    if len(recording_points) == 0:
                        recording_points = []
                        print(
                            "Iniciando gravação de pontos... (Clique em pontos para gravar)")
                    else:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        with open(f'recording_{timestamp}.json', 'w') as f:
                            json.dump(recording_points, f, indent=2)
                        print(
                            f"Gravação salva com {len(recording_points)} pontos: recording_{timestamp}.json")
                        recording_points = []

    except Exception as e:
        print(f"Erro ao conectar com o dispositivo: {e}")
        print(
            "Verifique se o OAK-D está conectado e não está sendo usado por outro processo")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
