"""
Sistema Completo: OAK-D Depth Measurement + Controle Robô Pincher
"""

import depthai as dai
import numpy as np
import cv2
from datetime import datetime
import json
import time
from math import pi

from pincher import Pincher

DEPTH_MIN = 100   # mm
DEPTH_MAX = 1000  # mm
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 480
LR_CHECK_THRESHOLD = 5

ROBOT_PORT = 'COM7'
ROBOT_CONNECTED = False

clicked_point = None
depth_value_at_click = None
p_cam_at_click = None
p_robot_at_click = None
mouse_hover = None
recording_points = []
show_crosshair = True
camera_intrinsics = None
robot = None

# Matriz de transformação câmera->robô (FIXA baseada na calibração)
T_CAM_TO_ROBOT = np.array([
    [-1.,  0.,  0.,   0.],   # Rotação em Y (π) + Translação Z (00aamm)
    [0.,  1.,  0.,   0.],
    [0.,  0., -1., 550.],
    [0.,  0.,  0.,   1.]
])


def init_robot():
    """
    Inicializa conexão com o robô
    """
    global robot, ROBOT_CONNECTED

    try:
        robot = Pincher(ROBOT_PORT)
        print(f"[ROBÔ] ✓ Conectado na porta {ROBOT_PORT}")

        robot.enable([1, 2, 3, 4])
        time.sleep(0.5)
        robot.enable([1, 2, 3, 4])  
        current_angles = robot.getangle([1, 2, 3, 4])
        print(f"[ROBÔ] Ângulos atuais: {current_angles}")

        print("[ROBÔ] Movendo para posição inicial segura...")
        robot.setangle([1, 2, 3, 4], [0.0, 0.0, 0.0, 0.0], speed=0.2)
        time.sleep(2)

        ROBOT_CONNECTED = True
        print("[ROBÔ] ✓ Inicialização completa!")

    except Exception as e:
        print(f"[ROBÔ]  Erro ao conectar: {e}")
        print("[ROBÔ] Continuando apenas com medição de profundidade...")
        ROBOT_CONNECTED = False


def transform_cam_to_robot(p_cam):
    """
    Transforma coordenadas da câmera para coordenadas do robô
    """
    if p_cam is None:
        return None

    p_cam_homogeneous = np.append(p_cam, 1)

    p_robot_homogeneous = T_CAM_TO_ROBOT @ p_cam_homogeneous

    return p_robot_homogeneous[:3]


def convert_to_robot_format(p_robot_mm):
    """
    Converte coordenadas do robô de mm para metros e formato correto
    """
    if p_robot_mm is None:
        return None

    p_robot_m = p_robot_mm / 1000.0

    # Formato esperado pelo robô: [x, y, z, phi]
    phi = pi/4  # HORIZONTAL

    p_target = np.array([p_robot_m[0], p_robot_m[1], p_robot_m[2], phi])

    
    return p_target


def move_robot_to_point(p_robot_mm):
    """
    Move o robô para as coordenadas especificadas
    """
    global robot, ROBOT_CONNECTED

    if not ROBOT_CONNECTED or robot is None:
        print("[ROBÔ] Robô não conectado!")
        return False

    try:
        p_target = convert_to_robot_format(p_robot_mm)

        if p_target is None:
            print("[ROBÔ]  Coordenadas inválidas!")
            return False

        print(f"[ROBÔ]  Calculando movimento para: {p_target}")

        """ CINEMATICA INVERSA CALCULADA AQUI!! """
        q = robot.ik(p_target)
        print(f"[ROBÔ] Ângulos calculados: {q}")

        if robot.admissible(q):
            print(f"[ROBÔ] ✓ Configuração válida! Movendo...")

            robot.setcompliance([1, 2, 3, 4], 32)
            robot.move([1, 2, 3, 4], q, speed=0.3, margin=0.1)  # Velocidade moderada
            print(f"[ROBÔ] ✓ Movimento concluído!")
            return True
        else:
            print(f"[ROBÔ]  Configuração não admissível!")
            return False

    except ValueError as e:
        print(f"[ROBÔ]  Erro de cinemática: {e}")
        return False
    except Exception as e:
        print(f"[ROBÔ] Erro no movimento: {e}")
        return False


def check_workspace_limits(p_robot_mm):
    """
    Verifica se o ponto está dentro do espaço de trabalho do robô
    """
    if p_robot_mm is None:
        return False, "Coordenadas inválidas"

    p_m = p_robot_mm / 1000.0

    """Limites aproximados do Pincher baseado nos parâmetros"""
    l1, l2 = 0.109, 0.109
    alcance_max =  0.3
    alcance_min = 0.05     

    # Verificar distância radial
    dist_radial = np.sqrt(p_m[0]**2 + p_m[1]**2)

    if dist_radial > alcance_max:
        return False, f"Muito longe! Max: {alcance_max:.3f}m, Atual: {dist_radial:.3f}m"

    if dist_radial < alcance_min:
        return False, f"Muito perto! Min: {alcance_min:.3f}m, Atual: {dist_radial:.3f}m"

    if p_m[2] < 0.01:
        return False, f"Muito baixo! Min: 0.05m, Atual: {p_m[2]:.3f}m"

    if p_m[2] > 0.35:
        return False, f"Muito alto! Max: 0.35m, Atual: {p_m[2]:.3f}m"

    return True, "Ponto OK"

# ================================
# FUNÇÕES ORIGINAIS DO OAK-D
# ================================
def create_optimized_pipeline():
    """Pipeline otimizado para máxima precisão em objetos próximos"""
    pipeline = dai.Pipeline()

    # RGB Camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480)
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

    # Mono Cameras
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

    # StereoDepth
    stereo = pipeline.create(dai.node.StereoDepth)
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    stereo.setDefaultProfilePreset(
        dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    stereo.setExtendedDisparity(True)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)

    # Configuração avançada
    config = stereo.initialConfig.get()

    try:
        config.postProcessing.thresholdFilter.minRange = DEPTH_MIN
        config.postProcessing.thresholdFilter.maxRange = DEPTH_MAX
        config.postProcessing.spatialFilter.enable = True
        config.postProcessing.spatialFilter.holeFillingRadius = 2
        config.postProcessing.spatialFilter.numIterations = 1
        config.postProcessing.spatialFilter.alpha = 0.5
        config.postProcessing.spatialFilter.delta = 20
        config.postProcessing.temporalFilter.enable = True
        config.postProcessing.temporalFilter.alpha = 0.4
        config.postProcessing.temporalFilter.delta = 20
        config.postProcessing.temporalFilter.persistencyMode = dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_8_OUT_OF_8
        config.postProcessing.speckleFilter.enable = True
        config.postProcessing.speckleFilter.speckleRange = 50
    except AttributeError as e:
        print(f"[WARNING] Alguns filtros não estão disponíveis: {e}")

    try:
        config.algorithmic.enableExtended = True
        config.algorithmic.enableLeftRightCheck = True
        config.algorithmic.leftRightCheckThreshold = LR_CHECK_THRESHOLD
    except AttributeError as e:
        print(f"[WARNING] Configurações algorítmicas não disponíveis: {e}")

    stereo.initialConfig.set(config)

    # Saídas do pipeline
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline


def get_camera_intrinsics(device):
    """Obtém a matriz intrínseca K da câmera RGB"""
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
        K_fallback = np.array([
            [525.0, 0, 320.0],
            [0, 525.0, 240.0],
            [0, 0, 1]
        ])
        print("Usando matriz intrínseca aproximada como fallback")
        return K_fallback


def pixel_to_camera_coordinates(u, v, depth_mm, K):
    """Converte coordenadas de pixel + profundidade para coordenadas 3D da câmera"""
    depth = depth_mm
    pixel_homogeneous = np.array([u, v, 1.0])
    K_inv = np.linalg.inv(K)
    p_cam = depth * (K_inv @ pixel_homogeneous)
    return p_cam


def mouse_callback(event, x, y, flags, param):
    """Callback para eventos do mouse"""
    global clicked_point, mouse_hover

    if event == cv2.EVENT_LBUTTONDOWN:
        if x < 640:
            clicked_point = (x, y)
        else:  
            clicked_point = (x - 640, y)

    mouse_hover = (x, y)


def get_depth_at_point(depth_frame, x, y, window_size=5):
    """Obtém valor de profundidade em um ponto com média local para estabilidade"""
    h, w = depth_frame.shape

    x = max(window_size//2, min(x, w - window_size//2 - 1))
    y = max(window_size//2, min(y, h - window_size//2 - 1))

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
    """Cria mapa de cores otimizado para visualização de profundidade"""
    depth_normalized = np.clip(depth_frame, DEPTH_MIN, DEPTH_MAX)
    depth_normalized = ((depth_normalized - DEPTH_MIN) /
                        (DEPTH_MAX - DEPTH_MIN) * 255).astype(np.uint8)

    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)

    invalid_mask = (depth_frame < DEPTH_MIN) | (depth_frame > DEPTH_MAX)
    depth_colormap[invalid_mask] = [0, 0, 0]

    return depth_colormap


def draw_interface(rgb_frame, depth_colormap, depth_frame):
    """Desenha interface com informações e marcadores + STATUS DO ROBÔ"""
    global clicked_point, depth_value_at_click, p_cam_at_click, p_robot_at_click, mouse_hover, recording_points, camera_intrinsics

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

    robot_status = "ROBO: CONECTADO" if ROBOT_CONNECTED else "ROBO: DESCONECTADO"
    robot_color = (0, 255, 0) if ROBOT_CONNECTED else (0, 0, 255)
    cv2.putText(canvas, robot_status, (650, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, robot_color, 2)

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
            p_robot_at_click = transform_cam_to_robot(p_cam_at_click)

            workspace_ok, workspace_msg = check_workspace_limits(
                p_robot_at_click)

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
                f"p_robot Z: {p_robot_at_click[2]:.1f} mm",
                f"Workspace: {'OK' if workspace_ok else 'ERRO'}",
                f"{workspace_msg}"
            ]

            y_offset = 90  
            for i, text in enumerate(info_text):
                y_pos = y_offset + i * 22
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(canvas, (10, y_pos - text_height - 3),
                              (15 + text_width, y_pos + 3), (0, 0, 0), -1)

                if "CAMERA" in text or "ROBO" in text:
                    color = (255, 255, 255)  # Branco para headers
                elif i >= 4 and i <= 6:  # p_cam
                    color = (0, 255, 255)   # Ciano para câmera
                elif i >= 8 and i <= 10:  # p_robot
                    color = (255, 0, 255)   # Magenta para robô
                elif "OK" in text:
                    color = (0, 255, 0)     # Verde para workspace OK
                elif "ERRO" in text:
                    color = (0, 0, 255)     # Vermelho para workspace erro
                else:
                    color = (200, 200, 200)  # Cinza para resto

                cv2.putText(canvas, text, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    instructions = [
        "Click: Medir profundidade + p_cam + p_robot",
        "M: Mover robo para ponto clicado",
        "C: Limpar marcacao",
        "S: Salvar medicao",
        "H: Home position (robo)",
        "T: Toggle crosshair",
        "Q/ESC: Sair"
    ]

    for i, instruction in enumerate(instructions):
        y_pos = WINDOW_HEIGHT - 20 - (len(instructions) - i - 1) * 18
        cv2.putText(canvas, instruction, (WINDOW_WIDTH - 350, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Estatísticas da profundidade
    valid_depth = depth_frame[(depth_frame > DEPTH_MIN)
                              & (depth_frame < DEPTH_MAX)]
    if len(valid_depth) > 0:
        stats_text = f"Prof. media: {np.mean(valid_depth):.1f}mm | Min: {np.min(valid_depth):.1f}mm | Max: {np.max(valid_depth):.1f}mm"
        cv2.putText(canvas, stats_text, (10, WINDOW_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return canvas


def save_measurement(rgb_frame, depth_frame, clicked_point, depth_value, p_cam_point, p_robot_point):
    """Salva medição atual em arquivo incluindo todas as coordenadas"""
    if clicked_point and depth_value and p_cam_point is not None and p_robot_point is not None:
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
            'robot_coordinates_mm': {
                'X': float(p_robot_point[0]),
                'Y': float(p_robot_point[1]),
                'Z': float(p_robot_point[2])
            },
            'robot_connected': ROBOT_CONNECTED,
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
            f"p_robot: X={p_robot_point[0]:.1f}, Y={p_robot_point[1]:.1f}, Z={p_robot_point[2]:.1f} mm")


def cleanup_resources():
    """Limpa recursos e fecha conexões"""
    global robot, ROBOT_CONNECTED

    print("\n Limpando recursos...")

    if ROBOT_CONNECTED and robot:
        try:
            print("[ROBÔ] Desabilitando torque...")
            robot.disable([1, 2, 3, 4])
            print("[ROBÔ] Fechando conexão...")
            robot.close()
            print("[ROBÔ] ✓ Conexão fechada com sucesso!")
        except Exception as e:
            print(f"[ROBÔ] Erro ao fechar conexão: {e}")

    cv2.destroyAllWindows()
    print("✓ Janelas OpenCV fechadas")
    print("✓ Sistema finalizado!")


def main():
    global clicked_point, show_crosshair, recording_points, camera_intrinsics
    global depth_value_at_click, p_cam_at_click, p_robot_at_click

    print("=" * 60)
    print("SISTEMA INTEGRADO OAK-D + ROBÔ PINCHER")
    print("=" * 60)

    init_robot()

    pipeline = create_optimized_pipeline()

    try:
        with dai.Device(pipeline) as device:
            print(f"Dispositivo OAK-D conectado: {device.getDeviceName()}")

            camera_intrinsics = get_camera_intrinsics(device)

            q_rgb = device.getOutputQueue(
                name="rgb", maxSize=4, blocking=False)
            q_depth = device.getOutputQueue(
                name="depth", maxSize=4, blocking=False)

            cv2.namedWindow("OAK-D Robot Control", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("OAK-D Robot Control",
                             WINDOW_WIDTH, WINDOW_HEIGHT)
            cv2.setMouseCallback("OAK-D Robot Control", mouse_callback)

            print("\n SISTEMA INICIADO!")
            print("• Clique em pontos para medir coordenadas")
            print("• Pressione 'M' para mover o robô para o ponto clicado")
            print("• Pressione 'H' para posição home do robô")

            while True:
                in_rgb = q_rgb.get()
                in_depth = q_depth.get()

                if in_rgb and in_depth:
                    rgb_frame = in_rgb.getCvFrame()
                    depth_frame = in_depth.getFrame()

                    depth_colormap = create_depth_colormap(depth_frame)
                    display_frame = draw_interface(
                        rgb_frame, depth_colormap, depth_frame)
                    cv2.imshow("OAK-D Robot Control", display_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # Q ou ESC
                    print("\n Finalizando sistema...")
                    break
                elif key == ord('c'):  # Limpar
                    clicked_point = None
                    depth_value_at_click = None
                    p_cam_at_click = None
                    p_robot_at_click = None
                    print("Marcação limpa")
                elif key == ord('s'):  # Salvar
                    if all([clicked_point, depth_value_at_click, p_cam_at_click, p_robot_at_click]):
                        save_measurement(rgb_frame, depth_frame, clicked_point,
                                         depth_value_at_click, p_cam_at_click, p_robot_at_click)
                    else:
                        print("Nenhum ponto selecionado para salvar")
                elif key == ord('t'):  # Toggle crosshair
                    show_crosshair = not show_crosshair
                    print(f"Crosshair: {'ON' if show_crosshair else 'OFF'}")
                elif key == ord('m'):  # MOVER ROBÔ
                    if p_robot_at_click is not None:
                        print("\n INICIANDO MOVIMENTO DO ROBÔ...")
                        success = move_robot_to_point(p_robot_at_click)
                        if success:
                            print(" Robô movido com sucesso!")
                        else:
                            print(" Falha no movimento do robô!")
                    else:
                        print(" Clique em um ponto primeiro!")
                elif key == ord('h'):  # HOME POSITION
                    if ROBOT_CONNECTED:
                        print(" Movendo robô para posição HOME...")
                        try:
                            robot.setangle(
                                [1, 2, 3, 4], [0.0, 0.0, 0.0, 0.0], speed=0.2)
                            print(" Robô na posição HOME!")
                        except Exception as e:
                            print(f" Erro ao mover para HOME: {e}")
                    else:
                        print(" Robô não conectado!")
                elif key == ord('r'):  # RESETAR ROBÔ
                    if ROBOT_CONNECTED:
                        print("Resetando robô...")
                        try:
                            robot.disable([1, 2, 3, 4])
                            time.sleep(0.5)
                            robot.enable([1, 2, 3, 4])
                            time.sleep(0.5)
                            robot.setangle(
                                [1, 2, 3, 4], [0.0, 0.0, 0.0, 0.0], speed=0.2)
                            print(" Robô resetado com sucesso!")
                        except Exception as e:
                            print(f"Erro ao resetar robô: {e}")
                    else:
                        print(" Robô não conectado!")
                elif key == ord('p'):  # PRINT STATUS
                    print("\n STATUS DO SISTEMA:")
                    print(
                        f"• Robô conectado: {'SIM' if ROBOT_CONNECTED else 'NÃO'}")
                    print(
                        f"• Ponto clicado: {clicked_point if clicked_point else 'Nenhum'}")
                    if p_cam_at_click is not None:
                        print(
                            f"• Coordenadas câmera: X={p_cam_at_click[0]:.1f}, Y={p_cam_at_click[1]:.1f}, Z={p_cam_at_click[2]:.1f} mm")
                    if p_robot_at_click is not None:
                        print(
                            f"• Coordenadas robô: X={p_robot_at_click[0]:.1f}, Y={p_robot_at_click[1]:.1f}, Z={p_robot_at_click[2]:.1f} mm")
                        workspace_ok, workspace_msg = check_workspace_limits(
                            p_robot_at_click)
                        print(f"• Workspace: {workspace_msg}")
                    if ROBOT_CONNECTED:
                        try:
                            current_angles = robot.getangle([1, 2, 3, 4])
                            print(f"• Ângulos atuais: {current_angles}")
                        except Exception as e:
                            print(f"• Erro ao ler ângulos: {e}")
                elif key == ord('d'):  # DEBUG MODE
                    print("\n🔍 MODO DEBUG ATIVADO:")
                    if clicked_point:
                        x, y = clicked_point
                        depth_value, confidence = get_depth_at_point(
                            depth_frame, x, y)
                        print(f"• Pixel: ({x}, {y})")
                        print(f"• Profundidade raw: {depth_value} mm")
                        print(f"• Confiança: {confidence*100:.1f}%")

                        window_size = 11
                        x_start = max(0, x - window_size//2)
                        x_end = min(
                            depth_frame.shape[1], x + window_size//2 + 1)
                        y_start = max(0, y - window_size//2)
                        y_end = min(
                            depth_frame.shape[0], y + window_size//2 + 1)

                        window = depth_frame[y_start:y_end, x_start:x_end]
                        valid_depths = window[(window > DEPTH_MIN) & (
                            window < DEPTH_MAX)]

                        if len(valid_depths) > 0:
                            print(
                                f"• Região {window_size}x{window_size}: min={np.min(valid_depths):.1f}, max={np.max(valid_depths):.1f}, std={np.std(valid_depths):.1f}")

                        if camera_intrinsics is not None and depth_value:
                            p_cam_debug = pixel_to_camera_coordinates(
                                x, y, depth_value, camera_intrinsics)
                            print(f"• p_cam calculado: {p_cam_debug}")

                            p_robot_debug = transform_cam_to_robot(p_cam_debug)
                            print(f"• p_robot calculado: {p_robot_debug}")

                            if ROBOT_CONNECTED:
                                p_target = convert_to_robot_format(
                                    p_robot_debug)
                                if p_target is not None:
                                    try:
                                        q_calc = robot.ik(p_target)
                                        admissible = robot.admissible(q_calc)
                                        print(f"• Ângulos IK: {q_calc}")
                                        print(
                                            f"• Configuração admissível: {admissible}")
                                    except Exception as e:
                                        print(f"• Erro IK: {e}")
                elif key == ord('i'):  # INFO SISTEMA
                    print("\n INFORMAÇÕES DO SISTEMA:")
                    print("CONTROLES DISPONÍVEIS:")
                    print("• CLICK: Medir profundidade e calcular coordenadas")
                    print("• M: Mover robô para ponto clicado")
                    print("• H: Posição HOME do robô")
                    print("• R: Resetar robô (disable/enable)")
                    print("• C: Limpar marcação atual")
                    print("• S: Salvar medição em arquivo")
                    print("• T: Toggle crosshair do mouse")
                    print("• P: Mostrar status completo")
                    print("• D: Modo debug detalhado")
                    print("• I: Mostrar esta informação")
                    print("• Q/ESC: Sair do programa")
                    print(f"\nCONFIGURAÇÕES:")
                    print(f"• Faixa profundidade: {DEPTH_MIN}-{DEPTH_MAX} mm")
                    print(
                        f"• Resolução janela: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")
                    print(f"• Porta robô: {ROBOT_PORT}")

    except Exception as e:
        print(f" Erro no sistema: {e}")
        import traceback
        traceback.print_exc()

    finally:
        cleanup_resources()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n Interrompido pelo usuário (Ctrl+C)")
        cleanup_resources()
    except Exception as e:
        print(f"\n Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        cleanup_resources()
