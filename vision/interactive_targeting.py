import numpy as np
import cv2
import depthai as dai


def extract_mask_bowls(rgb_image):
    """
    Gera uma máscara binária das regiões azuladas dos bowls.
    """
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

    # Continue ajustando estes valores para melhorar a máscara
    lower_blue = np.array([85, 25, 25])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Filtros para limpar a máscara
    mask = cv2.medianBlur(mask, 7)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def get_candidate_points(depth_frame, pincher, T_cam_to_robo, rgb_frame=None, min_depth=100, max_depth=430):
    """
    Versão com DIAGNÓSTICO DETALHADO para encontrar a falha.
    """
    if rgb_frame is None or depth_frame is None:
             #falha um tanto improvável, mas bom ter a verificação
        print("[FALHA] Frame RGB ou de Profundidade não está chegando.")
        return []

    IMG_WIDTH, IMG_HEIGHT = 640, 480

    try:
        if rgb_frame.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
            rgb_frame_sane = cv2.resize(rgb_frame, (IMG_WIDTH, IMG_HEIGHT))
        else:
            rgb_frame_sane = rgb_frame

        if depth_frame.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
            depth_frame_sane = cv2.resize(depth_frame, (IMG_WIDTH, IMG_HEIGHT))
        else:
            depth_frame_sane = depth_frame
    except Exception as e:
        print(f"[FALHA] Erro ao redimensionar frames: {e}")
        return []

    # A partir daqui, usamos apenas os frames "sane"
    bowl_mask = extract_mask_bowls(rgb_frame_sane)
    contours, _ = cv2.findContours(
        bowl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # A CAUSA MAIS PROVÁVEL x_x
        print("[FALHA DE DETECÇÃO] Nenhum contorno encontrado. Ajuste os valores de cor (HSV) ou a iluminação.")
        return []

    main_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main_contour)
    if area < 500:
        print(
            f"[FALHA DE DETECÇÃO] Contorno encontrado é muito pequeno (Área: {area}). Objeto muito longe ou ruído.")
        return []

    M = cv2.moments(main_contour)
    if M["m00"] == 0:
        print("[FALHA DE CÁLCULO] Momento do contorno é zero.")
        return []

    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])

    if not (0 <= u < IMG_WIDTH and 0 <= v < IMG_HEIGHT):
        print(
            f"[FALHA DE CÁLCULO] Coordenada do centroide ({u},{v}) está fora dos limites da imagem.")
        return []

    # Leitura de profundidade
    half_window = 5
    v_start, v_end = max(0, v - half_window), min(IMG_HEIGHT, v + half_window)
    u_start, u_end = max(0, u - half_window), min(IMG_WIDTH, u + half_window)

    depth_region = depth_frame_sane[v_start:v_end, u_start:u_end]
    valid_depths = depth_region[depth_region > 0]

    if valid_depths.size < 5:
        print(
            f"[FALHA DE PROFUNDIDADE] Não há leitura de profundidade válida na região do alvo (u={u}, v={v}).")
        return []

    z_mm = np.median(valid_depths)
    if not (min_depth < z_mm < max_depth):
        print(
            f"[FALHA DE PROFUNDIDADE] Profundidade medida ({z_mm:.0f}mm) está fora do alcance permitido ({min_depth}-{max_depth}mm).")
        return []

    # Conversão e checagem de IK
    z = z_mm / 1000.0
    calib_data = pincher.device.readCalibration()
    intrinsics = calib_data.getCameraIntrinsics(
        dai.CameraBoardSocket.RGB, IMG_WIDTH, IMG_HEIGHT)
    fx, fy, cx, cy = intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    p_cam_h = np.array([x, y, z, 1])
    p_robo_h = T_cam_to_robo @ p_cam_h
    x_r, y_r, z_r = p_robo_h[:3]

    if not pincher.within_workspace(np.array([x_r, y_r, z_r])):
        return []

    q = pincher.try_ik_with_phi_range([x_r, y_r, z_r])
    if q is None:
        print(
            f"[FALHA DE IK] Ponto ({x_r:.2f}, {y_r:.2f}, {z_r:.2f}) é alcançável, mas nenhuma pose da garra funcionou.")
        return []

    print(
        f"[ALVO VÁLIDO ENCONTRADO] Ponto (u,v): ({u},{v}) -> Robô: ({x_r:.2f}, {y_r:.2f}, {z_r:.2f})")

    return [(x, y, z, u, v)]


def draw_targets_on_rgb(rgb_frame, points):
    """
    Desenha os alvos na imagem RGB.
    """
    if rgb_frame.shape[1] != 640 or rgb_frame.shape[0] != 480:
        rgb_frame = cv2.resize(rgb_frame, (640, 480))

    for _, _, _, u, v in points:
        if u is not None and v is not None:
            cv2.drawMarker(rgb_frame, (int(u), int(v)), (0, 255, 0), markerType=cv2.MARKER_CROSS,
                           markerSize=20, thickness=2)
    return rgb_frame
