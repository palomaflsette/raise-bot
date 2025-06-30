# calibrate_simple_stable.py - Versão Simples e Estável
import numpy as np
import cv2
import depthai as dai
import traceback
import time

# --- CONFIGURAÇÕES ---
CAMERA_PREVIEW_W, CAMERA_PREVIEW_H = 640, 480
CALIBRATION_FILE = "calib_matrix.txt"
CENTER_PIXEL = (CAMERA_PREVIEW_W // 2, CAMERA_PREVIEW_H // 2)

# Tamanho real do marcador ArUco em metros
ARUCO_SIZE = 0.065  # 65mm

# --- FILTRO SIMPLES ---


class SimpleFilter:
    def init(self, alpha=0.3):
        # Fator de suavização (0.1 = muito suave, 0.9 = responsivo)
        self.alpha = alpha
        self.filtered_pos = None
        self.filtered_rot = None
        self.detection_count = 0

    def update(self, tvec, rvec):
        if self.filtered_pos is None:
            # Primeira detecção
            self.filtered_pos = tvec.copy()
            self.filtered_rot = rvec.copy()
        else:
            # Filtro passa baixa simples
            self.filtered_pos = self.alpha * tvec + \
                (1 - self.alpha) * self.filtered_pos
            self.filtered_rot = self.alpha * rvec + \
                (1 - self.alpha) * self.filtered_rot

        self.detection_count += 1

    def is_ready(self):
        return self.detection_count > 10  # Precisa de pelo menos 10 detecções

    def get_pose(self):
        return self.filtered_pos, self.filtered_rot


def detect_aruco_simple(frame):
    """Detecção ArUco simples mas eficaz."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apenas melhora básica do contraste
    gray = cv2.equalizeHist(gray)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters_create()

    # Parâmetros básicos mas funcionais
    parameters.adaptiveThreshWinSizeMin = 7
    parameters.adaptiveThreshWinSizeMax = 30
    parameters.adaptiveThreshConstant = 7
    parameters.minMarkerPerimeterRate = 0.02
    parameters.maxMarkerPerimeterRate = 3.0

    corners, ids, _ = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)

    return corners, ids


def main():
    print("=== CALIBRAÇÃO SIMPLES E ESTÁVEL ===")
    print("1. Posicione o ArUco ID=0 bem visível")
    print("2. Aguarde a detecção se estabilizar (>10 detecções)")
    print("3. Pressione 'S' para calibrar")
    print("4. Pressione 'ESC' para sair")
    print("=" * 45)

    filter = SimpleFilter(alpha=0.2)  # Filtro suave

    try:
        # Pipeline mais simples
        pipeline = dai.Pipeline()

        # Só câmera RGB
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(CAMERA_PREVIEW_W, CAMERA_PREVIEW_H)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(15)  # FPS menor para reduzir carga

        # Mono cameras para depth (simplificado)
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        depth = pipeline.create(dai.node.StereoDepth)

        mono_left.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_400_P)  # Resolução menor
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        depth.setDefaultProfilePreset(
            dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)

        mono_left.out.link(depth.left)
        mono_right.out.link(depth.right)

        # Outputs
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        xout_depth.setStreamName("depth")

        cam_rgb.preview.link(xout_rgb.input)
        depth.depth.link(xout_depth.input)

        with dai.Device(pipeline) as device:
            print("✅ Câmera iniciada!")

            q_rgb = device.getOutputQueue("rgb", maxSize=2, blocking=False)
            q_depth = device.getOutputQueue("depth", maxSize=2, blocking=False)

            # Matriz da câmera
            calib_data = device.readCalibration()
            camera_matrix = np.array(calib_data.getCameraIntrinsics(
                dai.CameraBoardSocket.RGB, CAMERA_PREVIEW_W, CAMERA_PREVIEW_H
            ))
            dist_coeffs = np.array(
                calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))

            print("Aguardando detecções...")

            while True:
                # Pega frames
                in_rgb = q_rgb.tryGet()
                in_depth = q_depth.tryGet()

                if in_rgb is None:
                    continue

                frame = in_rgb.getCvFrame()
                depth_frame = in_depth.getFrame() if in_depth is not None else None

                # Detecta ArUco
                corners, ids = detect_aruco_simple(frame)

                # Desenha crosshair
                cv2.drawMarker(frame, CENTER_PIXEL, (0, 0, 255),
                               cv2.MARKER_CROSS, 20, 2)

                detected = False
                robot_pos = None

                if ids is not None and 0 in ids:
                    # Encontra marcador ID=0
                    marker_idx = np.where(ids.flatten() == 0)[0][0]

                    # Estima pose
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[marker_idx]], ARUCO_SIZE, camera_matrix, dist_coeffs
                    )

                    tvec = tvecs[0][0]
                    rvec = rvecs[0][0]

                    # Atualiza filtro
                    filter.update(tvec, rvec)

                    # Desenha marcador
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                    # Calcula posição da base do robô
                    filteredtvec,  = filter.get_pose()
                    robot_pos = filtered_tvec + \
                        np.array([0, 0, -0.15])  # 15cm abaixo

                    detected = True

                # Status
                if detected:
                    status_color = (0, 255, 0)
                    status_text = f"DETECTADO - Count: {filter.detection_count}"
                else:
                    status_color = (0, 0, 255)
                    status_text = "NÃO DETECTADO"

                cv2.putText(frame, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                # Info da posição
                if robot_pos is not None:
                    pos_text = f"Base: X={robot_pos[0]:.3f} Y={robot_pos[1]:.3f} Z={robot_pos[2]:.3f}"
                    cv2.putText(frame, pos_text, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    dist = np.linalg.norm(robot_pos)
                    cv2.putText(frame, f"Dist: {dist:.3f}m", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                # Instrução
                if filter.is_ready():
                    cv2.putText(frame, "PRONTO! Pressione 'S' para calibrar", (10, 420),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Aguardando estabilizar...", (10, 420),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                cv2.imshow("Calibração Simples", frame)
                key = cv2.waitKey(1) & 0xFF

                # CALIBRAR
                if key == ord('s') or key == ord('S'):
                    if not filter.is_ready():
                        print("❌ Aguarde mais detecções!")
                        continue

                    if robot_pos is None:
                        print("❌ Posição do robô não disponível!")
                        continue

                    print("\n--- CALIBRANDO ---")

                    # Pega pose filtrada
                    filtered_tvec, filtered_rvec = filter.get_pose()

                    # Matriz de transformação
                    R_marker_to_cam,  = cv2.Rodrigues(filtered_rvec)
                    T_marker_to_cam = np.eye(4)
                    T_marker_to_cam[:3, :3] = R_marker_to_cam
                    T_marker_to_cam[:3, 3] = filtered_tvec.flatten()

                    # Transformação câmera -> robô
                    T_cam_to_robot = np.eye(4)
                    T_cam_to_robot[:3, :3] = R_marker_to_cam.T
                    T_cam_to_robot[:3, 3] = robot_pos

                    # Salva
                    try:
                        np.savetxt(CALIBRATION_FILE,
                                   T_cam_to_robot, fmt='%.8f')
                        print(
                            f"✅ SUCESSO! Matriz salva em '{CALIBRATION_FILE}'")
                        print(f"Baseado em {filter.detection_count} detecções")
                        print(f"Posição da base: {robot_pos}")
                        print(f"Distância: {np.linalg.norm(robot_pos):.3f}m")

                    except Exception as e:
                        print(f"❌ Erro ao salvar: {e}")

                elif key == 27:  # ESC
                    break

            cv2.destroyAllWindows()

    except Exception as e:
        print(f"❌ ERRO:")
        traceback.print_exc()


if __name__ == "__main__":
    main()