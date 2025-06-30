import depthai as dai
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import time
import threading
from queue import Queue
from vision.depth_stream import create_pipeline, create_simple_pipeline
from vision.interactive_targeting import get_candidate_points
from vision.interactive_targeting import get_candidate_points, draw_targets_on_rgb
from vision.depth_stream import create_pipeline, create_simple_pipeline, filter_depth_range
from gui.plot_utils import render_profile_plot, render_depth_colormap

_last_rgb_update = 0
_last_depth_update = 0
_last_plot_update = 0
RGB_UPDATE_INTERVAL = 0.001  
DEPTH_UPDATE_INTERVAL = 0.001
PLOT_UPDATE_INTERVAL = 0.001   # ~10fps

_rgb_image_cache = None
_depth_image_cache = None



def start_camera_stream(gui):
    """
    Versão final e corrigida.
    Inicia o stream da câmera, sendo o único local que se conecta ao dispositivo.
    """
    # 1. Configura parâmetros da GUI e modo de performance
    set_performance_mode(gui, mode="quality")

    if not hasattr(gui, 'min_depth'):
        gui.min_depth = 100
    if not hasattr(gui, 'max_depth'):
        gui.max_depth = 430

    gui._camera_running = True
    if not hasattr(gui, 'candidate_points'):
        gui.candidate_points = []

    gui._frame_skip_counter = 0

    pipeline = None
    try:
        print("[INFO] Tentando criar pipeline otimizado...")
        pipeline = create_pipeline()
    except Exception as e:
        print(f"[WARNING] Falha no pipeline otimizado: {e}")
        print("[INFO] Tentando pipeline simplificado...")
        try:
            pipeline = create_simple_pipeline()
        except Exception as e2:
            print(f"[ERROR] Falha também no pipeline simplificado: {e2}")
            print("[ERROR] Não foi possível criar um pipeline válido. Abortando.")
            return  

    try:
        print("[INFO] Conectando ao dispositivo (única chamada)...")
        gui.device = dai.Device(pipeline)
        print("[INFO] Conexão bem-sucedida!")

        if hasattr(gui, 'pincher'):
            gui.pincher.device = gui.device
            print("[INFO] Dispositivo associado ao robô Pincher.")

        print("[INFO] Configurando filas de saída...")
        gui.rgb_queue = gui.device.getOutputQueue(
            name="rgb", maxSize=4, blocking=False)
        gui.depth_queue = gui.device.getOutputQueue(
            name="depth", maxSize=4, blocking=False)

        print("[INFO] Iniciando atualização de frames...")
        threading.Thread(target=update_camera_loop_thread,
                         args=(gui,), daemon=True).start()

        print("[INFO] Stream da câmera iniciado com sucesso!")

    except Exception as e:
        print(
            f"[ERROR] Erro ao conectar com o dispositivo ou configurar filas: {e}")
        print("[INFO] Possíveis soluções:")
        print("- Verifique se o cabo USB está conectado firmemente.")
        print("- Tente uma porta USB diferente (preferencialmente USB 3.0).")
        print("- Reinicie o dispositivo (desconecte e reconecte o cabo USB).")
        print("- Verifique no Gerenciador de Tarefas se não há outro processo 'python.exe' usando a câmera.")
        return

def update_camera_loop_thread(gui):
    global _last_rgb_update, _last_depth_update, _last_plot_update
    print("[THREAD] Loop da câmera iniciado.")
    
    while getattr(gui, "_camera_running", False):
        now = time.time()

        if now - _last_rgb_update >= RGB_UPDATE_INTERVAL:
            if hasattr(gui, 'rgb_queue') and gui.rgb_queue:
                _update_rgb_frame_fast(gui)
                _last_rgb_update = now

        if now - _last_depth_update >= DEPTH_UPDATE_INTERVAL:
            if hasattr(gui, 'depth_queue') and gui.depth_queue:
                _update_depth_frame_fast(gui)
                _last_depth_update = now

        if now - _last_plot_update >= PLOT_UPDATE_INTERVAL:
            if hasattr(gui, 'depth_queue') and gui.depth_queue:
                _update_plot_frame_fast(gui)
                _last_plot_update = now

        time.sleep(0.005)  


def update_camera_frames_optimized(gui):
    """
    Versão ULTRA-OTIMIZADA da atualização de frames
    """
    if not hasattr(gui, '_camera_running') or not gui._camera_running:
        return

    current_time = time.time()
    global _last_rgb_update, _last_depth_update, _last_plot_update

    try:
        gui._frame_skip_counter += 1

        if current_time - _last_rgb_update >= RGB_UPDATE_INTERVAL:
            if hasattr(gui, 'rgb_queue') and gui.rgb_queue:
                _update_rgb_frame_fast(gui)
                _last_rgb_update = current_time

        if current_time - _last_depth_update >= DEPTH_UPDATE_INTERVAL:
            if hasattr(gui, 'depth_queue') and gui.depth_queue:
                _update_depth_frame_fast(gui)
                _last_depth_update = current_time

        if current_time - _last_plot_update >= PLOT_UPDATE_INTERVAL:
            if hasattr(gui, 'depth_queue') and gui.depth_queue:
                _update_plot_frame_fast(gui)
                _last_plot_update = current_time

    except Exception as e:
        print(f"[ERROR] Erro geral na atualização de frames: {e}")

    # Agendar próxima atualização com intervalo menor
    try:
        # ~60fps scheduler
        gui.after(16, lambda: update_camera_frames_optimized(gui))
    except Exception as e:
        print(f"[ERROR] Erro ao agendar próxima atualização: {e}")

def extract_mask_bowls(rgb_image):
    """
    Gera uma máscara binária das regiões azuladas dos bowls
    """
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    # Tons de azul ajustados
    lower_blue = np.array([90, 60, 50])
    upper_blue = np.array([135, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = cv2.medianBlur(mask, 7)

    # Detectar apenas regiões circulares grandes (usando contornos ou Hough)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros_like(mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if 30 < radius < 100:  
                cv2.drawContours(output, [cnt], -1, 255, -1)

    return output



def _update_rgb_frame_fast(gui):
    """
    Atualização otimizada apenas do frame RGB
    """
    try:
        in_rgb = gui.rgb_queue.tryGet()
        if not in_rgb:
            return
        

        rgb_frame = in_rgb.getCvFrame()
        gui._rgb_raw = rgb_frame.copy()

        mask_rgb = extract_mask_bowls(rgb_frame)
        debug_frame = rgb_frame.copy()
        cv2.putText(debug_frame, "Janela de Debug - Aperte 'Q' para fechar",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Debug da Mascara de Cor", mask_rgb)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
        gui.mask_rgb = mask_rgb 

        
        """DETECÇÃO DO ARUCO (ID 0) E MATRIZ T_cam_to_robo """
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            parameters = cv2.aruco.DetectorParameters_create()
            marker_length = 0.065  # 65 mm

            """ Matrizes intrínsecas da camera"""
            # camera_matrix = np.array([[615, 0, 320],
            #                         [0, 615, 240],
            #                         [0, 0, 1]], dtype=np.float32)
            calib_data = gui.device.readCalibration()
            camera_matrix = np.array(calib_data.getCameraIntrinsics(
                dai.CameraBoardSocket.RGB, 640, 480))
            dist_coeffs = np.zeros((5,)) 

            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=parameters)

            if ids is not None and 0 in ids:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_length, camera_matrix, dist_coeffs)

                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id == 0:
                        rvec = rvecs[i]
                        tvec = tvecs[i]
                        cv2.putText(rgb_frame, "ARUCO DETECTADO", (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                        cv2.drawFrameAxes(rgb_frame, camera_matrix,
                                        dist_coeffs, rvec, tvec, 0.05)
                        cv2.aruco.drawDetectedMarkers(rgb_frame, corners)

                        # Matriz homogênea
                        R, _ = cv2.Rodrigues(rvec)
                        T = np.eye(4)
                        T[:3, :3] = R
                        T[:3, 3] = tvec.flatten()
                        T_inv = np.linalg.inv(T)

                        gui.T_cam_to_robo = T_inv  
                        print(f"[INFO] T_cam_to_robo atualizado: {gui.T_cam_to_robo}")

        except Exception as e:
            print(f"[Aruco] Erro na detecção do marcador: {e}")

        #calculo dos pontos  candidatos
        if hasattr(gui, 'candidate_points'):
            rgb_frame = draw_targets_on_rgb(rgb_frame, gui.candidate_points)


        
        gui.rgb_canvas.bind("<Button-1>", lambda e: on_rgb_click(e, gui))

        canvas_width = gui.rgb_canvas.winfo_width()
        canvas_height = gui.rgb_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 440, 350

        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        rgb_resized = cv2.resize(
            rgb_frame, (canvas_width, canvas_height), interpolation=cv2.INTER_NEAREST)
        img = Image.fromarray(rgb_resized)
        imgtk = ImageTk.PhotoImage(img)

        gui.rgb_canvas.delete("all")
        x = canvas_width // 2
        y = canvas_height // 2
        gui.rgb_canvas.create_image(x, y, image=imgtk, anchor="center")
        gui.rgb_canvas.image = imgtk
        
        if hasattr(gui, 'candidate_points') and gui.candidate_points:
            rgb_frame = draw_targets_on_rgb(rgb_frame, gui.candidate_points)


    except Exception as e:
        print(f"[WARNING] Erro ao processar frame RGB: {e}")
        

def start_candidate_thread(gui):
    """
    Inicia uma thread dedicada para encontrar pontos candidatos de forma contínua
    e segura.
    """
    if not hasattr(gui, '_camera_running'):
        gui._camera_running = True

    def update_loop():
        print("[THREAD CANDIDATOS] Iniciada.")
        while getattr(gui, '_camera_running', False):
            try:
                # **CHAVE DA CORREÇÃO**: Checar se todos os componentes necessários estão prontos
                if not all(hasattr(gui, attr) for attr in ['last_depth_frame', 'T_cam_to_robo', 'pincher', '_rgb_raw']):
                    time.sleep(0.5) # Aguarda os componentes serem inicializados
                    continue
                
                # Garante que os valores não são None
                if gui.last_depth_frame is None or gui.T_cam_to_robo is None or gui.pincher is None or gui._rgb_raw is None:
                    # Imprime um status útil para debug na primeira vez
                    if not hasattr(gui, '_thread_wait_logged'):
                        print("[THREAD CANDIDATOS] Aguardando inicialização completa (câmera, ArUco, robô)...")
                        gui._thread_wait_logged = True
                    time.sleep(0.5)
                    continue
                
                # Se chegou aqui, tudo está pronto
                if hasattr(gui, '_thread_wait_logged'):
                    print("[THREAD CANDIDATOS] Componentes prontos. Iniciando busca de alvos.")
                    delattr(gui, '_thread_wait_logged')


                # Chama a nova função de busca de pontos (que é mais estável)
                pontos = get_candidate_points(
                    gui.last_depth_frame,
                    pincher=gui.pincher,
                    T_cam_to_robo=gui.T_cam_to_robo,
                    rgb_frame=gui._rgb_raw
                )
                
                # A nova função retorna uma lista vazia ou com um ponto estável
                gui.candidate_points = pontos

            except Exception as e:
                print(f"[THREAD CANDIDATOS] Erro ao atualizar candidatos: {e}")

            # Roda a busca de alvos a uma taxa mais controlada (e.g., 2 vezes por segundo)
            time.sleep(0.5) 

    threading.Thread(target=update_loop, daemon=True).start()


def on_rgb_click(event, gui):
    w, h = gui.rgb_canvas.winfo_width(), gui.rgb_canvas.winfo_height()
    u_click = int(event.x * 640 / w)
    v_click = int(event.y * 480 / h)

    # Verificar qual ponto clicado está mais próximo
    if hasattr(gui, 'candidate_points'):
        for pt in gui.candidate_points:
            _, _, _, u, v = pt
            if abs(u - u_click) < 10 and abs(v - v_click) < 10:
                gui.selected_point = pt
                print(f"Ponto selecionado (cam): {pt[:3]}")



def _update_depth_frame_fast(gui):
    """
    Atualização otimizada apenas do colormap de profundidade
    """
    try:
        in_depth = gui.depth_queue.tryGet()
        if not in_depth:
            return

        depth_frame = in_depth.getFrame()

        if depth_frame is None or depth_frame.size == 0:
            return

        depth_filtered = _filter_depth_fast(
            depth_frame, gui.min_depth, gui.max_depth)
        
        gui.last_depth_frame = depth_filtered.copy()


        render_depth_colormap(depth_filtered, gui.depth_canvas,
                              gui, gui.min_depth, gui.max_depth)

    except Exception as e:
        print(f"[WARNING] Erro ao processar frame de profundidade: {e}")


def _update_plot_frame_fast(gui):
    """
    Atualização otimizada apenas do gráfico de análise
    """
    try:
        in_depth = gui.depth_queue.tryGet()
        if not in_depth:
            return

        depth_frame = in_depth.getFrame()

        if depth_frame is None or depth_frame.size == 0:
            return

        depth_filtered = _filter_depth_fast(
            depth_frame, gui.min_depth, gui.max_depth)

        render_profile_plot(depth_filtered, gui.normals_canvas, gui)

    except Exception as e:
        print(f"[WARNING] Erro ao processar plot de análise: {e}")


def _filter_depth_fast(depth_frame, min_depth, max_depth):
    """
    Versão ultra-otimizada do filtro de profundidade
    """
    filtered = depth_frame.astype(np.float32)
    mask = (filtered >= min_depth) & (filtered <= max_depth) & (filtered > 0)
    filtered[~mask] = 0

    valid_count = np.count_nonzero(mask)
    if valid_count > 1000:  
        try:
            # Filtro mais leve
            filtered = cv2.bilateralFilter(
                filtered, d=5, sigmaColor=50, sigmaSpace=50)
        except:
            pass  

    return filtered.astype(np.uint16)


def update_camera_frames(gui):
    """
    Versão original mantida para compatibilidade
    """
    update_camera_frames_optimized(gui)


def cleanup_camera_stream(gui):
    """
    Limpa recursos da câmera
    """
    # Parar atualizações
    if hasattr(gui, '_camera_running'):
        gui._camera_running = False

    try:
        if hasattr(gui, 'device') and gui.device:
            gui.device.close()
            print("[INFO] Dispositivo fechado com sucesso")
    except Exception as e:
        print(f"[WARNING] Erro ao fechar dispositivo: {e}")

    try:
        if hasattr(gui, 'rgb_queue'):
            gui.rgb_queue = None
        if hasattr(gui, 'depth_queue'):
            gui.depth_queue = None
    except Exception as e:
        print(f"[WARNING] Erro ao limpar filas: {e}")


def check_device_connection():
    """
    Verifica se há dispositivos DepthAI conectados
    """
    try:
        devices = dai.Device.getAllAvailableDevices()
        if len(devices) == 0:
            print("[WARNING] Nenhum dispositivo DepthAI encontrado")
            return False
        else:
            print(f"[INFO] {len(devices)} dispositivo(s) encontrado(s):")
            for i, device in enumerate(devices):
                print(f"  Device {i}: {device.getMxId()}")
            return True
    except Exception as e:
        print(f"[ERROR] Erro ao verificar dispositivos: {e}")
        return False


def get_device_info(device):
    """
    Obtém informações do dispositivo para debug
    """
    try:
        info = {
            'mxid': device.getMxId(),
            'usb_speed': device.getUsbSpeed(),
            'device_name': device.getDeviceName(),
            'product_name': device.getProductName(),
        }

        print("[INFO] Informações do dispositivo:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        return info
    except Exception as e:
        print(f"[WARNING] Erro ao obter informações do dispositivo: {e}")
        return None



def set_performance_mode(gui, mode="balanced"):
    """
    Ajusta parâmetros de performance
    mode: "fast", "balanced", "quality"
    """
    global RGB_UPDATE_INTERVAL, DEPTH_UPDATE_INTERVAL, PLOT_UPDATE_INTERVAL

    if mode == "fast":
        RGB_UPDATE_INTERVAL = 0.050    # 20fps
        DEPTH_UPDATE_INTERVAL = 0.100  # 10fps
        PLOT_UPDATE_INTERVAL = 0.200   # 5fps
        print("[INFO] Modo de performance: RÁPIDO")

    elif mode == "balanced":
        RGB_UPDATE_INTERVAL = 0.033    # 30fps
        DEPTH_UPDATE_INTERVAL = 0.050  # 20fps
        PLOT_UPDATE_INTERVAL = 0.100   # 10fps
        print("[INFO] Modo de performance: BALANCEADO")

    elif mode == "quality":
        RGB_UPDATE_INTERVAL = 0.016    # 60fps
        DEPTH_UPDATE_INTERVAL = 0.033  # 30fps
        PLOT_UPDATE_INTERVAL = 0.050   # 20fps
        print("[INFO] Modo de performance: QUALIDADE MÁXIMA")

    else:
        print(
            f"[WARNING] Modo de performance desconhecido: {mode}. Usando modo balanceado.")
        RGB_UPDATE_INTERVAL = 0.033
        DEPTH_UPDATE_INTERVAL = 0.050
        PLOT_UPDATE_INTERVAL = 0.100
