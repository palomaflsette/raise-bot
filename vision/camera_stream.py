import depthai as dai
import cv2
import numpy as np
from PIL import Image, ImageTk
from vision.depth_stream import create_pipeline, create_simple_pipeline, filter_depth_range
from gui.plot_utils import render_profile_plot, render_depth_colormap
from customtkinter import CTkImage


def start_camera_stream(gui):
    """
    Inicia o stream da câmera com tratamento de erros melhorado
    """
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
            print(
                "[ERROR] Não foi possível inicializar a câmera. Verifique a conexão do dispositivo.")
            return

    try:
        print("[INFO] Conectando ao dispositivo...")
        gui.device = dai.Device(pipeline)

        print("[INFO] Configurando filas de saída...")
        gui.rgb_queue = gui.device.getOutputQueue(
            name="rgb", maxSize=4, blocking=False)
        gui.depth_queue = gui.device.getOutputQueue(
            name="depth", maxSize=4, blocking=False)

        print("[INFO] Iniciando atualização de frames...")
        update_camera_frames(gui)
        print("[INFO] Stream da câmera iniciado com sucesso!")

    except Exception as e:
        print(f"[ERROR] Erro ao conectar com o dispositivo: {e}")
        print("[INFO] Possíveis soluções:")
        print("- Verifique se o cabo USB está conectado")
        print("- Tente uma porta USB diferente")
        print("- Reinicie o dispositivo")
        print("- Verifique se não há outro processo usando a câmera")


def update_camera_frames(gui):
    """
    Atualiza os frames da câmera com tratamento de erros
    """
    try:
        # RGB
        if hasattr(gui, 'rgb_queue') and gui.rgb_queue:
            in_rgb = gui.rgb_queue.tryGet()
            if in_rgb:
                try:
                    rgb_frame = in_rgb.getCvFrame()
                    img = Image.fromarray(cv2.cvtColor(
                        rgb_frame, cv2.COLOR_BGR2RGB))
                    imgtk = CTkImage(light_image=img, size=(440, 350))
                    gui.rgb_canvas.configure(image=imgtk, text="")
                    gui.rgb_canvas.image = imgtk
                except Exception as e:
                    print(f"[WARNING] Erro ao processar frame RGB: {e}")

        # Depth
        if hasattr(gui, 'depth_queue') and gui.depth_queue:
            in_depth = gui.depth_queue.tryGet()
            if in_depth:
                try:
                    depth_frame = in_depth.getFrame()

                    if depth_frame is not None and depth_frame.size > 0:
                        depth_frame = filter_depth_range(depth_frame)
                        depth_frame = depth_frame.astype(np.uint16)

                        try:
                            render_profile_plot(
                                depth_frame, gui.normals_canvas, gui)
                        except Exception as e:
                            print(
                                f"[WARNING] Erro ao renderizar plot de perfil: {e}")

                        depth_clip = np.clip(
                            depth_frame, gui.min_depth, gui.max_depth)

                        depth_range = gui.max_depth - gui.min_depth
                        if depth_range > 0:
                            depth_norm = (
                                (depth_clip - gui.min_depth) / depth_range * 255).astype(np.uint8)
                        else:
                            depth_norm = np.zeros_like(
                                depth_clip, dtype=np.uint8)

                        depth_colormap = cv2.applyColorMap(
                            depth_norm, cv2.COLORMAP_JET)

                        img = Image.fromarray(depth_colormap)
                        imgtk = CTkImage(light_image=img, size=(440, 350))
                        #gui.depth_canvas.configure(image=imgtk, text="")
                        render_depth_colormap(
                            depth_frame, gui.depth_canvas, gui)

                        gui.depth_canvas.image = imgtk
                    else:
                        print("[WARNING] Frame de profundidade inválido recebido")

                except Exception as e:
                    print(
                        f"[WARNING] Erro ao processar frame de profundidade: {e}")

    except Exception as e:
        print(f"[ERROR] Erro geral na atualização de frames: {e}")

    try:
        gui.after(30, lambda: update_camera_frames(gui))
    except Exception as e:
        print(f"[ERROR] Erro ao agendar próxima atualização: {e}")


def cleanup_camera_stream(gui):
    """
    Limpa recursos da câmera
    """
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
