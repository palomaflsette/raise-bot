import cv2
import depthai as dai
import numpy as np

# Função para não fazer nada, necessária para os sliders


def nothing(x):
    pass


# --- PIPELINE CORRIGIDO ---
pipeline = dai.Pipeline()

# 1. Câmera Colorida
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)

# 2. Canal de Saída (para receber as imagens)
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# 3. Canal de Entrada (para enviar os controles) - ESTA É A CORREÇÃO
control_in = pipeline.create(dai.node.XLinkIn)
control_in.setStreamName("control")
control_in.out.link(cam_rgb.inputControl)
# --- FIM DA CORREÇÃO DO PIPELINE ---


# Conectar ao dispositivo e iniciar o pipeline
with dai.Device(pipeline) as device:
    # Obter as filas de entrada e saída
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_control = device.getInputQueue(
        name="control")  # Obter a fila de controle

    # Criar uma janela para a imagem e outra para os controles
    cv2.namedWindow("Controles da Camera")
    cv2.namedWindow("Camera View")

    # Criar os sliders (trackbars) para controlar a câmera
    cv2.createTrackbar(
        'Exposure (us)', 'Controles da Camera', 1, 33000, nothing)
    cv2.createTrackbar('ISO', 'Controles da Camera', 100, 1600, nothing)
    cv2.createTrackbar('White Balance (K)',
                       'Controles da Camera', 1000, 12000, nothing)

    # Definir valores iniciais para os sliders
    cv2.setTrackbarPos('Exposure (us)', 'Controles da Camera', 12000)
    cv2.setTrackbarPos('ISO', 'Controles da Camera', 500)
    # Um valor inicial mais neutro
    cv2.setTrackbarPos('White Balance (K)', 'Controles da Camera', 4000)

    print("\n--- INSTRUÇÕES ---")
    print("1. Mova os sliders na janela 'Controles da Camera'.")
    print("2. Observe o resultado na janela 'Camera View'.")
    print("3. Seu objetivo: Deixar a imagem com aparência natural e o objeto bem visível.")
    print("4. Anote os valores que ficam bons.")
    print("5. Aperte a tecla 'q' para fechar o programa.")
    print("------------------\n")

    while True:
        # Obter os valores atuais dos sliders
        exp_time = cv2.getTrackbarPos('Exposure (us)', 'Controles da Camera')
        iso_val = cv2.getTrackbarPos('ISO', 'Controles da Camera')
        wb_manual = cv2.getTrackbarPos(
            'White Balance (K)', 'Controles da Camera')

        # Criar o objeto de controle
        control = dai.CameraControl()
        control.setManualExposure(exp_time, iso_val)
        control.setManualWhiteBalance(wb_manual)

        # Enviar o controle para a fila de entrada - ESTA É A CORREÇÃO
        q_control.send(control)

        # Mostrar o frame da câmera
        in_rgb = q_rgb.get()
        frame = in_rgb.getCvFrame()

        # Mostrar os valores atuais na tela
        print(
            f"\rExposure: {exp_time} | ISO: {iso_val} | White Balance: {wb_manual}K", end="")
        cv2.imshow("Camera View", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    print("\n\nValores finais anotados! Agora, copie-os para o seu projeto principal.")
