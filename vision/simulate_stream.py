import sys
import os
sys.path.append(os.path.abspath(".."))

import tkinter as tk
from customtkinter import CTkImage
import math
import random
import time
from PIL import Image, ImageTk
import numpy as np
import cv2
from gui.plot_utils import render_profile_plot, render_depth_colormap


class DebugSimulator:
    def __init__(self):
        self.scenario = "moving_object"  # Cenário atual
        self.frame_count = 0
        self.time_start = time.time()
        self.noise_level = 0.1
        self.object_speed = 0.02
        self.paused = False

        # Parâmetros dos cenários
        self.scenarios = {
            "moving_object": "Objeto se movendo horizontalmente",
            "surface_wave": "Superfície ondulada",
            "step_surface": "Superfície com degraus",
            "noisy_data": "Dados com ruído",
            "real_objects": "Simulação de objetos reais",
            "calibration": "Padrão de calibração"
        }

    def generate_depth_frame(self, width=640, height=480):
        """Gera frame de profundidade baseado no cenário atual"""
        self.frame_count += 1
        time_factor = (time.time() - self.time_start) * self.object_speed

        if self.scenario == "moving_object":
            return self._generate_moving_object(width, height, time_factor)
        elif self.scenario == "surface_wave":
            return self._generate_surface_wave(width, height, time_factor)
        elif self.scenario == "step_surface":
            return self._generate_step_surface(width, height, time_factor)
        elif self.scenario == "noisy_data":
            return self._generate_noisy_data(width, height, time_factor)
        elif self.scenario == "real_objects":
            return self._generate_real_objects(width, height, time_factor)
        elif self.scenario == "calibration":
            return self._generate_calibration_pattern(width, height, time_factor)
        else:
            return self._generate_surface_wave(width, height, time_factor)

    def _generate_moving_object(self, width, height, time_factor):
        """Objeto cilíndrico se movendo horizontalmente"""
        depth_frame = np.full((height, width), 400,
                              dtype=np.uint16)  # Background a 400mm

        # Posição do objeto (oscila da esquerda para direita)
        obj_center_x = int(width/2 + 150 * math.sin(time_factor))
        obj_radius = 50
        obj_depth = 250  # Objeto mais próximo

        # Criar objeto cilíndrico
        y, x = np.ogrid[:height, :width]
        mask = (x - obj_center_x)**2 + (y - height/2)**2 <= obj_radius**2
        depth_frame[mask] = obj_depth

        # Adicionar ruído realista
        noise = np.random.normal(0, self.noise_level * 10, (height, width))
        depth_frame = depth_frame.astype(np.float32) + noise
        depth_frame = np.clip(depth_frame, 100, 500).astype(np.uint16)

        return depth_frame

    def _generate_surface_wave(self, width, height, time_factor):
        """Superfície ondulada que muda com o tempo"""
        x = np.linspace(0, 4 * np.pi, width)
        y = np.linspace(0, 2 * np.pi, height)
        X, Y = np.meshgrid(x, y)

        # Ondas se movendo
        wave = 200 + 50 * np.sin(X + time_factor) + \
            30 * np.sin(Y * 2 + time_factor * 0.5)

        # Adicionar componente de profundidade base
        depth_frame = (300 + wave).astype(np.uint16)

        # Adicionar ruído
        noise = np.random.normal(0, self.noise_level * 5, (height, width))
        depth_frame = depth_frame.astype(np.float32) + noise
        depth_frame = np.clip(depth_frame, 150, 450).astype(np.uint16)

        return depth_frame

    def _generate_step_surface(self, width, height, time_factor):
        """Superfície com degraus - útil para testar detecção de bordas"""
        depth_frame = np.full((height, width), 350, dtype=np.uint16)

        # Criar degraus
        step_positions = [width//4, width//2, 3*width//4]
        step_heights = [250, 200, 150]

        for i, (pos, depth) in enumerate(zip(step_positions, step_heights)):
            # Degrau com movimento sutil
            actual_pos = pos + int(20 * math.sin(time_factor + i))
            depth_frame[:, :actual_pos] = depth

        # Adicionar ruído
        noise = np.random.normal(0, self.noise_level * 3, (height, width))
        depth_frame = depth_frame.astype(np.float32) + noise
        depth_frame = np.clip(depth_frame, 100, 400).astype(np.uint16)

        return depth_frame

    def _generate_noisy_data(self, width, height, time_factor):
        """Dados com muito ruído - para testar robustez dos algoritmos"""
        # Base senoidal
        x = np.linspace(0, 2 * np.pi, width)
        base_profile = 300 + 80 * np.sin(x + time_factor)
        depth_frame = np.tile(base_profile, (height, 1))

        # Adicionar ruído intenso
        noise = np.random.normal(0, self.noise_level * 30, (height, width))
        depth_frame = depth_frame + noise

        # Adicionar alguns pontos inválidos (zeros)
        invalid_mask = np.random.random((height, width)) < 0.05
        depth_frame[invalid_mask] = 0

        depth_frame = np.clip(depth_frame, 0, 500).astype(np.uint16)
        return depth_frame

    def _generate_real_objects(self, width, height, time_factor):
        """Simulação mais realista de objetos comuns"""
        depth_frame = np.full((height, width), 450,
                              dtype=np.uint16)  # Background

        # Objeto 1: Caixa retangular
        box_x = int(width/3 + 30 * math.sin(time_factor))
        box_y = int(height/2)
        box_w, box_h = 80, 60
        box_depth = 200

        y1, y2 = max(0, box_y - box_h//2), min(height, box_y + box_h//2)
        x1, x2 = max(0, box_x - box_w//2), min(width, box_x + box_w//2)
        depth_frame[y1:y2, x1:x2] = box_depth

        # Objeto 2: Cilindro
        cyl_x = int(2*width/3 + 40 * math.cos(time_factor * 0.7))
        cyl_y = int(height/2)
        cyl_radius = 35
        cyl_depth = 180

        y, x = np.ogrid[:height, :width]
        cyl_mask = (x - cyl_x)**2 + (y - cyl_y)**2 <= cyl_radius**2
        depth_frame[cyl_mask] = cyl_depth

        # Adicionar ruído realista
        noise = np.random.normal(0, self.noise_level * 8, (height, width))
        depth_frame = depth_frame.astype(np.float32) + noise
        depth_frame = np.clip(depth_frame, 100, 500).astype(np.uint16)

        return depth_frame

    def _generate_calibration_pattern(self, width, height, time_factor):
        """Padrão de calibração - grade regular"""
        depth_frame = np.full((height, width), 300, dtype=np.uint16)

        # Criar grade
        grid_size = 40
        for i in range(0, width, grid_size):
            for j in range(0, height, grid_size):
                # Alternar profundidades
                if ((i // grid_size) + (j // grid_size)) % 2 == 0:
                    depth_frame[j:j+grid_size//2, i:i+grid_size//2] = 250
                else:
                    depth_frame[j:j+grid_size//2, i:i+grid_size//2] = 350

        # Ruído mínimo
        noise = np.random.normal(0, self.noise_level * 2, (height, width))
        depth_frame = depth_frame.astype(np.float32) + noise
        depth_frame = np.clip(depth_frame, 200, 400).astype(np.uint16)

        return depth_frame

    def generate_rgb_frame(self, width=640, height=480):
        """Gera frame RGB correspondente ao cenário"""
        if self.scenario == "moving_object":
            return self._generate_rgb_moving_object(width, height)
        elif self.scenario == "real_objects":
            return self._generate_rgb_real_objects(width, height)
        else:
            return self._generate_rgb_gradient(width, height)

    def _generate_rgb_moving_object(self, width, height):
        """RGB para objeto em movimento"""
        rgb_frame = np.full((height, width, 3), [
                            40, 40, 40], dtype=np.uint8)  # Fundo escuro

        # Objeto com cor
        time_factor = (time.time() - self.time_start) * self.object_speed
        obj_center_x = int(width/2 + 150 * math.sin(time_factor))
        obj_radius = 50

        y, x = np.ogrid[:height, :width]
        mask = (x - obj_center_x)**2 + (y - height/2)**2 <= obj_radius**2
        rgb_frame[mask] = [100, 150, 200]  # Azul

        return rgb_frame

    def _generate_rgb_real_objects(self, width, height):
        """RGB para objetos reais"""
        rgb_frame = np.full((height, width, 3), [60, 60, 60], dtype=np.uint8)

        time_factor = (time.time() - self.time_start) * self.object_speed

        # Caixa vermelha
        box_x = int(width/3 + 30 * math.sin(time_factor))
        box_y = int(height/2)
        box_w, box_h = 80, 60

        y1, y2 = max(0, box_y - box_h//2), min(height, box_y + box_h//2)
        x1, x2 = max(0, box_x - box_w//2), min(width, box_x + box_w//2)
        rgb_frame[y1:y2, x1:x2] = [200, 100, 100]  # Vermelho

        # Cilindro verde
        cyl_x = int(2*width/3 + 40 * math.cos(time_factor * 0.7))
        cyl_y = int(height/2)
        cyl_radius = 35

        y, x = np.ogrid[:height, :width]
        cyl_mask = (x - cyl_x)**2 + (y - cyl_y)**2 <= cyl_radius**2
        rgb_frame[cyl_mask] = [100, 200, 100]  # Verde

        return rgb_frame

    def _generate_rgb_gradient(self, width, height):
        """RGB gradiente padrão"""
        rgb_frame = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(3):
            rgb_frame[..., i] = np.linspace(0, 255, width, dtype=np.uint8)
        return rgb_frame

    def set_scenario(self, scenario):
        """Muda o cenário de simulação"""
        if scenario in self.scenarios:
            self.scenario = scenario
            self.frame_count = 0
            self.time_start = time.time()
            print(f"Cenário alterado para: {self.scenarios[scenario]}")

    def set_noise_level(self, level):
        """Ajusta nível de ruído (0.0 a 1.0)"""
        self.noise_level = max(0.0, min(1.0, level))
        print(f"Nível de ruído: {self.noise_level:.2f}")

    def set_speed(self, speed):
        """Ajusta velocidade da simulação"""
        self.object_speed = max(0.001, min(0.1, speed))
        print(f"Velocidade: {self.object_speed:.3f}")

    def toggle_pause(self):
        """Pausa/despausa a simulação"""
        self.paused = not self.paused
        return self.paused


# Instância global do simulador
debug_simulator = DebugSimulator()


def start_simulated_stream(gui):
    """Inicia o stream simulado melhorado"""
    print("Modo simulado iniciado - Versão Avançada")
    print("Cenários disponíveis:")
    for key, desc in debug_simulator.scenarios.items():
        print(f"  {key}: {desc}")

    gui.rgb_queue = None
    gui.depth_queue = None

    # Inicializar atributos para controle das imagens
    if not hasattr(gui, 'min_depth'):
        gui.min_depth = 100
    if not hasattr(gui, 'max_depth'):
        gui.max_depth = 500

    # Adicionar controles de debug à GUI
    add_debug_controls(gui)

    update_simulated_frames(gui)


def add_debug_controls(gui):
    """Adiciona controles de debug à interface"""
    try:
        # Criar frame para controles de debug
        debug_frame = tk.Frame(gui, bg="#2b2b2b")
        debug_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        # Título
        title_label = tk.Label(debug_frame, text="Controles de Debug",
                               fg="white", bg="#2b2b2b", font=("Arial", 10, "bold"))
        title_label.pack()

        # Frame para controles
        controls_frame = tk.Frame(debug_frame, bg="#2b2b2b")
        controls_frame.pack(fill=tk.X, pady=5)

        # Cenário
        scenario_frame = tk.Frame(controls_frame, bg="#2b2b2b")
        scenario_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(scenario_frame, text="Cenário:",
                 fg="white", bg="#2b2b2b").pack()
        scenario_var = tk.StringVar(value=debug_simulator.scenario)
        scenario_menu = tk.OptionMenu(scenario_frame, scenario_var,
                                      *debug_simulator.scenarios.keys(),
                                      command=lambda x: debug_simulator.set_scenario(x))
        scenario_menu.pack()

        # Ruído
        noise_frame = tk.Frame(controls_frame, bg="#2b2b2b")
        noise_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(noise_frame, text="Ruído:", fg="white", bg="#2b2b2b").pack()
        noise_scale = tk.Scale(noise_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                               command=lambda x: debug_simulator.set_noise_level(float(x)/100))
        noise_scale.set(debug_simulator.noise_level * 100)
        noise_scale.pack()

        # Velocidade
        speed_frame = tk.Frame(controls_frame, bg="#2b2b2b")
        speed_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(speed_frame, text="Velocidade:",
                 fg="white", bg="#2b2b2b").pack()
        speed_scale = tk.Scale(speed_frame, from_=1, to=100, orient=tk.HORIZONTAL,
                               command=lambda x: debug_simulator.set_speed(float(x)/1000))
        speed_scale.set(debug_simulator.object_speed * 1000)
        speed_scale.pack()

        # Botão de pausa
        pause_button = tk.Button(controls_frame, text="Pausar",
                                 command=lambda: toggle_pause_button(pause_button))
        pause_button.pack(side=tk.LEFT, padx=10)

        gui.debug_controls = {
            'scenario_var': scenario_var,
            'noise_scale': noise_scale,
            'speed_scale': speed_scale,
            'pause_button': pause_button
        }

    except Exception as e:
        print(f"Erro ao adicionar controles de debug: {e}")


def toggle_pause_button(button):
    """Toggle do botão de pausa"""
    paused = debug_simulator.toggle_pause()
    button.config(text="Continuar" if paused else "Pausar")


def update_simulated_frames(gui):
    """Atualiza frames simulados"""
    try:
        if debug_simulator.paused:
            gui.after(100, lambda: update_simulated_frames(gui))
            return

        # Gerar frames
        depth_frame = debug_simulator.generate_depth_frame()
        rgb_frame = debug_simulator.generate_rgb_frame()

        # Processar RGB
        img_rgb = Image.fromarray(rgb_frame)
        canvas_width = gui.rgb_canvas.winfo_width()
        canvas_height = gui.rgb_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 440, 350

        # Redimensionar RGB
        img_width, img_height = img_rgb.size
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        scale = min(scale_x, scale_y) * 0.95

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        img_rgb_resized = img_rgb.resize(
            (new_width, new_height), Image.LANCZOS)
        imgtk_rgb = ImageTk.PhotoImage(img_rgb_resized)

        
        # Atualizar canvas RGB
        gui.rgb_canvas.delete("all")
        x = canvas_width // 2
        y = canvas_height // 2
        gui.rgb_canvas.create_image(x, y, image=imgtk_rgb, anchor="center")
        gui.rgb_canvas.image = imgtk_rgb

        # Renderizar gráficos de análise
        try:
            render_profile_plot(depth_frame, gui.normals_canvas, gui)
        except Exception as e:
            print(f"Erro ao renderizar perfil: {e}")
        
        # Renderizar depth colormap
        try:
            render_depth_colormap(depth_frame, gui.depth_canvas, gui)
        except Exception as e:
            print(f"Erro ao renderizar depth: {e}")
            # Fallback para método antigo
            depth_clip = np.clip(depth_frame, gui.min_depth, gui.max_depth)
            depth_norm = ((depth_clip - gui.min_depth) /
                          (gui.max_depth - gui.min_depth) * 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

            img_depth = Image.fromarray(depth_colormap)
            depth_canvas_width = gui.depth_canvas.winfo_width()
            depth_canvas_height = gui.depth_canvas.winfo_height()

            if depth_canvas_width <= 1 or depth_canvas_height <= 1:
                depth_canvas_width, depth_canvas_height = 440, 350

            depth_img_width, depth_img_height = img_depth.size
            depth_scale_x = depth_canvas_width / depth_img_width
            depth_scale_y = depth_canvas_height / depth_img_height
            depth_scale = min(depth_scale_x, depth_scale_y) * 0.95

            depth_new_width = int(depth_img_width * depth_scale)
            depth_new_height = int(depth_img_height * depth_scale)

            img_depth_resized = img_depth.resize(
                (depth_new_width, depth_new_height), Image.LANCZOS)
            imgtk_depth = ImageTk.PhotoImage(img_depth_resized)

            gui.depth_canvas.delete("all")
            depth_x = depth_canvas_width // 2
            depth_y = depth_canvas_height // 2
            gui.depth_canvas.create_image(
                depth_x, depth_y, image=imgtk_depth, anchor="center")
            gui.depth_canvas.image = imgtk_depth

        # Info no título da janela
        scenario_name = debug_simulator.scenarios[debug_simulator.scenario]
        gui.title(
            f"RAISE - Debug Mode: {scenario_name} (Frame {debug_simulator.frame_count})")

    except Exception as e:
        print(f"Erro em update_simulated_frames: {e}")
        # Mostrar erro nos canvas
        try:
            gui.rgb_canvas.delete("all")
            gui.rgb_canvas.create_text(
                gui.rgb_canvas.winfo_width() // 2,
                gui.rgb_canvas.winfo_height() // 2,
                text="Erro ao carregar RGB",
                fill="#ff6b6b", font=("Arial", 12)
            )

            gui.depth_canvas.delete("all")
            gui.depth_canvas.create_text(
                gui.depth_canvas.winfo_width() // 2,
                gui.depth_canvas.winfo_height() // 2,
                text="Erro ao carregar Depth",
                fill="#ff6b6b", font=("Arial", 12)
            )
        except:
            pass

    # Próxima atualização
    gui.after(30, lambda: update_simulated_frames(gui))
