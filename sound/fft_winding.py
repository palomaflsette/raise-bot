"""
Visualização Multi-Topológica Sonora - Múltiplas Representações
Diferentes topologias para análise completa do biofeedback sonoro
"""

import serial
import numpy as np
import matplotlib.pyplot as plt
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import savgol_filter, hilbert
from scipy.fft import fft, fftfreq
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# -----------------------------
# Múltiplas Topologias
# -----------------------------


class MultiTopologyAnalyzer:
    def __init__(self):
        self.buffer_size = 512
        self.history_size = 50
        self.signal_history = []

    def winding_curve(self, signal):
        """Topologia 1: Curva Winding clássica"""
        theta = np.linspace(0, 2 * np.pi * len(signal) / 100, len(signal))
        x = signal * np.cos(theta)
        y = signal * np.sin(theta)
        return x, y, "Winding Curve - Frequência vs Amplitude"

    def phase_space(self, signal):
        """Topologia 2: Espaço de Fase (signal vs derivative)"""
        derivative = np.gradient(signal)
        return signal[:-1], derivative[:-1], "Espaço de Fase - Sinal vs Derivada"

    def attractor_3d(self, signal):
        """Topologia 3: Atrator 3D (embedding dimensional)"""
        # Takens embedding
        tau = 1  # delay
        m = 3    # dimensão

        if len(signal) < m * tau:
            return signal[:10], signal[:10], "Atrator 3D - Dados Insuficientes"

        x = signal[:-2*tau]
        y = signal[tau:-tau]
        z = signal[2*tau:]
        return x, y, "Atrator 3D - Embedding de Takens", z

    def hilbert_transform(self, signal):
        """Topologia 4: Transformada de Hilbert - Envelope Complexo"""
        analytic = hilbert(signal)
        envelope = np.abs(analytic)
        phase = np.angle(analytic)

        x = envelope * np.cos(phase)
        y = envelope * np.sin(phase)
        return x, y, "Hilbert - Envelope vs Fase"

    def frequency_spiral(self, signal):
        """Topologia 5: Espiral de Frequências"""
        fft_vals = np.abs(fft(signal))[:len(signal)//2]
        freqs = np.arange(len(fft_vals))

        # Criar espiral baseada no espectro
        theta = freqs * 2 * np.pi / len(fft_vals)
        r = fft_vals

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y, "Espiral Espectral - Frequências"

    def poincare_map(self, signal):
        """Topologia 6: Mapa de Poincaré"""
        # Encontrar cruzamentos pelo zero
        zero_crossings = []
        for i in range(1, len(signal)):
            if signal[i-1] * signal[i] < 0:  # mudança de sinal
                zero_crossings.append(i)

        if len(zero_crossings) < 10:
            t = np.arange(len(signal))
            return signal, t, "Poincaré - Poucos Cruzamentos"

        # Valores nos cruzamentos
        crossing_values = signal[zero_crossings[:-1]]
        crossing_derivatives = np.gradient(signal)[zero_crossings[:-1]]

        return crossing_values, crossing_derivatives, "Mapa de Poincaré - Cruzamentos"

    def recurrence_plot_coords(self, signal):
        """Topologia 7: Coordenadas para Plot de Recorrência"""
        # Simplificado - distâncias entre pontos
        n = min(50, len(signal))  # Limitar para performance
        indices = np.linspace(0, len(signal)-1, n, dtype=int)
        subsignal = signal[indices]

        distances = []
        coords_x, coords_y = [], []

        threshold = np.std(subsignal) * 0.5

        for i in range(len(subsignal)):
            for j in range(len(subsignal)):
                if abs(subsignal[i] - subsignal[j]) < threshold:
                    coords_x.append(i)
                    coords_y.append(j)

        return coords_x, coords_y, "Recurrence Plot - Padrões Repetidos"

    def wavelet_topology(self, signal):
        """Topologia 8: Representação Wavelet simplificada"""
        # Transformada wavelet simples usando convolução
        scales = np.arange(1, 20)
        wavelet_coeffs = []

        for scale in scales:
            # Wavelet Ricker (Mexican hat) simplificado
            t = np.arange(-scale*2, scale*2+1)
            wavelet = (2/(np.sqrt(3*scale)*np.pi**0.25)) * \
                (1 - (t/scale)**2) * np.exp(-(t/scale)**2/2)

            # Convolução
            if len(signal) >= len(wavelet):
                coeff = np.convolve(signal, wavelet, mode='valid')
                wavelet_coeffs.append(np.max(np.abs(coeff)))
            else:
                wavelet_coeffs.append(0)

        x = scales
        y = wavelet_coeffs
        return x, y, "Wavelet - Escalas vs Coeficientes"


class MultiTopologyVisualizer:
    def __init__(self):
        self.analyzer = MultiTopologyAnalyzer()
        self.running = False
        self.ser = None
        self.data_counter = 0
        self.current_topology = 0
        self.topologies = [
            "winding_curve", "phase_space", "attractor_3d", "hilbert_transform",
            "frequency_spiral", "poincare_map", "recurrence_plot_coords", "wavelet_topology"
        ]

    def setup_serial(self):
        """Conecta com Arduino"""
        ports = ['COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9']

        for port in ports:
            try:
                print(f"Tentando {port}...")
                self.ser = serial.Serial(port, 115200, timeout=1)
                time.sleep(2)
                print(f"Conectado em {port}!")
                return True
            except:
                continue
        return False

    def setup_gui(self):
        """Interface com múltiplas topologias"""
        ctk.set_appearance_mode("dark")

        self.root = ctk.CTk()
        self.root.geometry("1400x900")
        self.root.title("Multi-Topologia Sonora - Biofeedback Avançado")

        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)

        control_frame = ctk.CTkFrame(main_frame)
        control_frame.pack(side="left", fill="y", padx=(0, 5))

        ctk.CTkLabel(control_frame, text="MULTI-TOPOLOGIA",
                     font=("Arial", 16, "bold")).pack(pady=10)

        # Seletor de topologia
        self.topology_var = ctk.StringVar(value="winding_curve")
        topology_menu = ctk.CTkOptionMenu(
            control_frame,
            values=[
                "winding_curve", "phase_space", "attractor_3d", "hilbert_transform",
                "frequency_spiral", "poincare_map", "recurrence_plot_coords", "wavelet_topology"
            ],
            variable=self.topology_var
        )
        ctk.CTkLabel(control_frame, text="Topologia:").pack(pady=(10, 2))
        topology_menu.pack(pady=5)

        # Botões
        button_frame = ctk.CTkFrame(control_frame)
        button_frame.pack(pady=10, fill="x")

        self.start_button = ctk.CTkButton(button_frame, text="▶ Iniciar",
                                          command=self.start_visualization, width=80)
        self.start_button.pack(pady=2)

        self.stop_button = ctk.CTkButton(button_frame, text="⏸ Parar",
                                         command=self.stop_visualization, width=80)
        self.stop_button.pack(pady=2)

        test_button = ctk.CTkButton(button_frame, text="🧪 Teste",
                                    command=self.test_all_topologies, width=80)
        test_button.pack(pady=2)

        cycle_button = ctk.CTkButton(button_frame, text="🔄 Ciclar",
                                     command=self.cycle_topology, width=80)
        cycle_button.pack(pady=2)

        # Status
        self.status_label = ctk.CTkLabel(control_frame, text="Pronto",
                                         font=("Arial", 12))
        self.status_label.pack(pady=10)

        # Métricas
        self.metrics_text = ctk.CTkTextbox(
            control_frame, width=280, height=300)
        self.metrics_text.pack(pady=10, fill="both", expand=True)

        # Frame de visualização - GRID 2x4
        viz_frame = ctk.CTkFrame(main_frame)
        viz_frame.pack(side="right", fill="both", expand=True)

        # Criar figura com subplots 2x4
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor('black')

        # 8 subplots para as 8 topologias
        self.axes = []
        for i in range(8):
            ax = plt.subplot(2, 4, i+1)
            ax.set_facecolor('black')
            self.axes.append(ax)

        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        plt.tight_layout()

    def generate_test_signal(self):
        """Sinal de teste rico"""
        t = np.linspace(0, 4*np.pi, self.analyzer.buffer_size)
        # Sinal complexo simulando biofeedback
        signal = (200 +
                  100 * np.sin(t + time.time()) +
                  50 * np.sin(3*t + 0.5*time.time()) +
                  30 * np.sin(7*t) +
                  20 * np.random.randn(len(t)) +
                  10 * np.sin(15*t) * np.exp(-t/10))
        return signal.astype(int)

    def read_arduino_data(self):
        """Lê dados do Arduino ou gera teste"""
        if not self.ser:
            return self.generate_test_signal()

        buffer = []
        attempts = 0

        while len(buffer) < self.analyzer.buffer_size and attempts < 1000:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8').strip()
                    if line.isdigit():
                        buffer.append(int(line))
                attempts += 1
            except:
                attempts += 1

        if len(buffer) < self.analyzer.buffer_size:
            test_data = self.generate_test_signal()
            buffer.extend(
                test_data[:(self.analyzer.buffer_size - len(buffer))])

        return np.array(buffer)

    def visualize_all_topologies(self, signal):
        """Visualiza todas as 8 topologias simultaneamente"""
        methods = [
            self.analyzer.winding_curve,
            self.analyzer.phase_space,
            self.analyzer.attractor_3d,
            self.analyzer.hilbert_transform,
            self.analyzer.frequency_spiral,
            self.analyzer.poincare_map,
            self.analyzer.recurrence_plot_coords,
            self.analyzer.wavelet_topology
        ]

        colors = ['lime', 'cyan', 'yellow', 'magenta',
                  'orange', 'red', 'white', 'pink']

        # Limpar todos os axes
        for ax in self.axes:
            ax.clear()
            ax.set_facecolor('black')

        metrics_text = f"ANÁLISE MULTI-TOPOLÓGICA\nFrame: {self.data_counter}\n\n"

        for i, (method, color, ax) in enumerate(zip(methods, colors, self.axes)):
            try:
                result = method(signal)

                if len(result) == 4:  # 3D case
                    x, y, title, z = result
                    # Para 3D, plotar projeção
                    ax.scatter(x[:100], y[:100], c=z[:100],
                               cmap='viridis', s=1, alpha=0.7)
                else:
                    x, y, title = result

                    if len(x) > 0 and len(y) > 0:
                        if i == 6:  # Recurrence plot
                            ax.scatter(x, y, c=color, s=1, alpha=0.6)
                        else:
                            ax.plot(x, y, color=color,
                                    linewidth=1.5, alpha=0.8)

                        # Adicionar métricas
                        if len(x) > 1 and len(y) > 1:
                            energy = np.sum(np.array(x)**2 + np.array(y)**2)
                            max_val = max(np.max(np.abs(x)), np.max(np.abs(y)))
                            metrics_text += f"{i+1}. {title.split('-')[0]}\n"
                            metrics_text += f"   Energia: {energy:.1f}\n"
                            metrics_text += f"   Max: {max_val:.1f}\n\n"

                ax.set_title(title, color='white', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.tick_params(colors='white', labelsize=6)

                # Labels específicos por topologia
                if i == 0:  # Winding
                    ax.set_xlabel('Amp×cos(θ)', color='white', fontsize=7)
                    ax.set_ylabel('Amp×sin(θ)', color='white', fontsize=7)
                elif i == 1:  # Phase space
                    ax.set_xlabel('Sinal', color='white', fontsize=7)
                    ax.set_ylabel('Derivada', color='white', fontsize=7)
                elif i == 4:  # Frequency spiral
                    ax.set_xlabel('Freq×cos(θ)', color='white', fontsize=7)
                    ax.set_ylabel('Freq×sin(θ)', color='white', fontsize=7)
                elif i == 7:  # Wavelet
                    ax.set_xlabel('Escala', color='white', fontsize=7)
                    ax.set_ylabel('Coeficiente', color='white', fontsize=7)

            except Exception as e:
                ax.text(0.5, 0.5, f"Erro: {str(e)[:30]}",
                        transform=ax.transAxes, ha='center', color='red')
                ax.set_title(f"Topologia {i+1} - Erro",
                             color='red', fontsize=8)

        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", metrics_text)

        plt.tight_layout()
        self.canvas.draw()

    def cycle_topology(self):
        """Alterna entre topologias automaticamente"""
        self.current_topology = (
            self.current_topology + 1) % len(self.topologies)
        new_topology = self.topologies[self.current_topology]
        self.topology_var.set(new_topology)

    def test_all_topologies(self):
        """Teste com todas as topologias"""
        test_signal = self.generate_test_signal()
        test_signal = test_signal - np.mean(test_signal)

        self.visualize_all_topologies(test_signal)
        self.status_label.configure(text="Teste de Todas as Topologias - OK")

    def update_visualization(self):
        """Loop principal"""
        if not self.running:
            return

        try:
            signal = self.read_arduino_data()
            signal = signal - np.mean(signal)

            if len(signal) >= 51:
                signal = savgol_filter(signal, 51, 3)

            self.data_counter += 1

            self.visualize_all_topologies(signal)

            status = f"ATIVO - Frame {self.data_counter}"
            if self.ser:
                status += " | Arduino OK"
            else:
                status += " | Modo Teste"
            self.status_label.configure(text=status)

        except Exception as e:
            print(f"Erro: {e}")

        if self.running:
            # 5 FPS para estabilidade
            self.root.after(200, self.update_visualization)

    def start_visualization(self):
        if not self.running:
            self.running = True
            if not self.ser:
                self.setup_serial()
            self.update_visualization()

    def stop_visualization(self):
        self.running = False

def main():
    visualizer = MultiTopologyVisualizer()
    visualizer.setup_gui()

    instructions = """TOPOLOGIAS DISPONÍVEIS:

1. WINDING CURVE - Clássica frequência/amplitude
2. PHASE SPACE - Sinal vs derivada (dinâmica)
3. ATTRACTOR 3D - Embedding de Takens
4. HILBERT - Envelope complexo
5. FREQUENCY SPIRAL - Espiral espectral
6. POINCARÉ MAP - Cruzamentos por zero
7. RECURRENCE PLOT - Padrões repetidos
8. WAVELET - Análise multi-escala

Use 'Teste' para ver todas funcionando!
"""

    visualizer.metrics_text.insert("1.0", instructions)
    visualizer.root.mainloop()

    if visualizer.ser:
        visualizer.ser.close()


if __name__ == "__main__":
    main()
