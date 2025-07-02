import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


class CameraRobotSimulator:
    def __init__(self):

        # Matriz intrínseca simulada da câmera
        self.fx, self.fy = 400, 400
        self.cx, self.cy = 320, 240
        self.K = np.array([[self.fx, 0, self.cx],
                          [0, self.fy, self.cy],
                          [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.K)

        self.camera_tilt = 8  
        self.camera_height = 0.500 
        self.camera_position = np.array([0.005, 0.004, self.camera_height])

        self.current_click = None
        self.current_depth = 0.45
        self.trajectory_points = []
        self.animation_frame = 0

        self.setup_interface()
        self.update_transformations()

    def setup_interface(self):
        """Configura a interface gráfica"""
        self.fig = plt.figure(figsize=(16, 10))

        self.ax_camera = self.fig.add_subplot(121)
        self.ax_3d = self.fig.add_subplot(122, projection='3d')

        self.ax_camera.set_xlim(0, 640)
        self.ax_camera.set_ylim(480, 0)  
        self.ax_camera.set_aspect('equal')
        self.ax_camera.set_title(
            'Câmera Virtual - Clique para selecionar ponto', fontsize=12, fontweight='bold')
        self.ax_camera.set_xlabel('u (pixels)')
        self.ax_camera.set_ylabel('v (pixels)')

        
        self.draw_virtual_scene()

        self.ax_3d.set_title(
            'Visualização 3D - Sistema Câmera-Robô', fontsize=12, fontweight='bold')
        self.setup_3d_scene()

        self.setup_controls()

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def draw_virtual_scene(self):
        """Desenha uma cena virtual na câmera"""
        mesa = Rectangle((100, 200), 440, 200, linewidth=2,
                         edgecolor='brown', facecolor='burlywood', alpha=0.3)
        self.ax_camera.add_patch(mesa)

        objects = [
            {'pos': (200, 250), 'size': 20, 'color': 'red', 'label': 'Obj 1'},
            {'pos': (350, 280), 'size': 25, 'color': 'blue', 'label': 'Obj 2'},
            {'pos': (450, 320), 'size': 18, 'color': 'green', 'label': 'Obj 3'}
        ]

        for obj in objects:
            circle = plt.Circle(obj['pos'], obj['size'],
                                color=obj['color'], alpha=0.6)
            self.ax_camera.add_patch(circle)
            self.ax_camera.text(obj['pos'][0], obj['pos'][1]-40, obj['label'],
                                ha='center', fontsize=8, fontweight='bold')

        self.ax_camera.axvline(x=self.cx, color='gray',
                               linestyle='--', alpha=0.5)
        self.ax_camera.axhline(y=self.cy, color='gray',
                               linestyle='--', alpha=0.5)

        self.ax_camera.grid(True, alpha=0.3)

    def setup_3d_scene(self):
        """Configura a visualização 3D baseada no setup real"""
        self.ax_3d.set_xlim([-0.15, 0.35])
        self.ax_3d.set_ylim([-0.15, 0.35])
        self.ax_3d.set_zlim([-0.1, 0.6])  
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')

        self.draw_3d_table()

    def draw_3d_table(self):
        """Desenha uma mesa 3D na altura correta"""
        xx, yy = np.meshgrid(np.linspace(-0.1, 0.3, 10),
                             np.linspace(-0.1, 0.3, 10))
        zz = np.ones_like(xx) * 0.0  # mesa na altura da base do robô
        self.ax_3d.plot_surface(xx, yy, zz, alpha=0.2, color='brown')

    def setup_controls(self):
        """Configura controles deslizantes"""
        
        ax_depth = plt.axes([0.15, 0.02, 0.3, 0.03])
        ax_tilt = plt.axes([0.55, 0.02, 0.3, 0.03])

        self.slider_depth = Slider(ax_depth, 'Profundidade (m)', 0.2, 0.8,
                                   valinit=self.current_depth, valfmt='%.3f')
        self.slider_tilt = Slider(ax_tilt, 'Inclinação Câmera (°)', 0, 15,
                                  valinit=self.camera_tilt, valfmt='%.1f')

        self.slider_depth.on_changed(self.update_depth)
        self.slider_tilt.on_changed(self.update_tilt)

        ax_clear = plt.axes([0.02, 0.85, 0.08, 0.04])
        self.btn_clear = Button(ax_clear, 'Limpar')
        self.btn_clear.on_clicked(self.clear_trajectory)

    def update_transformations(self):
        """Atualiza as matrizes de transformação - VERSÃO CORRIGIDA"""
        theta = np.radians(self.camera_tilt)

        self.R_cam = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        

        self.T_cam_to_robot = np.eye(4)
        self.T_cam_to_robot[:3, :3] = self.R_cam  # Rotação direta
        self.T_cam_to_robot[:3, 3] = self.camera_position  # Posição da câmera

        self.T_robot_to_cam = np.linalg.inv(self.T_cam_to_robot)

    def pixel_to_camera_coords(self, u, v, depth):
        """Converte coordenadas de pixel para coordenadas 3D da câmera"""
        pixel_hom = np.array([u, v, 1])
        ray_direction = self.K_inv @ pixel_hom

        p_cam = depth * ray_direction

        return p_cam

    def camera_to_robot_coords(self, p_cam):
        """Transforma coordenadas da câmera para o sistema do robô - VERSÃO CORRIGIDA"""
        p_cam_hom = np.append(p_cam, 1)
        p_robot_hom = self.T_cam_to_robot @ p_cam_hom
        p_robot = p_robot_hom[:3]

        return p_robot

    def robot_to_camera_coords(self, p_robot):
        """Transforma coordenadas do robô para o sistema da câmera"""
        p_robot_hom = np.append(p_robot, 1)
        p_cam_hom = self.T_robot_to_cam @ p_robot_hom
        p_cam = p_cam_hom[:3]

        return p_cam

    def project_to_image(self, p_cam):
        """Projeta ponto 3D da câmera para coordenadas de pixel"""
        if p_cam[2] <= 0:  
            return None

        pixel_hom = self.K @ (p_cam / p_cam[2])
        return pixel_hom[:2]

    def on_click(self, event):
        """Manipula cliques na câmera virtual"""
        if event.inaxes != self.ax_camera:
            return

        u, v = int(event.xdata), int(event.ydata)
        self.current_click = (u, v)

        p_cam = self.pixel_to_camera_coords(u, v, self.current_depth)
        p_robot = self.camera_to_robot_coords(p_cam)

        p_cam_check = self.robot_to_camera_coords(p_robot)
        pixel_check = self.project_to_image(p_cam_check)

        print(f"\n=== DEBUG TRANSFORMAÇÃO ===")
        print(f"Pixel clicado: ({u}, {v})")
        print(f"Ponto câmera: {p_cam}")
        print(f"Ponto robô: {p_robot}")
        print(f"Verificação - Câmera: {p_cam_check}")
        if pixel_check is not None:
            print(
                f"Verificação - Pixel: ({pixel_check[0]:.1f}, {pixel_check[1]:.1f})")
        print("="*30)

        self.trajectory_points.append({
            'pixel': (u, v),
            'p_cam': p_cam,
            'p_robot': p_robot,
            'timestamp': len(self.trajectory_points)
        })

        self.update_visualization()

    def update_visualization(self):
        """Atualiza as visualizações"""
        self.ax_camera.clear()
        self.draw_virtual_scene()

        for i, point in enumerate(self.trajectory_points):
            u, v = point['pixel']
            color = plt.cm.viridis(i / max(len(self.trajectory_points)-1, 1))
            self.ax_camera.scatter(
                u, v, s=100, c=[color], marker='x', linewidths=3)
            self.ax_camera.text(
                u+15, v-15, f'P{i+1}', fontweight='bold', color=color)

        self.ax_3d.clear()
        self.setup_3d_scene()
        self.draw_coordinate_systems()
        self.draw_trajectory_3d()

        self.fig.canvas.draw()

    def draw_coordinate_systems(self):
        """Desenha os sistemas de coordenadas conforme setup real"""
        cam_origin = self.camera_position
        cam_axes = self.R_cam @ np.eye(3) * 0.08

        colors = ['red', 'green', 'blue']
        labels = ['X_cam', 'Y_cam', 'Z_cam']

        for i, (color, label) in enumerate(zip(colors, labels)):
            self.ax_3d.quiver(*cam_origin, *cam_axes[:, i],
                              color=color, alpha=0.8, arrow_length_ratio=0.1, linewidth=2)

        robot_base = np.array([0, 0, 0])  # Base do robô
        robot_work = np.array([0, 0, 0.15])  # Centro de trabalho do robô
        robot_axes = np.eye(3) * 0.08

        for i, (color, label) in enumerate(zip(colors, labels)):
            self.ax_3d.quiver(*robot_base, *robot_axes[:, i],
                              color=color, alpha=0.4, linestyle=':',
                              arrow_length_ratio=0.1, linewidth=1)

        for i, (color, label) in enumerate(zip(colors, labels)):
            self.ax_3d.quiver(*robot_work, *robot_axes[:, i],
                              color=color, alpha=0.8, linestyle='--',
                              arrow_length_ratio=0.1, linewidth=2)

        self.ax_3d.scatter(*cam_origin, color='black',
                           s=120, marker='o', label='Câmera')
        self.ax_3d.scatter(*robot_base, color='gray', s=100,
                           marker='s', label='Base Robô')
        self.ax_3d.scatter(*robot_work, color='orange', s=120,
                           marker='^', label='Centro Trabalho Robô')

        self.ax_3d.plot([cam_origin[0], robot_work[0]],
                        [cam_origin[1], robot_work[1]],
                        [cam_origin[2], robot_work[2]],
                        'k--', alpha=0.3, linewidth=1)

    def draw_trajectory_3d(self):
        """Desenha a trajetória no espaço 3D com configuração correta"""
        if not self.trajectory_points:
            return

        cam_origin = self.camera_position
        robot_work = np.array([0, 0, 0.15])

        for i, point in enumerate(self.trajectory_points):
            color = plt.cm.viridis(i / max(len(self.trajectory_points)-1, 1))

            p_cam_global = cam_origin + self.R_cam @ point['p_cam']
            self.ax_3d.scatter(*p_cam_global, color=color,
                               s=80, alpha=0.7, marker='o', label=f'Câmera P{i+1}' if i < 3 else "")

            self.ax_3d.plot([cam_origin[0], p_cam_global[0]],
                            [cam_origin[1], p_cam_global[1]],
                            [cam_origin[2], p_cam_global[2]],
                            '--', color='gray', alpha=0.4)

        for i, point in enumerate(self.trajectory_points):
            color = plt.cm.plasma(i / max(len(self.trajectory_points)-1, 1))
            self.ax_3d.scatter(*point['p_robot'], color=color, s=120, alpha=0.9, marker='^',
                               edgecolors='black', linewidth=1, label=f'Robô P{i+1}' if i < 3 else "")

            self.ax_3d.plot([robot_work[0], point['p_robot'][0]],
                            [robot_work[1], point['p_robot'][1]],
                            [robot_work[2], point['p_robot'][2]],
                            '-', color='red', alpha=0.8, linewidth=2)

            self.ax_3d.text(point['p_robot'][0], point['p_robot'][1], point['p_robot'][2] + 0.02,
                            f'P{i+1}', fontsize=10, fontweight='bold')

        if len(self.trajectory_points) > 1:
            robot_points = np.array([p['p_robot']
                                    for p in self.trajectory_points])
            for i in range(len(robot_points)-1):
                p1, p2 = robot_points[i], robot_points[i+1]
                self.ax_3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                'k-', alpha=0.6, linewidth=2)

        self.ax_3d.legend()

    def update_depth(self, val):
        """Atualiza a profundidade atual"""
        self.current_depth = val

    def update_tilt(self, val):
        """Atualiza a inclinação da câmera"""
        self.camera_tilt = val
        self.update_transformations()
        self.update_visualization()

    def clear_trajectory(self, event):
        """Limpa a trajetória"""
        self.trajectory_points = []
        self.update_visualization()

    def get_robot_command(self, point_idx):
        """Gera comando para o robô (simulado)"""
        if point_idx >= len(self.trajectory_points):
            return None

        point = self.trajectory_points[point_idx]
        p_robot = point['p_robot']

        return {
            'position': p_robot,
            'orientation': [0, 0, 0], 
            'timestamp': point['timestamp']
        }

    def print_status(self):
        """Imprime status atual do sistema"""
        print("\n" + "="*50)
        print("STATUS DO SISTEMA CÂMERA-ROBÔ")
        print("="*50)
        print(f"Inclinação da câmera: {self.camera_tilt:.1f}°")
        print(f"Profundidade atual: {self.current_depth:.3f}m")
        print(f"Pontos na trajetória: {len(self.trajectory_points)}")

        if self.trajectory_points:
            print("\nÚltimo ponto:")
            last = self.trajectory_points[-1]
            print(f"  Pixel: ({last['pixel'][0]}, {last['pixel'][1]})")
            print(
                f"  Câmera: ({last['p_cam'][0]:.3f}, {last['p_cam'][1]:.3f}, {last['p_cam'][2]:.3f})")
            print(
                f"  Robô: ({last['p_robot'][0]:.3f}, {last['p_robot'][1]:.3f}, {last['p_robot'][2]:.3f})")


if __name__ == "__main__":
    print("Simulador Câmera-Robô Iniciado!")
    print("Clique na câmera virtual (esquerda) para selecionar pontos")
    print("Use os controles deslizantes para ajustar parâmetros")
    print("Botão 'Limpar' remove todos os pontos")
    print("Debug da transformação será exibido no console")

    simulator = CameraRobotSimulator()
    plt.show()

    simulator.print_status()
