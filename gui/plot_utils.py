from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import customtkinter as ctk
import matplotlib
matplotlib.use("Agg")

"""Importar função do depth_stream atualizado"""
try:
    from vision.depth_stream import extract_stable_profile_line
except ImportError:
    """Fallback se não conseguir importar"""
    def extract_stable_profile_line(depth_frame, line_y=240, window_size=5):
        return depth_frame[line_y, :].astype(np.float32)


def render_profile_plot(depth_frame, target_widget, parent_gui):
    """
    Renderiza gráfico de perfil otimizado para objetos próximos
    """
    # Usar função otimizada para extrair linha estável
    z_raw = extract_stable_profile_line(depth_frame, line_y=240, window_size=7)

    z_raw[z_raw == 0] = np.nan # Converter zeros para nan
    
    # Filtrando apenas objetos próximos (100-400mm) com tolerância maior
    z_raw[(z_raw < 100) | (z_raw > 400)] = np.nan

    # Verificando se há pontos suficientes
    valid_count = np.count_nonzero(~np.isnan(z_raw))
    if valid_count < 15:
        print(f"[AVISO] Apenas {valid_count} pontos válidos. Pulando frame.")
        return

    # Aplicar suavização adicional
    z = np.copy(z_raw)

    # Interpolando melhor pontos faltantes
    valid_mask = ~np.isnan(z)
    if np.sum(valid_mask) > 10:
        x_coords = np.arange(len(z))
        valid_coords = x_coords[valid_mask]
        valid_values = z[valid_mask]

        try:
            # Interp. linear
            z = np.interp(x_coords, valid_coords, valid_values)

            # filtro gaussiano para suavizar - revisar depois
            z = gaussian_filter1d(z, sigma=2.0)

        except Exception as e:
            print(f"[ERRO] Falha na interpolação: {e}")
            return
    else:
        print("[AVISO] Poucos pontos válidos para interpolação.")
        return

    # Detectar região de interesse (objetos próximos)
    close_mask = (z > 100) & (z < 400)
    close_points = np.sum(close_mask)

    if close_points < 10:
        print(f"[AVISO] Apenas {close_points} pontos próximos detectados.")
        return

    """Calcular normais de forma mais estável
    Usar janela maior para gradiente mais suave"""
    dx = 2  # Espaçamento para os gradientes
    dz = np.gradient(z, dx)

    # Limitando gradientes extremos
    dz = np.clip(dz, -40, 40)
    # Suavizanso gradiente
    dz = gaussian_filter1d(dz, sigma=1.0)
    # Calculando normais 2D
    normals = np.stack([-dz, np.ones_like(dz)], axis=1)
    # Normalizando os vetores
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Evitar divisão por zero
    normals = normals / norms

    # Criar plot com informações de debug
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), dpi=80)

    # Plot principal - perfil de profundidade
    x_coords = np.arange(len(z))
    ax1.plot(x_coords, z, color="cyan", linewidth=2,
             label="Perfil de Profundidade")
    ax1.fill_between(x_coords, z, alpha=0.3, color="cyan")

    # Destacando região de objetos próximos
    close_indices = np.where(close_mask)[0]
    if len(close_indices) > 0:
        ax1.scatter(close_indices, z[close_indices],
                    color="yellow", s=3, alpha=0.7, label="Objetos Próximos")

    # plot das normais apenas na região próxima (reduzir densidade visual)
    step = max(1, len(close_indices) // 20)  # Máximo de 20 vetores
    sample_indices = close_indices[::step]

    if len(sample_indices) > 0:
        ax1.quiver(
            sample_indices,
            z[sample_indices],
            normals[sample_indices, 0] * 30,  # Escalar para visualização
            -normals[sample_indices, 1] * 30,
            color="red", scale=1, scale_units='xy', angles='xy',
            width=0.003, alpha=0.8, label="Normais"
        )

    ax1.set_ylim(100, 700)
    ax1.set_xlim(0, len(z))
    ax1.set_title(
        f"Perfil de Superfície - {valid_count} pontos válidos, {close_points} próximos")
    ax1.set_xlabel("Pixel (X)")
    ax1.set_ylabel("Profundidade (mm)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)

    # Plot secundário - gradientes e qualidade
    ax2.plot(x_coords, dz, color="orange",
             linewidth=1, label="Gradiente dZ/dx")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylim(-30, 30)
    ax2.set_xlim(0, len(z))
    ax2.set_title("Gradiente da Superfície (Derivada)")
    ax2.set_xlabel("Pixel (X)")
    ax2.set_ylabel("Gradiente (mm/pixel)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)

    # estatísticas como texto
    stats_text = f"Profundidade média: {np.mean(z[close_mask]):.1f}mm\n"
    stats_text += f"Desvio padrão: {np.std(z[close_mask]):.1f}mm\n"
    stats_text += f"Range: {np.min(z[close_mask]):.0f}-{np.max(z[close_mask]):.0f}mm"

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=8)

    plt.tight_layout()

    # Converter para imagem e exibir
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    imgtk = ctk.CTkImage(light_image=img, size=(440, 300)
                         )  # Altura maior para 2 subplots
    target_widget.configure(image=imgtk, text="")
    target_widget.image = imgtk
    plt.close(fig)


def compute_surface_normals_3d(depth_frame, focal_length=525.0, baseline=75.0):
    """
    Calcula normais 3D da superfície a partir do mapa de profundidade completo
    Útil para análise mais detalhada da superfície
    """
    # Converter para float e filtrar
    depth = depth_frame.astype(np.float32)
    depth[depth == 0] = np.nan

    # Calcular gradientes em X e Y
    grad_x = np.gradient(depth, axis=1)
    grad_y = np.gradient(depth, axis=0)
    # Limitar gradientes extremos
    grad_x = np.clip(grad_x, -50, 50)
    grad_y = np.clip(grad_y, -50, 50)
    # Calcular normais 3D
    # Normal = (-dZ/dx, -dZ/dy, 1)
    normals = np.stack([-grad_x, -grad_y, np.ones_like(depth)], axis=2)

    # Normalizar
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms

    return normals


def analyze_surface_curvature(depth_profile):
    """
    Analisa curvatura da superfície para detectar características
    """
    if len(depth_profile) < 10:
        return None

    # Calcular primeira e segunda derivadas
    first_deriv = np.gradient(depth_profile)
    second_deriv = np.gradient(first_deriv)

    # Suavizar para reduzir ruído
    first_deriv = gaussian_filter1d(first_deriv, sigma=1.0)
    second_deriv = gaussian_filter1d(second_deriv, sigma=1.0)

    # Calcular curvatura: κ = |f''| / (1 + f'^2)^(3/2)
    curvature = np.abs(second_deriv) / np.power(1 + first_deriv**2, 1.5)

    # Detectar pontos de alta curvatura (bordas, cantos)
    high_curvature_threshold = np.percentile(
        curvature[~np.isnan(curvature)], 90)
    high_curvature_points = curvature > high_curvature_threshold

    return {
        'curvature': curvature,
        'high_curvature_points': high_curvature_points,
        'mean_curvature': np.nanmean(curvature),
        'max_curvature': np.nanmax(curvature)
    }


def extract_object_boundaries(depth_frame, min_depth=100, max_depth=430):
    """
    Detecta bordas de objetos próximos para melhor compreensão da cena
    """
    # Filtrar apenas objetos próximos
    filtered = depth_frame.astype(np.float32)
    filtered[(filtered < min_depth) | (filtered > max_depth)] = np.nan

    # Converter para formato adequado para detecção de bordas
    depth_for_edges = np.copy(filtered)
    depth_for_edges[np.isnan(depth_for_edges)] = 0
    depth_for_edges = depth_for_edges.astype(np.uint8)

    # Aplicar detecção de bordas Canny
    edges = cv2.Canny(depth_for_edges, 50, 150)

    # Aplicar operações morfológicas para limpar bordas
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return edges


def render_enhanced_depth_analysis(depth_frame, target_widget, parent_gui):
    """
    Versão melhorada do render_profile_plot com análises adicionais
    """
    # Análise do perfil principal
    z_profile = extract_stable_profile_line(
        depth_frame, line_y=240, window_size=7)

    # Análise de curvatura
    curvature_data = analyze_surface_curvature(z_profile)

    # Detecção de bordas
    edges = extract_object_boundaries(depth_frame)

    # Criar visualização completa
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=80)

    # Plot 1: Perfil de profundidade com normais
    ax1 = axes[0, 0]
    valid_mask = ~np.isnan(z_profile) & (z_profile > 99) & (z_profile < 430)
    x_coords = np.arange(len(z_profile))

    if np.sum(valid_mask) > 10:
        ax1.plot(x_coords[valid_mask],
                 z_profile[valid_mask], 'b-', linewidth=2)
        ax1.fill_between(x_coords[valid_mask],
                         z_profile[valid_mask], alpha=0.3)

        # Calcular e plotar normais
        dz = np.gradient(z_profile[valid_mask])
        dz = gaussian_filter1d(dz, sigma=1.0)
        normals_2d = np.stack([-dz, np.ones_like(dz)], axis=1)
        normals_2d = normals_2d / \
            np.linalg.norm(normals_2d, axis=1, keepdims=True)

        # Plotar algumas normais
        step = max(1, len(x_coords[valid_mask]) // 15)
        sample_idx = x_coords[valid_mask][::step]
        ax1.quiver(sample_idx, z_profile[sample_idx],
                   normals_2d[::step, 0] * 20, -normals_2d[::step, 1] * 20,
                   color='red', scale=1, scale_units='xy', angles='xy', width=0.002)

    ax1.set_title("Perfil de Profundidade com Normais")
    ax1.set_ylabel("Profundidade (mm)")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Curvatura
    ax2 = axes[0, 1]
    if curvature_data:
        ax2.plot(curvature_data['curvature'], 'g-', linewidth=1)
        high_curv_points = curvature_data['high_curvature_points']
        if np.any(high_curv_points):
            ax2.scatter(np.where(high_curv_points)[0],
                        curvature_data['curvature'][high_curv_points],
                        color='red', s=20, alpha=0.7)
    ax2.set_title("Análise de Curvatura")
    ax2.set_ylabel("Curvatura")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Mapa de profundidade
    ax3 = axes[1, 0]
    depth_display = np.copy(depth_frame).astype(np.float32)
    depth_display[(depth_display < 100) | (depth_display > 430)] = np.nan
    im3 = ax3.imshow(depth_display, cmap='viridis', aspect='auto')
    ax3.axhline(y=240, color='red', linestyle='--',
                alpha=0.7, label='Linha de análise')
    ax3.set_title("Mapa de Profundidade Filtrado")
    ax3.legend()

    # Plot 4: Detecção de bordas
    ax4 = axes[1, 1]
    ax4.imshow(edges, cmap='gray', aspect='auto')
    ax4.set_title("Bordas de Objetos Próximos")

    plt.tight_layout()

    # Converter e exibir
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    imgtk = ctk.CTkImage(light_image=img, size=(440, 350))
    target_widget.configure(image=imgtk, text="")
    target_widget.image = imgtk
    plt.close(fig)

def render_depth_colormap(depth_frame, target_widget, parent_gui, min_depth=100, max_depth=470):
    """
    Renderiza a imagem de profundidade com colormap visível
    e cores sólidas para regiões além do alcance útil.
    """
    depth = depth_frame.astype(np.float32)
    depth[depth == 0] = np.nan
    depth_display = np.copy(depth)

    # Tudo acima do limite será vermelho escuro
    red_color = np.array([0, 0, 128], dtype=np.uint8)  # BGR
    colormap = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    # Normalizar para 8 bits para aplicar colormap
    clipped = np.clip(depth_display, min_depth, max_depth)
    norm = ((clipped - min_depth) / (max_depth - min_depth)) * 255
    norm[np.isnan(depth)] = 0
    norm = norm.astype(np.uint8)

    jet = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    valid_mask = (depth >= min_depth) & (depth <= max_depth)
    colormap[valid_mask] = jet[valid_mask]
    colormap[~valid_mask] = red_color  # Fora da faixa

    img = Image.fromarray(cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB))
    imgtk = ctk.CTkImage(light_image=img, size=(440, 300))
    target_widget.configure(image=imgtk, text="")
    target_widget.image = imgtk
