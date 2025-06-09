import customtkinter as ctk
import tkinter as tk
from PIL import ImageTk, Image
""" 
organização dos frames e widgets 
"""


def create_section(parent, title, placeholder_text):
    """
    Cria uma seção responsiva com título e canvas
    
    Args:
        parent: Widget pai
        title: Título da seção
        placeholder_text: Texto placeholder para o canvas
    
    Returns:
        tuple: (frame, canvas) - frame da seção e canvas interno
    """
    # Frame principal da seção
    section_frame = ctk.CTkFrame(parent)

    # Configurar grid para ser responsivo
    section_frame.grid_columnconfigure(0, weight=1)
    section_frame.grid_rowconfigure(0, weight=0)  # Título (altura fixa)
    section_frame.grid_rowconfigure(1, weight=1)  # Canvas (expansível)

    # Título da seção
    title_label = ctk.CTkLabel(
        section_frame,
        text=title,
        font=ctk.CTkFont(size=16, weight="bold"),
        height=30
    )
    title_label.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

    # Frame para o canvas
    canvas_frame = ctk.CTkFrame(section_frame)
    canvas_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
    canvas_frame.grid_columnconfigure(0, weight=1)
    canvas_frame.grid_rowconfigure(0, weight=1)

    # Canvas para exibir imagens
    canvas = tk.Canvas(
        canvas_frame,
        bg="#3a3a3a",
        highlightthickness=1,
        highlightbackground="#565656",
        relief="flat", width=800,  # Largura mínima maior
        height=600
    )
    canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    # Expansão vertical e horizontal
    canvas_frame.pack_propagate(False)

    # Função para redimensionar o canvas e centralizar o texto
    def resize_canvas(event=None):
        canvas.delete("placeholder")
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:  # Garantir que o canvas foi renderizado
            center_x = canvas_width // 2
            center_y = canvas_height // 2

            canvas.create_text(
                center_x, center_y,
                text=placeholder_text,
                fill="#8a8a8a",
                font=("Arial", 12),
                tags="placeholder"
            )

    canvas.bind("<Configure>", resize_canvas)

    canvas.configure(width=400, height=250)  # valor base maior

    return section_frame, canvas


def update_canvas_image(canvas, image_data):
    """
    Atualiza o canvas com uma nova imagem
    
    Args:
        canvas: Canvas do tkinter
        image_data: Dados da imagem (PIL Image ou PhotoImage)
    """
    try:
        canvas.delete("all")

        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        if hasattr(image_data, 'resize'):
            img_width, img_height = image_data.size
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            scale = min(scale_x, scale_y) * 0.95

            new_width = int(img_width * scale)
            new_height = int(img_height * scale)

            resized_image = image_data.resize(
                (new_width, new_height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized_image)

            x = canvas_width // 2
            y = canvas_height // 2

            canvas.create_image(x, y, image=photo, anchor="center")
            canvas.image = photo

    except Exception as e:
        print(f"Erro ao atualizar canvas: {e}")
        canvas.create_text(
            canvas_width // 2, canvas_height // 2,
            text="Erro ao carregar imagem",
            fill="#ff6b6b",
            font=("Arial", 12)
        )


def create_responsive_button_frame(parent, buttons_config):
    """
    Cria um frame responsivo para botões
    
    Args:
        parent: Widget pai
        buttons_config: Lista de dicionários com configuração dos botões
                       [{"text": "Texto", "command": callback}, ...]
    
    Returns:
        frame: Frame contendo os botões
    """
    button_frame = ctk.CTkFrame(parent)

    num_buttons = len(buttons_config)
    for i in range(num_buttons):
        button_frame.grid_columnconfigure(i, weight=1)
    button_frame.grid_rowconfigure(0, weight=1)

    buttons = []
    for i, config in enumerate(buttons_config):
        button = ctk.CTkButton(
            button_frame,
            text=config["text"],
            command=config["command"],
            height=40,
            font=ctk.CTkFont(size=12)
        )
        button.grid(row=0, column=i, padx=5, pady=10, sticky="ew")
        buttons.append(button)

    return button_frame, buttons


