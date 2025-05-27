# main.py

from gui.main_gui import RaiseGui
from vision.rgb_stream import start_rgb_stream
from vision.depth_stream import start_depth_stream
from sound.piezo_stream import start_piezo_capture
from robot.movement import RoboticController


def main():
    # Inicialização dos módulos do sistema
    print("Inicializando módulos...")

    # Inicia GUI
    app = RaiseGui()
    app.mainloop()


if __name__ == "__main__":
    main()
