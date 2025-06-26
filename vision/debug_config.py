"""
Configurações e cenários de teste para o modo debug
"""

import numpy as np
import json
import os
from datetime import datetime


class DebugConfig:
    def __init__(self):
        self.config_file = "debug_settings.json"
        self.load_config()

    def load_config(self):
        """Carrega configurações do arquivo JSON"""
        default_config = {
            "scenarios": {
                "moving_object": {
                    "description": "Objeto cilíndrico se movendo",
                    "background_depth": 400,
                    "object_depth": 250,
                    "object_radius": 50,
                    "movement_amplitude": 150,
                    "noise_level": 0.1
                },
                "surface_wave": {
                    "description": "Superfície ondulada",
                    "base_depth": 300,
                    "wave_amplitude": 50,
                    "wave_frequency": 4,
                    "noise_level": 0.05
                },
                "step_surface": {
                    "description": "Superfície com degraus",
                    "background_depth": 350,
                    "step_depths": [250, 200, 150],
                    # Proporções da largura
                    "step_positions": [0.25, 0.5, 0.75],
                    "noise_level": 0.03
                },
                "noisy_data": {
                    "description": "Dados com muito ruído",
                    "base_depth": 300,
                    "wave_amplitude": 80,
                    "noise_level": 0.3,
                    "invalid_pixel_rate": 0.05
                },
                "real_objects": {
                    "description": "Simulação de objetos reais",
                    "background_depth": 450,
                    "box_depth": 200,
                    "cylinder_depth": 180,
                    "noise_level": 0.08
                },
                "calibration": {
                    "description": "Padrão de calibração",
                    "grid_size": 40,
                    "depth_a": 250,
                    "depth_b": 350,
                    "noise_level": 0.02
                }
            },
            "camera_params": {
                "width": 640,
                "height": 480,
                "focal_length": 525.0,
                "baseline": 75.0,
                "min_depth": 100,
                "max_depth": 500
            },
            "simulation": {
                "fps": 30,
                "speed_multiplier": 1.0,
                "auto_cycle_scenarios": False,
                "cycle_interval": 10.0  # segundos
            },
            "analysis": {
                "profile_line_y": 240,
                "profile_window_size": 7,
                "close_object_min": 100,
                "close_object_max": 400,
                "gradient_limit": 40
            }
        }

        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    self.config = self._merge_configs(
                        default_config, loaded_config)
            else:
                self.config = default_config
                self.save_config()
        except Exception as e:
            print(f"Erro ao carregar configuração: {e}")
            self.config = default_config

    def save_config(self):
        """Salva configurações no arquivo JSON"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Erro ao salvar configuração: {e}")

    def _merge_configs(self, default, loaded):
        """Merge configurações carregadas com defaults"""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def get_scenario_config(self, scenario_name):
        """Retorna configuração de um cenário específico"""
        return self.config["scenarios"].get(scenario_name, {})

    def get_camera_params(self):
        """Retorna parâmetros da câmera"""
        return self.config["camera_params"]

    def get_simulation_params(self):
        """Retorna parâmetros da simulação"""
        return self.config["simulation"]

    def get_analysis_params(self):
        """Retorna parâmetros de análise"""
        return self.config["analysis"]

    def update_scenario_config(self, scenario_name, new_config):
        """Atualiza configuração de um cenário"""
        if scenario_name in self.config["scenarios"]:
            self.config["scenarios"][scenario_name].update(new_config)
            self.save_config()

    def create_test_sequence(self):
        """Cria uma sequência de teste automática"""
        sequence = [
            {"scenario": "calibration", "duration": 5,
                "description": "Verificar padrão de calibração"},
            {"scenario": "moving_object", "duration": 10,
                "description": "Testar detecção de objeto em movimento"},
            {"scenario": "step_surface", "duration": 8,
                "description": "Testar detecção de bordas"},
            {"scenario": "surface_wave", "duration": 8,
                "description": "Testar superfície contínua"},
            {"scenario": "noisy_data", "duration": 6,
                "description": "Testar robustez com ruído"},
            {"scenario": "real_objects", "duration": 12,
                "description": "Testar cenário realista"}
        ]
        return sequence

    def generate_test_report_template(self):
        """Gera template para relatório de teste"""
        template = {
            "test_date": "",
            "test_duration": 0,
            "scenarios_tested": [],
            "performance_metrics": {
                "avg_fps": 0,
                "frame_drops": 0,
                "processing_time_avg": 0
            },
            "algorithm_performance": {
                "profile_extraction": {"success_rate": 0, "avg_points": 0},
                "normal_calculation": {"success_rate": 0, "stability": 0},
                "edge_detection": {"precision": 0, "recall": 0}
            },
            "issues_found": [],
            "recommendations": []
        }
        return template


class TestDataGenerator:
     """Gerador de dados de teste específicos"""
    
     @staticmethod
     def generate_edge_test_report():
          """Gera um relatório de teste simulado para cenários com bordas"""
          report = {
               "test_date": datetime.now().isoformat(),
               "test_duration": 45,
               "scenarios_tested": ["step_surface", "calibration"],
               "performance_metrics": {
                    "avg_fps": 28.6,
                    "frame_drops": 3,
                    "processing_time_avg": 35.4  # em milissegundos
               },
               "algorithm_performance": {
                    "profile_extraction": {"success_rate": 0.97, "avg_points": 620},
                    "normal_calculation": {"success_rate": 0.94, "stability": 0.91},
                    "edge_detection": {"precision": 0.89, "recall": 0.92}
               },
               "issues_found": [
                    "Oscilações nas normais em regiões planas",
                    "Ruído extremo afeta precisão da borda em 'noisy_data'"
               ],
               "recommendations": [
                    "Ajustar filtro gaussiano na extração de perfil",
                    "Adicionar validação cruzada de normais para bordas"
               ]
          }
          return report

