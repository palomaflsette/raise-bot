import numpy as np
import cv2
from robot.pincher import Pincher

import numpy as np


def get_candidate_points(depth_frame, pincher, T_cam_to_robo, last_valid_point=None, num_points=5, min_depth=100, max_depth=430):
     h, w = depth_frame.shape
     mask = (depth_frame > min_depth) & (depth_frame < max_depth)
     valid_indices = np.argwhere(mask)

     if len(valid_indices) == 0:
          print("[INFO] Sem pixels válidos. Usando último ponto válido.")
          return [last_valid_point] if last_valid_point is not None else []

     np.random.shuffle(valid_indices)
     print(f"[DEBUG] Total de pixels válidos: {len(valid_indices)}")

     for v, u in valid_indices:
          z = depth_frame[v, u] / 1000.0
          fx, fy = 615, 615
          cx, cy = 320, 240
          x = (u - cx) * z / fx
          y = (v - cy) * z / fy
          p_cam_h = np.array([x, y, z, 1])
          p_robo_h = T_cam_to_robo @ p_cam_h
          x_r, y_r, z_r = p_robo_h[:3]

          try:
               if not pincher.within_workspace(np.array([x_r, y_r, z_r])):
                    continue

               pose = [x_r, y_r, z_r, 0.0]
               q = pincher.ik(pose)
               if q is None or not pincher.admissible(q):
                    continue

               ponto_final = (x, y, z, u, v)
               print(f"[FINAL] Novo ponto fixado: {ponto_final}")
               return [ponto_final]

          except Exception as e:
               print(f"[EXCEPTION] Falha ao processar ponto candidato: {e}")
               continue

     print("[INFO] Nenhum ponto novo. Mantendo anterior.")
     return [last_valid_point] if last_valid_point is not None else []




def draw_targets_on_rgb(rgb_frame, points):
     for _, _, _, u, v in points:
        if u is not None and v is not None:
            cv2.circle(rgb_frame, (int(u), int(v)), 8, (0, 255, 0), 2)
     return rgb_frame

