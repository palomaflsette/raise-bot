import numpy as np
import cv2
from robot.pincher import Pincher

import numpy as np

def extract_mask_bowls(rgb_image):
     hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
     lower_blue = np.array([90, 60, 50])
     upper_blue = np.array([135, 255, 255])
     mask = cv2.inRange(hsv, lower_blue, upper_blue)
     mask = cv2.medianBlur(mask, 7)
     return mask


def get_candidate_points(depth_frame, pincher, T_cam_to_robo, rgb_frame=None, last_valid_point=None, num_points=3, min_depth=100, max_depth=430):
     h, w = depth_frame.shape
     mask = (depth_frame > min_depth) & (depth_frame < max_depth)
     
     if rgb_frame is not None:
          bowl_mask = extract_mask_bowls(rgb_frame)
          bowl_mask = cv2.resize(
              bowl_mask, (depth_frame.shape[1], depth_frame.shape[0]))

          mask &= (bowl_mask > 0)

     valid_indices = np.argwhere(mask)

     if len(valid_indices) == 0:
          return [last_valid_point] if last_valid_point is not None else []

     np.random.shuffle(valid_indices)

     pontos_validos = []

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
               dz_dx = depth_frame[v, min(u+1, w-1)] - depth_frame[v, max(u-1, 0)]
               dz_dy = depth_frame[min(v+1, h-1), u] - depth_frame[max(v-1, 0), u]
               norm_angle = np.degrees(np.arctan2(np.hypot(dz_dx, dz_dy), 2))
               if norm_angle > 30:
                    continue

               if not pincher.within_workspace(np.array([x_r, y_r, z_r])):
                    continue

               pose = [x_r, y_r, z_r, 0.0]
               q = pincher.ik(pose)
               if q is None or not pincher.admissible(q):
                    continue

               pontos_validos.append((x, y, z, u, v))

               if len(pontos_validos) >= num_points:
                    break

          except:
               continue

     if len(pontos_validos) > 0:
          print(f"[FIXO] {len(pontos_validos)} ponto(s) fixados.")
          return pontos_validos
     else:
          print("[INFO] Nenhum ponto novo. Mantendo anterior.")
          return [last_valid_point] if last_valid_point else []




def draw_targets_on_rgb(rgb_frame, points):
     for _, _, _, u, v in points:
        if u is not None and v is not None:
            cv2.circle(rgb_frame, (int(u), int(v)), 8, (0, 255, 0), 2)
     return rgb_frame

