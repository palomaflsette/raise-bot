
import numpy as np
from math import pi, sin, cos, atan2, sqrt, radians
from .dynamixel import Arbotix
from transforms.transformations import HT,HZ,HX, origin, zaxis, get_ee, get_frame, normalize_angles


class Pincher(Arbotix):
    def __init__(self, port=None):
        super(Pincher, self).__init__(port)
        self.d1 = 0.106 + 0.047  # = 0.153
        self.a2 = 0.108
        self.a3 = 0.109
        self.a4 = 0.080 + 0.045  

        self.arm_ids = [1, 2, 3, 4]
        self.gripper_id = 5

    def fk(self, q):
        """
        Cinemática Direta ("O Corpo") - Implementação fiel do modelo DH do lab3.
        """
        theta1, theta2, theta3, theta4 = q[0], q[1], q[2], q[3]

        theta2_com_offset = theta2 + pi/2

        H01 = HZ(theta1) @ HT([0, 0, self.d1]) @ HX(-pi/2)
        H12 = HZ(theta2_com_offset) @ HT([self.a2, 0, 0])
        H23 = HZ(theta3) @ HT([self.a3, 0, 0])
        H3e = HZ(theta4) @ HT([self.a4, 0, 0])


        return [H01, H12, H23, H3e]

    def jacobian(self, q):
        """Calcula a Jacobiana geométrica (os "Nervos")."""
        tr = self.fk(q)
        J = np.zeros((3, len(q)))
        o_n = origin(get_ee(tr))

        z0 = np.array([0, 0, 1])
        J[:, 0] = np.cross(z0, o_n)

        # outras juntas
        for i in range(1, len(q)):
            h_i_minus_1 = get_frame(tr, i-1)
            o_i_minus_1 = origin(h_i_minus_1)
            z_i_minus_1 = zaxis(h_i_minus_1)
            J[:, i] = np.cross(z_i_minus_1, o_n - o_i_minus_1)
        return J

    def ik(self, target_pos, q_guess=np.zeros(4)):
        """
        Cinemática Inversa Diferencial (a "Mente").
        Acha os ângulos 'q' para chegar em 'target_pos'.
        """
        q = np.copy(q_guess)

        for _ in range(100):  
            current_pos = origin(get_ee(self.fk(q)))
            error = target_pos - current_pos

            if np.linalg.norm(error) < 1e-4:  
                if not self.admissible(q):
                    raise ValueError("Solução encontrada, mas não é segura.")
                return q  

            J = self.jacobian(q)
            lambda_sq = 0.01
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(3))

            q = q + J_pinv @ error
            q = normalize_angles(q)

        raise ValueError("IK não convergiu para uma solução.")

    def admissible(self, q):
        """Verifica se a configuração 'q' é segura."""
        q_limits = [2.9, 2.0, 2.0, 2.9]
        for i, angle in enumerate(q):
            if abs(angle) > q_limits[i]:
                return False
        tr = self.fk(q)
        for i in range(len(tr)):
          if origin(get_frame(tr, i))[2] < 0.01:
            return False
        return True


    def get_ik_solution(self, robot_target_pos):
        """Tenta encontrar uma solução de IK para a posição alvo."""
        try:
            q_guess = self.getangle(self.arm_ids)
            if not q_guess or len(q_guess) != 4:
                q_guess = np.zeros(4)
        except:
            q_guess = np.zeros(4)

        try:
            q_solution = self.ik(robot_target_pos, q_guess)
            print("[IK SUCESSO] Solução diferencial encontrada.")
            return q_solution
        except ValueError as e:
            # print(f"[FALHA DE IK] {e}")
            return None

    def within_workspace(self, p_robo):
        min_radius, max_radius = 0.10, 0.40
        min_z, max_z = 0.02, 0.45
        x, y, z = p_robo[:3]
        radius = np.sqrt(x**2 + y**2)
        return (min_radius <= radius <= max_radius) and (min_z <= z <= max_z)
