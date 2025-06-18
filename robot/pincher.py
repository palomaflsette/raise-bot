#!/usr/bin/env python
# (c) 2022 Wouter Caarls, PUC-RIO

"""Pincher driver."""

import numpy as np
from math import pi, sin, cos, atan2, sqrt
from .dynamixel import Arbotix
from transforms.transformations import *

class Pincher(Arbotix):
    def __init__(self, port=None):
        super(Pincher, self).__init__(port)
        self.l1 = 0.109
        self.l2 = 0.109
        self.l3 = 0.08
        self.zbase = 0.106+0.047

    def admissible(self, q):
        """Verifies if q is an admissible configuration."""
        if np.any(np.abs(q) > 2*pi/3):
           return False
        tr = self.fk(q)
        for i in range(1, len(tr)):
          if origin(get_frame(tr, i))[2] < 0.05:
            return False
        return True

    def fk(self, q):
        """Returns a list of coordinate frames corresponding
           to the configuration q."""
        H0b = H(T=[0,0,0.106])
        H10 = H(RZ(q[0]), [0,0,0.047])@H(RX(-pi/2))
        H21 = H(RZ(q[1]-pi/2))@H(T=[self.l1,0,0])
        H32 = H(RZ(q[2]))@H(T=[self.l2,0,0])
        H43 = H(RZ(q[3]))@H(T=[self.l3,0,0])
        He4 = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]).T
    
        return [H0b, H10, H21, H32, H43, He4]

    def jacobian(self, q):
        """Returns the analytical Jacobian at configuration q."""
        tr = self.fk(q)
        J = np.zeros((4, len(q)))
        # Linear part
        for i in range(len(q)):
            z = get_frame(tr, i)[0:3,2]
            d = origin(get_ee(tr)) - origin(get_frame(tr, i))
            J[0:3,i] = np.cross(z, d)

        # Angular part (angle in rotated XZ)
        J[3, 1:4] = 1
        return J

    def ik(self, x):
        """Returns the algebraic inverse kinematics solution for position x."""
        xp = np.copy(x)
        q = np.zeros(4)
        q[0] = atan2(x[1], x[0])

        # Rewrite to link 2 origin of 3-link planar arm in XZ
        phi = -x[3]
        z = x[2]-self.l3*sin(phi)-self.zbase
        x = sqrt(x[0]**2 + x[1]**2) - self.l3*cos(phi)

        # Siciliano, p. 92
        norm2 = x**2+z**2
        c2 = (norm2-self.l1**2-self.l2**2)/(2*self.l1*self.l2)
        if c2 < -1 or c2 > 1:
            print(f"[IK] Ponto descartado: c2={c2:.5f} fora do domínio válido")
            return None  

        s2 = -sqrt(1-c2**2) # Always arm up
        t2 = atan2(s2, c2)
        c1 = ((self.l1+self.l2*c2)*x+self.l2*s2*z)/norm2
        s1 = ((self.l1+self.l2*c2)*z-self.l2*s2*x)/norm2
        t1 = atan2(s1, c1)
        t3 = phi - t1 - t2

        # Rewrite to pincher angle configuration
        q[1] = pi/2-t1
        q[2] = -t2
        q[3] = -t3

        if not self.admissible(q):
            raise ValueError(str(q) + "is not a valid solution for " + str(xp))

        return q
    
    def try_ik_with_phi_range(pincher, x, y, z, phi_range_deg=(-90, 95), step_deg=5):
        from math import radians
        for deg in range(int(phi_range_deg[0]), int(phi_range_deg[1]), step_deg):
            phi = radians(deg)
            pose = [x, y, z, phi]
            try:
                q = pincher.ik(pose)
                print(f"Sucesso com phi = {deg}° → q (graus) = {np.degrees(q)}")
                return q
            except ValueError:
                continue
        print("Nenhum valor de phi no intervalo funcionou.")
        return None



    def pose(self, f):
        """Returns the pose of frame f. The orientation is the angle of the
           end effector frame around the base Y axis rotated by theta_1.
           Angle 0 points along the nominal base X axis."""
        x = np.zeros(4)
        x[0:3] = origin(f)
        fr = H(RZ(-atan2(x[1], x[0])))@f
        x[3] = -atan2(fr[0,0], fr[0,2])
        return x

    def within_workspace(self, p_robo, min_radius=0.12, max_radius=0.70, min_z=-0.15, max_z=0.30):
        """
        Checa se o ponto transformado (em metros) está dentro do workspace realista:
        → visível pela câmera e fisicamente alcançável pelo robô.
        """
        x, y, z = p_robo[:3]
        radius = np.sqrt(x**2 + y**2)
        
        dentro = (min_radius <= radius <= max_radius) and (min_z <= z <= max_z)
        if not dentro:
            print(f"[WORKSPACE REJECTION] Raio={radius:.3f}m, Altura z={z:.3f}m -> FORA")
        return dentro

