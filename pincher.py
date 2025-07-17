#!/usr/bin/env python
# (c) 2022 Wouter Caarls, PUC-RIO

"""Pincher driver."""

import numpy as np
from math import pi, sin, cos, atan2, sqrt
from dynamixel import Arbotix
from transformations2 import *

class Pincher(Arbotix):
    def __init__(self, port=None):
        super(Pincher, self).__init__(port)
        self.l1 = 0.109
        self.l2 = 0.109
        self.l3 = 0.12
        self.zbase = 0.106+0.047

    def admissible(self, q):
        """Verifies if q is an admissible configuration."""
        
        # if np.abs(q[0]) > 3.1:
        #          return False
        if np.abs(q[1]) > 3*pi/4:
                 return False
        if np.abs(q[2]) > 3*pi/4:
                 return False
        if np.abs(q[3]) > pi/2:
                 return False
        tr = self.fk(q)
        for i in range(1, len(tr)):
          if origin(get_frame(tr, i))[2] < 0.01:
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

    def pose(self, f):
        """Returns the pose of frame f. The orientation is the angle of the
           end effector frame around the base Y axis rotated by theta_1.
           Angle 0 points along the nominal base X axis."""
        x = np.zeros(4)
        x[0:3] = origin(f)
        fr = H(RZ(-atan2(x[1], x[0])))@f
        x[3] = -atan2(fr[0,0], fr[0,2])
        return x

