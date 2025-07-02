#!/usr/bin/env python
# (c) 2022 Wouter Caarls, PUC-RIO

import matplotlib.pyplot as plt
import numpy as np
import math


def RX(phi=0):
    """Returns a 3x3 rotation matrix about the Z axis."""
    return np.array([[1, 0, 0], [0, math.cos(phi), math.sin(phi)], [0, -math.sin(phi), math.cos(phi)]]).T


def RY(phi=0):
    """Returns a 3x3 rotation matrix about the Y axis."""
    return np.array([[math.cos(phi), 0, -math.sin(phi)], [0, 1, 0], [math.sin(phi), 0, math.cos(phi)]]).T


def RZ(phi=0):
    """Returns a 3x3 rotation matrix about the Z axis."""
    return np.array([[math.cos(phi), math.sin(phi), 0], [-math.sin(phi), math.cos(phi), 0], [0, 0, 1]]).T


def H(R=np.eye(3), T=[0, 0, 0]):
    """Returns a 4x4 transformation matrix with rotation R and translation T."""
    return np.append(np.append(R, np.atleast_2d(T).T, 1), [[0, 0, 0, 1]], 0)


def origin(h):
    """Returns the origin of a transformation matrix."""
    return h[0:3, 3]


def get_frame(tr, n):
    """Returns the transformation of frame n to the base frame."""
    h = H()
    for i in range(n+1):
        h = h @ tr[i]
    return h


def get_ee(tr):
    """Returns the transformation of the end effector frame to the base frame."""
    return get_frame(tr, len(tr)-1)


def normalize_angles(q):
    newq = np.copy(q)
    for i in range(len(newq)):
        newq[i] = ((newq[i] % (2*math.pi)) + 2*math.pi) % (2*math.pi)
        if newq[i] > math.pi:
            newq[i] -= 2*math.pi
    return newq


def plot_robot(ax, tr, l=0.2):
    """Plots all frames in tr."""
    cur = H()
    for artist in ax.lines + ax.collections + ax.texts:
        artist.remove()
    plot_frame(ax, cur, 'b', l=l)
    for i, h in enumerate(tr):
      cur = cur @ h
      if i == len(tr)-1:
          s = 'e'
      else:
          s = str(i)
      plot_frame(ax, cur, s, l=l)


def plot_robot_simple(ax, tr, l=0.2):
    """Plots robot in tr."""
    cur = H()
    for artist in ax.lines + ax.collections + ax.texts:
        artist.remove()
    plot_frame(ax, cur, 'b', l=l)
    p = origin(cur)
    for i, h in enumerate(tr):
      cur = cur @ h
      newp = origin(cur)
      if i == len(tr)-1:
        plot_frame(ax, cur, 'e', l=l)
      else:
        ax.plot([p[0], newp[0]], [p[1], newp[1]], [p[2], newp[2]])
      p = newp


def plot_frame(ax, h, name=None, l=1):
    """Plots the frame defined by a transformation matrix h."""
    c = np.eye(3)
    c = np.append(c, np.repeat(c, 2, 0), 0)
    t = h[0:3, 3].flatten()

    ax.quiver(np.repeat(t[0], 3), np.repeat(t[1], 3), np.repeat(t[2], 3),
              l*h[0, 0:3], l*h[1, 0:3], l*h[2, 0:3], colors=c)
    if name is not None:
        ax.text(t[0], t[1], t[2], name)
