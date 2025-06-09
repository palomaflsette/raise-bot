""" 
Extraindo intrinsecos e extrinsecos da camera
"""

import depthai as dai
import numpy as np


def get_intrinsics(device, resolution=(1280, 720)):
    calib = device.readCalibration()
    K_left = np.array(calib.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_B, *resolution))
    K_rgb = np.array(calib.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_A, *resolution))
    return K_left, K_rgb


def get_extrinsics(device):
    calib = device.readCalibration()
    T_left_to_rgb = np.array(calib.getCameraExtrinsics(
        dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_A))
    return T_left_to_rgb
