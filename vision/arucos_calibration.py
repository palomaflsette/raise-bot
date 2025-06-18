import cv2
import depthai as dai
import numpy as np
import threading


class ArucoTracker:
    def __init__(self, marker_size=0.065):
        self.T_cam_to_robo = None
        self.marker_size = marker_size
        self.running = False
        self.lock = threading.Lock()

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters_create()

        self.pipeline = dai.Pipeline()
        cam_rgb = self.pipeline.createColorCamera()
        cam_rgb.setPreviewSize(640, 480)
        cam_rgb.setInterleaved(False)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        xout_rgb = self.pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        self.device = dai.Device(self.pipeline)
        self.q_rgb = self.device.getOutputQueue(
            name="rgb", maxSize=4, blocking=False)

        calib_data = self.device.readCalibration()
        self.camera_matrix = np.array(calib_data.getCameraIntrinsics(
            dai.CameraBoardSocket.RGB, 640, 480))
        self.dist_coeffs = np.zeros((5,))

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _update_loop(self):
        while self.running:
            in_rgb = self.q_rgb.get()
            frame = in_rgb.getCvFrame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.parameters)

            if ids is not None and 0 in ids:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id == 0:
                        rvec = rvecs[i]
                        tvec = tvecs[i]

                        R, _ = cv2.Rodrigues(rvec)
                        T_aruco = np.eye(4)
                        T_aruco[:3, :3] = R
                        T_aruco[:3, 3] = tvec.flatten()

                        T_cam_to_robo = np.linalg.inv(T_aruco)

                        with self.lock:
                            self.T_cam_to_robo = T_cam_to_robo

    def get_T_cam_to_robo(self):
        with self.lock:
            return self.T_cam_to_robo.copy() if self.T_cam_to_robo is not None else None
