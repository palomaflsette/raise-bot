import cv2
import numpy as np


def extract_green_mask(rgb_image):
     """
     Gera uma máscara binária para uma cor verde-chromakey (cor dos bowls).
     """
     hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

     """Intervalo de HSV para VERDE BRILHANTE (Chromakey)
     Matiz (H) do verde fica em torno de 60 na escala do OpenCV (0-179)"""
     lower_green = np.array([35, 100, 100])
     upper_green = np.array([85, 255, 255])

     mask = cv2.inRange(hsv, lower_green, upper_green)

     """filtros morfológicos ajudarão a limpar qualquer pequeno ruído"""
     mask = cv2.medianBlur(mask, 5)
     kernel = np.ones((5, 5), np.uint8)
     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
     
     return mask