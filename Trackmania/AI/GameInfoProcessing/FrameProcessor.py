import pywinctl as gw
import pyautogui
import math
import cv2
import numpy as np

def takePhoto(w: str, f: int=-1):
    window: gw.Win32Window = gw.getWindowsWithTitle(w)[0]
    x, y, width, height = window.left, window.top, window.width, window.height
    image = cv2.cvtColor(np.array(pyautogui.screenshot(region=(x, y, width, height))), cv2.COLOR_BGR2RGB)  # specify the region if needed
    mask = cv2.inRange(image, np.array([0, 0, 0]), np.array([40, 39, 52]))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    origin = (width // 2, ((7/12)*height).__round__())
    num_lines = 15
    angle_increment = 180 / num_lines
    distances: list[float] = []
    for i in range(num_lines):
        angle = np.deg2rad(((i+1) * angle_increment)-6)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        for length in range(1, max(image.shape)):
            x = int(origin[0] + length * cos_angle)
            y = int(origin[1] - length * sin_angle)
            if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
                distances.append(math.sqrt((((x-origin[0])/width)**2+((y-origin[1])/height)**2)/2))
                break
            if mask[y, x] != 0:
                distances.append(math.sqrt((((x-origin[0])/width)**2+((y-origin[1])/height)**2)/2))
                break
            image[y, x] = (0, 0, 255)
    if f == -1: return distances
    else: cv2.imwrite('frame_' + str(f) + '.png', image)