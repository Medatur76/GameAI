import pygetwindow as gw
import pyautogui
import time
import math
import cv2
import numpy as np
import threading
import os

#time.sleep(10)

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
                distances.append(math.sqrt((x-origin[0])**2+(y-origin[1])**2))
                break
            if mask[y, x] != 0:
                distances.append(math.sqrt((x-origin[0])**2+(y-origin[1])**2))
                break
            image[y, x] = (0, 0, 255)
    if f == -1: return distances
    else: cv2.imwrite('frame_' + str(f) + '.png', image)

def video(w: str, t=10):
    window: gw.Win32Window = gw.getWindowsWithTitle(w)[0]

    # Get the window's position and size
    width, height = window.width, window.height

    n_frame = 0

    output_file = 'output_video.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    start_time = time.time()
    end_time = start_time + 1  # Run for 10 seconds

    threads: list[threading.Thread] = []

    # Capture frames
    while time.time() < end_time:
        print("Time: ", str(end_time-time.time()), ", Frame: ", str(n_frame))
        t = threading.Thread(target=takePhoto,args=[w, n_frame], daemon=True)
        threads.append(t)
        t.start()
        n_frame += 1
        time.sleep(0.01)

    for t in threads:
        t.join()

    video_writer = cv2.VideoWriter(output_file, fourcc, 100, (width, height))

    for f in range(n_frame):
        video_writer.write(cv2.imread('frame_'+str(f)+'.png'))
        #os.remove('frame_' + str(f) + '.png')

    video_writer.release()
    print("Video file saved as", output_file)

#video("Trackmania")