from Training.Training import *
from ActivationClasses.BinaryStep import *
from ActivationClasses.Sigmoid import *
from Training.Inputs import *
import time, pyautogui
import win32api, win32con

def press_key(key):
    win32api.keybd_event(key, 0, 0, 0)
    time.sleep(0.1)  # Adjust the sleep duration as needed
    win32api.keybd_event(key, 0, win32con.KEYEVENTF_KEYUP, 0)

# Inputs:
# 15 - Distance
# 1 - Speed

bestRacer = Training.genTrain(10, [BinaryStepActivation, BinaryStepActivation, BinaryStepActivation, BinaryStepActivation])
#while getInputs()[1][2] and not getInputs()[1][1]:
#    data, _ = getInputs()

#    output = bestRacer.forward(data)
#    keys = []
#    if output[0] == 1: keys.append('w')
#    if output[1] == 1: keys.append('s')
#    if output[2] == 1: keys.append('a')
#    if output[3] == 1: keys.append('d')

#    if not keys == []: pyautogui.press(keys)