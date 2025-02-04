# client example

import win32api, win32con

def press_key(key):
    win32api.keybd_event(key, 0, 0, 0)
    win32api.keybd_event(key, 0, win32con.KEYEVENTF_KEYUP, 0)

press_key(0x57)
press_key(0x53)
press_key(0x44)
press_key(0x41)
press_key(0x52)
press_key(win32con.VK_UP)
press_key(0x0D)
press_key(0x2E)