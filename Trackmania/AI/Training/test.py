import pyautogui, time, pydirectinput

time.sleep(5)

#pydirectinput.keyDown('s')
#pydirectinput.keyDown('d')
#time.sleep(0.1)
#pydirectinput.keyUp('s')
#pydirectinput.keyUp('d')

pydirectinput.press(['w', 'd', 's'])

pydirectinput.press('del')
pydirectinput.press(['r', 'up', 'enter', 'enter'])