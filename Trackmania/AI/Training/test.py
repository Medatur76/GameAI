# client example

from xdo import Xdo

def press_key(keys: list[str], id: int, xdo: Xdo):
    xdo.send_keysequence_window(id, keys)

xdo = Xdo()
press_key(['w', 'a', 's', 'd'], xdo.get_active_window(), xdo)
