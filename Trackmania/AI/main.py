from Training.Training import *
from ActivationClasses.BinaryStep import *
from ActivationClasses.Sigmoid import *
from Training.Inputs import *
from xdo import Xdo

def press_key(keys: list[str], id: int, xdo: Xdo):
    if (keys is str): keys = [keys]
    xdo.send_keysequence_window_down(id, keys)
    xdo.send_keysequence_window_up(id, keys)

# Inputs:
# 15 - Distance
# 1 - Speed

bestRacer = Training.genTrain(10, [BinaryStepActivation, BinaryStepActivation, BinaryStepActivation, BinaryStepActivation])
xdo = Xdo()
win_id = xdo.get_active_window()
while getInputs()[1][2] and not getInputs()[1][1]:
    data, _ = getInputs()

    output = bestRacer.forward(data)
    keys = []
    if output[0] == 1: keys.append('w')
    if output[1] == 1: keys.append('s')
    if output[2] == 1: keys.append('a')
    if output[3] == 1: keys.append('d')

    if not keys == []: press_key(keys, win_id, xdo)