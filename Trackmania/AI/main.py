from Training.Training import *
from ActivationClasses.BinaryStep import *
from ActivationClasses.Sigmoid import *
from Training.Inputs import *
import pydirectinput


nextUpKeys: list[str] = []

def pressKeys(keys: list[str]):
    for key in nextUpKeys:
        pydirectinput.keyUp(key)
    nextUpKeys.clear()
    for key in keys:
        pydirectinput.keyDown(key)
    nextUpKeys = keys.copy()

# Inputs:
# 15 - Distance
# 1 - Speed

training = Training()

bestRacer = training.genTrain(10, [BinaryStepActivation, BinaryStepActivation, BinaryStepActivation, BinaryStepActivation], generations=3)
while getInputs()[1][2] and not getInputs()[1][1]:
    data, _ = getInputs()

    output = bestRacer.forward(data)
    keys = []
    if output[0] == 1: keys.append('w')
    if output[1] == 1: keys.append('s')
    if output[2] == 1: keys.append('a')
    if output[3] == 1: keys.append('d')

    if not keys == []: pressKeys(keys)