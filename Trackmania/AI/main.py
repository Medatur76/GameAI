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

bestRacer: NeuralNetwork
fileExisis: bool
try:
    open("Racer.nn", "r+t")
    fileExisis = True
except:
    fileExisis = False
if fileExisis:
    choose = input("Please enter 1 if you wish to use the already saved NeuralNetwork, or enter 2 to train a new one: ")
    if choose == 1:
        bestRacer = NeuralNetwork.fromFile(json.load(open("Racer.nn", 'r')))
    elif choose == 2:
        bestRacer = Training().genTrain(10, [BinaryStepActivation, BinaryStepActivation, BinaryStepActivation, BinaryStepActivation], generations=3)
else:
    bestRacer = Training().genTrain(10, [BinaryStepActivation, BinaryStepActivation, BinaryStepActivation, BinaryStepActivation], generations=3)
while getInputs()[1][2] and not getInputs()[1][1]:
    data, _ = getInputs()

    output = bestRacer.forward(data)
    keys = []
    if output[0] == 1: keys.append('w')
    if output[1] == 1: keys.append('s')
    if output[2] == 1: keys.append('a')
    if output[3] == 1: keys.append('d')

    if not keys == []: pressKeys(keys)

bestRacer.save()