
from Training.Training import *
from ActivationClasses.BinaryStep import *
from ActivationClasses.Sigmoid import *
from Training.Inputs import *
import pydirectinput


class Main:
    nextUpKeys: list[str] = []

    def pressKeys(self, keys: list[str]):
        for key in keys:
            try:
                self.nextUpKeys.remove(key)
            except:
                pass
        for key in self.nextUpKeys:
            pydirectinput.keyUp(key)
        self.nextUpKeys.clear()
        for key in keys:
            pydirectinput.keyDown(key)
        self.nextUpKeys = keys.copy()

    def run(self):
        # Inputs:
        # 15 - Distance
        # 1 - Speed

        bestRacer: NeuralNetwork
        fileExisis: bool
        try:
            open("Racer.nn", "r")
            fileExisis = True
        except:
            fileExisis = False
        if fileExisis:
            choose = input("Please enter 1 if you wish to use the already saved NeuralNetwork, or enter 2 to train a new one: ")
            if choose == "1":
                bestRacer = NeuralNetwork.fromFile("Racer.nn")
            elif choose == "2":
                bestRacer = Training().genTrain(generations=int(input("Generations: ")), preset="Yosh")
        else:
            #bestRacer = Training().genTrain(10, [BinaryStepActivation, BinaryStepActivation, BinaryStepActivation, BinaryStepActivation], generations=3)
            bestRacer = Training().genTrain(generations=int(input("Generations: ")), preset="Yosh")
        bestRacer.save()
        while not getInputs()[1][2]: time.sleep(0.1)
        while getInputs()[1][2] and not getInputs()[1][1]:
            data, _ = getInputs()

            output = bestRacer.forward(data)
            keys = []
            if output[0] == 1: keys.append('w')
            if output[1] == 1: keys.append('s')
            if output[2] == 1: keys.append('a')
            if output[3] == 1: keys.append('d')

            if not keys == []: self.pressKeys(keys)

main = Main()

main.run()