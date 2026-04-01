from AI.Training.Training import *
from AI.Training.Inputs import *
from Server.Server import run
import pydirectinput, threading, time
import TUI

class Main:
    nextUpKeys: list[str] = []
    train = Training()

    def pressKeys(self, keys: list[str]) -> None:
        for key in keys:
            try:
                self.nextUpKeys.remove(key)
                keys.remove(key)
            except:
                pass
        for key in self.nextUpKeys:
            pydirectinput.keyUp(key)
        self.nextUpKeys.clear()
        for key in keys:
            pydirectinput.keyDown(key)
        self.nextUpKeys = keys.copy()

    def run(self) -> None:
        time.sleep(0.1)
        # Inputs:
        # 15 - Distance
        # 1 - Speed
        fileExisis: bool
        try:
            open("Racer.json", "r")
            fileExisis = True
        except:
            fileExisis = False
        if fileExisis and input("Would you like to use the \'Racer.json\' neural network? [y/N] ") == 'y':
                racer = NeuralNetwork.fromFile("Racer.json")
        else:
            method = input("Training method [trial/gen/ppo]: ")
            if method == "trial":
                racer = self.train.train()
            elif method == "gen":
                racer = self.train.genTrain()
            elif method == "ppo":
                racer = self.train.trainPPO()
        racer.save()
        print("Saved")
        while not getInputs()[1][2]: time.sleep(0.1)
        while getInputs()[1][2] and not getInputs()[1][1]:
            data, _ = getInputs()

            output = racer.forward(data)
            keys = []
            if output[0] == 1: keys.append('w')
            if output[1] == 1: keys.append('s')
            if output[2] == 1: keys.append('a')
            if output[3] == 1: keys.append('d')

            if not keys == []: self.pressKeys(keys)

main = Main()

trainingThread = threading.Thread(target=main.run)

trainingThread.start()

run(main.train)