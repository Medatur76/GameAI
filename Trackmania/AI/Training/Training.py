from AI.Training.Inputs import getInputs
from AI.ActivationClasses.Activation import Activation
from AI.NeuralClasses.NeuralNetwork import NeuralNetwork, preset
import time, pydirectinput

def seqPressKeys(keys: list[str]):
        for key in keys:
            pydirectinput.press(key)

class Training():
    nextUpKeys: list[str] = []
    isGen = False
    currentGen = -1
    currentAI = 0
    startTime = 0
    speed = 0
    position = [0.0, 0.0, 0.0]
    def getAIG(self):
        return [self.currentGen, self.currentAI]
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
    def genTrain(self, n_layers: int=None, output_activations: list[Activation]=None, base_activation: Activation=Activation, generations: int=1, ais: int = 1, nnpreset: preset = None):
        """Trains the AI by running a number of random neural networks (the ais input times 100) and calculates each of their accumulative reward. It collects the top 5% neural networks, multiplies equally to match the ais number times 100, then slightly modifies each one. This is repeated a number of times based on the generations input. After the last generation is process, the best ai is returned."""
        self.isGen = True
        print("Waiting")
        while not getInputs()[1][2]: time.sleep(0.1)
        print("Started!")
        if nnpreset == None: bestNetworks = [NeuralNetwork(16, n_layers, 4, output_activations, base_activation) for _ in range(ais*100)]
        else: bestNetworks = [NeuralNetwork.fromPreset(nnpreset) for _ in range(ais*100)]
        pydirectinput.press('del')
        for g in range(generations):
            self.currentGen = g+1
            scores: list[float] = []
            for n in range(len(bestNetworks)):
                self.currentAI = n+1
                ai = bestNetworks[n]
                possibleEnd = False
                endTicks = 0
                ai.train((1400/generations)/((g+1)*1000))
                runCompleted = False
                score: float = 0.0
                lastSpeed = 0
                lastZ = getInputs()[1][0][2]
                self.startTime = time.time()
                end_time = self.startTime + 20
                while not runCompleted and time.time() < end_time and not possibleEnd:
                    inputs, gameData = getInputs()
                    self.speed = inputs[15]
                    self.position: list[float] = gameData[0]
                    if gameData[1]:
                        score += 1000000
                        runCompleted = True
                        continue
                    if self.speed.__round__() == 0:
                        if endTicks <= 10:
                            endTicks += 1
                        else:
                            possibleEnd = True
                            score -= 1000000
                    else:
                        endTicks = 0
                    if self.position[1] < 10:
                        score -= 1000000
                        possibleEnd = True

                    if self.position[0] > 770 or self.position[0] < 643:
                        score += ((self.position[2] - lastZ)*2)-abs(self.position[2] - lastZ)
                    else:
                        score += ((lastZ - self.position[2])*2)-abs(self.position[2] - lastZ)
                    lastZ = self.position[2]

                    if self.speed > lastSpeed:
                        score += (self.speed - lastSpeed)
                    output = ai.forward(inputs)
                    keys = []
                    if output[0] == 1: 
                        keys.append('w')
                    if output[1] == 1: 
                        keys.append('s')
                    if output[2] == 1: 
                        keys.append('a')
                    if output[3] == 1: 
                        keys.append('d')

                    if not keys == []: self.pressKeys(keys)
                scores.append(score)
                pydirectinput.press(['r', 'up', 'up', 'down'])
                time.sleep(0.1)
                pydirectinput.typewrite(f"G{str(g)}A{str(n)}")
                pydirectinput.press(['down', 'enter', 'enter', 'del'])
            sortedAis = [ai for _, ai in sorted(zip(scores, bestNetworks), key=lambda x: x[0], reverse=True)]
            bestNetworks.clear()
            returnedAis = ((len(sortedAis)*5)/100).__round__()
            multiplesOfAi = (100/returnedAis).__round__()
            bestNetworks5 = sortedAis[:returnedAis].copy()
            for n in bestNetworks5:
                for _ in range(multiplesOfAi):
                    bestNetworks.append(n)
        return bestNetworks[0]

    def train(self, n_layers: int=None, output_activations: list[Activation]=None, base_activation: Activation=Activation, runs: int=100, preset: preset = None):
        """Trains the AI by running a single neural network multiple times (the runs input). Deviates slightly if the reward is greater than the last reward. Will try back propagation to make good adjustments it the reward is good."""
        print("Waiting")
        while not getInputs()[1][2]: time.sleep(0.1)
        print("Started!")
        if preset == None:
            nn = NeuralNetwork(16, n_layers, 4, output_activations, base_activation)
        else:
            nn = NeuralNetwork.fromPreset(preset)
        lastScore = 0
        pydirectinput.press('del')
        if runs == None or runs == -1:
            running = True
            self.currentAI = 0
            while running:
                self.currentAI += 1
                possibleEnd = False
                endTicks = 0
                nn.train(0.25)
                runCompleted = False
                score: float = 0.0
                lastSpeed = 0
                lastZ = getInputs()[1][0][2]
                self.startTime = time.time()
                end_time = self.startTime + 20
                while not runCompleted and time.time() < end_time and not possibleEnd:
                    inputs, gameData = getInputs()
                    self.speed = inputs[15]
                    self.position: list[float] = gameData[0]
                    running = gameData[2]
                    if gameData[1]:
                        score += 1000
                        runCompleted = True
                        continue
                    if self.speed.__round__() == 0:
                        if endTicks <= 10:
                            endTicks += 1
                        else:
                            possibleEnd = True
                            score -= 10000
                    else:
                        endTicks = 0
                    if self.position[1] < 10:
                        score -= 1000000
                        possibleEnd = True

                    if self.position[0] > 770 or self.position[0] < 643:
                        score += ((self.position[2] - lastZ)*2)-abs(self.position[2] - lastZ)
                    else:
                        score += ((lastZ - self.position[2])*2)-abs(self.position[2] - lastZ)
                    lastZ = self.position[2]

                    if self.speed > lastSpeed:
                        score += (self.speed - lastSpeed)
                    output = nn.forward(inputs)
                    keys = []
                    if output[0] == 1: 
                        keys.append('w')
                    if output[1] == 1: 
                        keys.append('s')
                    if output[2] == 1: 
                        keys.append('a')
                        if self.position[2] <= 673.595 and self.position[0] < 770:
                            score += 10
                    if output[3] == 1: 
                        keys.append('d')
                        if self.position[2] >= 766.825:
                            score += 10
                    if self.position[2] <= 673.595 and self.position[0] < 770 and output[2] != 1:
                            score -= 10
                    if self.position[2] >= 766.825 and output[2] != 1:
                            score -= 10

                    if not keys == []: self.pressKeys(keys)
                pydirectinput.press(['r', 'up', 'up', 'down'])
                time.sleep(0.1)
                pydirectinput.typewrite(str(r))
                pydirectinput.press(['down', 'enter', 'enter', 'del'])
                if (score > lastScore): lastScore = score
                else: nn.revert()
        else:
            for r in range(runs):
                self.currentAI = r+1
                possibleEnd = False
                endTicks = 0
                nn.train(0.25)
                runCompleted = False
                score: float = 0.0
                lastSpeed = 0
                lastZ = getInputs()[1][0][2]
                self.startTime = time.time()
                end_time = self.startTime + 20
                while not runCompleted and time.time() < end_time and not possibleEnd:
                    inputs, gameData = getInputs()
                    self.speed = inputs[15]
                    self.position: list[float] = gameData[0]
                    if gameData[1]:
                        score += 1000
                        runCompleted = True
                        continue
                    if inputs[15].__round__() == 0:
                        if endTicks <= 10:
                            endTicks += 1
                        else:
                            possibleEnd = True
                            score -= 10000
                    else:
                        endTicks = 0
                    if self.position[1] < 10:
                        score -= 1000000
                        possibleEnd = True

                    if self.position[0] > 770 or self.position[0] < 643:
                        score += ((self.position[2] - lastZ)*2)-abs(self.position[2] - lastZ)
                    else:
                        score += ((lastZ - self.position[2])*2)-abs(self.position[2] - lastZ)
                    lastZ = self.position[2]

                    if inputs[15] > lastSpeed:
                        score += (inputs[15] - lastSpeed)
                    output = nn.forward(inputs)
                    keys = []
                    if output[0] == 1: 
                        keys.append('w')
                    if output[1] == 1: 
                        keys.append('s')
                    if output[2] == 1: 
                        keys.append('a')
                        if self.position[2] <= 673.595:
                            score += 10
                    if output[3] == 1: 
                        keys.append('d')
                        if self.position[2] >= 766.825:
                            score += 10

                    if self.position[2] <= 673.595 and self.position[0] < 770 and output[2] != 1:
                            score -= 10
                    if self.position[2] >= 766.825 and output[2] != 1:
                            score -= 10

                    if not keys == []: self.pressKeys(keys)
                pydirectinput.press(['r', 'up', 'up', 'down'])
                time.sleep(0.1)
                pydirectinput.typewrite(str(r))
                pydirectinput.press(['down', 'enter', 'enter', 'del'])
                if (score > lastScore): lastScore = score
                else: nn.revert()
        return nn
    def trainPPO(self, n_layers: int, output_activations: list[Activation], base_activation: Activation=Activation, runs: int=100, preset: preset = None):
        """PPO Training for the network. its like the ``train()`` function but instead uses back propagation to more quickly improve. More resource intesive."""
        