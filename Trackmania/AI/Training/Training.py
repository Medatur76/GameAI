from Training.Inputs import getInputs
from ActivationClasses.Activation import Activation
from NeuralClasses.NeuralNetwork import NeuralNetwork, preset
import time, pydirectinput

def seqPressKeys(keys: list[str]):
        for key in keys:
            pydirectinput.press(key)

class Training():
    nextUpKeys: list[str] = []
    currentGen = 1
    currentAI = 1
    def getAIG(self):
        return [self.currentGen, self.currentAI]
    def pressKeys(self, keys: list[str]):
        for key in self.nextUpKeys:
            pydirectinput.keyUp(key)
        self.nextUpKeys.clear()
        for key in keys:
            pydirectinput.keyDown(key)
        self.nextUpKeys = keys.copy()
    def genTrain(self, n_layers: int=None, n_output_activations: list[Activation]=None, base_activation: Activation=Activation, generations: int=1, ais: int = 1, preset: preset = None):
        """Trains the AI by running a number of random neural networks (the ais input times 100) and calculates each of their accumulative reward. It collects the top 5% neural networks, multiplies equally to match the ais number times 100, then slightly modifies each one. This is repeated a number of times based on the generations input. After the last generation is process, the best ai is returned."""
        print("Waiting")
        while not getInputs()[1][2]: time.sleep(0.1)
        print("Started!")
        # USING BEGINING CAR ROTATION
        # Z runs foward and back
        # X runs side to side
        # We want Z to go up
        # As soon as the car passes X 770 (The x is below 770) we want Z to go down
        # As soon as the car passes X 643 (The X is below 643) we want Z to go up
        if preset == None: bestNetworks = [NeuralNetwork(16, n_layers, 4, n_output_activations, base_activation) for _ in range(ais*100)]
        else: bestNetworks = [NeuralNetwork.fromPreset(preset) for _ in range(ais*100)]
        pydirectinput.press('del')
        for g in range(generations):
            self.currentGen = g+1
            scores: list[float] = []
            for n in range(len(bestNetworks)):
                self.currentAI = n+1
                ai = bestNetworks[n]
                end_time = time.time() + 23
                possibleEnd = False
                endTicks = 0
                ai.train(0.5)
                runCompleted = False
                score: float = 0.0
                lastSpeed = 0
                lastZ = getInputs()[1][0][2]
                while not runCompleted and time.time() < end_time and not possibleEnd:
                    inputs, gameData = getInputs()
                    position: list[float] = gameData[0]
                    if gameData[1]:
                        score += 1000000
                        runCompleted = True
                        continue
                    if inputs[15].__round__() == 0:
                        if endTicks <= 10:
                            endTicks += 1
                        else:
                            possibleEnd = True
                    else:
                        endTicks = 0
                    if position[1] < 10:
                        score = -1000000
                        possibleEnd = True

                    if position[0] > 770 or position[0] < 643:
                        score += (position[2] - lastZ)*2
                    else:
                        score += (lastZ - position[2])*2
                    lastZ = position[2]

                    if inputs[15] > lastSpeed:
                        score += (inputs[15] - lastSpeed)
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
                scores.append(score*-1)
                pydirectinput.press(['r', 'up', 'up', 'down'])
                time.sleep(0.1)
                pydirectinput.typewrite("G" + str(g) + "A" + str(n))
                pydirectinput.press(['down', 'enter', 'enter', 'del'])
            sortedAis = [ai for _, ai in sorted(zip(scores, bestNetworks), key=lambda x: x[0])]
            bestNetworks.clear()
            returnedAis: int = ((len(sortedAis)*5)/100).__round__()
            multiplesOfAi: int = (100/returnedAis).__round__()
            bestNetworks5 = sortedAis[:returnedAis].copy()
            for n in bestNetworks5:
                for _ in range(multiplesOfAi):
                    bestNetworks.append(n)
            input("Please enter anything to confirm that the next generation can go!")
            time.sleep(5)
        return bestNetworks[0]

    def train(self, n_layers: int, output_activations: list[Activation], base_activation: Activation=Activation, runs: int=100):
        """Trains the AI by running a single neural network multiple times (the runs input). Deviates slightly if the reward is greater than the last reward. Will try back propagation to make good adjustments it the reward is good."""
        while not getInputs()[1][2]: time.sleep(0.1)
        nn = NeuralNetwork(16, n_layers, 4, output_activations, base_activation)
        lastScore = 0
        for r in range(runs):
            press_key('del', win_id, xdo)
            lastSpeed = 0
            end_time = time.time() + 15 + r*5
            nn.train(0.05)
            runCompleted = False
            score: float = 0.0
            while not runCompleted and time.time() < end_time:
                inputs, gameData = getInputs()
                if gameData[1]:
                    score += 100
                    runCompleted = True
                    continue
                if (inputs[15] > lastSpeed):
                    score += (inputs[15]-lastSpeed)/10
                    lastSpeed = inputs[15]
                output = nn.forward(inputs)
                keys = []
                if output[0] == 1: 
                    keys.append('w')
                if output[1] == 1: 
                    keys.append('s')
                if output[2] == 1: 
                    keys.append('a')
                if output[3] == 1: 
                    keys.append('d')

                if not keys == []: press_key(keys, win_id, xdo)

                score += gameData[0]
            _, finalData = getInputs()
            score += finalData[0]
            if (score > lastScore): lastScore = score
            else: nn.revert()
        return nn