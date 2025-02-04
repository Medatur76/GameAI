from Training.Inputs import getInputs
from ActivationClasses.Activation import Activation
from NeuralClasses.NeuralNetwork import NeuralNetwork
import time
from xdo import Xdo

def press_key(keys: list[str], id: int, xdo: Xdo):
    if (keys is str): keys = [keys]
    xdo.send_keysequence_window_down(id, keys)
    xdo.send_keysequence_window_up(id, keys)

class Training():
    def genTrain(n_layers: int, n_output_activations: list[Activation], base_activation: Activation=Activation, generations: int=1, ais: int = 1):
        """Trains the AI by running a number of random neural networks (the ais input times 100) and calculates each of their accumulative reward. It collects the top 5% neural networks, multiplies equally to match the ais number times 100, then slightly modifies each one. This is repeated a number of times based on the generations input. After the last generation is process, the best ai is returned."""
        while not getInputs()[1][2]: time.sleep(0.1)
        xdo = Xdo()
        win_id = xdo.get_focused_window()
        bestNetworks = [NeuralNetwork(16, n_layers, 4, n_output_activations, base_activation) for _ in range(ais*100)]
        press_key('del', win_id, xdo)
        for g in range(generations):
            scores: list[float] = []
            for ai in bestNetworks:
                end_time = time.time() + 30 + g*5
                ai.train(0.05/(g+1))
                runCompleted = False
                score: float = 0.0
                while not runCompleted and time.time() < end_time:
                    inputs, gameData = getInputs()
                    if gameData[1]:
                        score += 100
                        runCompleted = True
                        continue
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

                    if not keys == []: press_key(keys, win_id, xdo)

                    score += gameData[0]
                _, finalData = getInputs()
                scores.append(score + finalData[0])
                press_key('del', win_id, xdo)
            press_key('r', win_id, xdo)
            press_key('up', win_id, xdo)
            press_key('enter', win_id, xdo)
            press_key('enter', win_id, xdo)
            sortedAis = [ai for _, ai in sorted(zip(scores, bestNetworks))]
            bestNetworks.clear()
            returnedAis: int = ((len(sortedAis)*5)/100)
            multiplesOfAi: int = 100/returnedAis
            bestNetworks5 = sortedAis[:returnedAis].copy()
            for n in bestNetworks5:
                for _ in range(multiplesOfAi):
                    bestNetworks.append(n)
        return bestNetworks[0]

    def train(n_layers: int, output_activations: list[Activation], base_activation: Activation=Activation, runs: int=100):
        """Trains the AI by running a single neural network multiple times (the runs input). Deviates slightly if the reward is greater than the last reward. Will try back propagation to make good adjustments it the reward is good."""
        while not getInputs()[1][2]: time.sleep(0.1)
        xdo = Xdo()
        win_id = xdo.get_active_window()
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