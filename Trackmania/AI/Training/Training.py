from Inputs import getInputs
from ActivationClasses.Activation import Activation
from NeuralClasses.NeuralNetwork import NeuralNetwork
import pyautogui, time

class Training():
    def genTrain(n_layers: int, n_output_activations: list[Activation], base_activation: Activation=Activation, generations: int=1, ais: int = 1):
        """Trains the AI by running a number of random neural networks (the ais input times 100) and calculates each of their accumulative reward. It collects the top 5% neural networks, multiplies equally to match the ais number times 100, then slightly modifies each one. This is repeated a number of times based on the generations input. After the last generation is process, the best ai is returned."""
        while not getInputs()[0][2]:
            time.sleep(0.1)
        nOutputs = 4
        if not n_output_activations == None: nOutputs = len(n_output_activations)
        bestNetworks = [NeuralNetwork(16, n_layers, nOutputs, n_output_activations, base_activation) for _ in range(ais*100)]
        for g in range(generations):
            scores: list[float] = []
            for ai in bestNetworks:
                ai.train(0.05/(g+1))
                runCompleted = False
                possibleEnd = False
                endTicks: int = 0
                score: float = 0.0
                while not runCompleted:
                    inputs, gameData = getInputs()
                    if gameData[1]:
                        score += 100
                        runCompleted = True
                        continue
                    if gameData[0] == 0.0:
                        if not possibleEnd:
                            possibleEnd = True
                        else:
                            endTicks += 1
                        if endTicks > 10:
                            runCompleted = True
                            continue
                    output = ai.forward(inputs)
                    keys = []
                    if output[0] == 1: keys.append('w')
                    if output[1] == 1: keys.append('s')
                    if output[2] == 1: keys.append('a')
                    if output[3] == 1: keys.append('d')

                    if not keys == []: pyautogui.press(keys)

                    score += gameData[0]
                scores.append(score)
            # TODO: Macro to save a replay or put a marker or somethin
            sortedAis = [ai for _, ai in sorted(zip(scores, bestNetworks))]
            bestNetworks.clear()
            returnedAis: int = ((len(sortedAis)*5)/100)
            multiplesOfAi: int = 100/returnedAis
            bestNetworks5 = sortedAis[:returnedAis].copy()
            for n in bestNetworks5:
                for _ in range(multiplesOfAi):
                    bestNetworks.append(n)
        return bestNetworks[0]

    def train(self, n_inputs: int, n_layers: int, n_outputs: int, n_output_activations: list[Activation], base_activation: Activation=Activation, runs: int=100):
        """Trains the AI by running a single neural network multiple times (the runs input). Deviates slightly if the reward is greater than 0. Will try back propagation to make good adjustments it the reward is good."""
        pass
