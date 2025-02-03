from Inputs import getInputs
from ActivationClasses.Activation import Activation
from NeuralClasses.NeuralNetwork import NeuralNetwork
import pyautogui

class Training():
    def genTrain(n_layers: int, n_output_activations: list[Activation], base_activation: Activation=Activation, generations: int=1, ais: int = 100):
        """Trains the AI by running a number of random neural networks (the ais input) and calculates each of their accumulative reward. It collects the top 5% neural networks, multiplies equally to match the ais number, then slightly modifies each one. This is repeated a number of times based on the generations input. After the last generation is process, the best ai is returned."""
        n_outputs = 4
        if not n_output_activations == None: n_outputs = len(n_output_activations)
        best_networks = [NeuralNetwork(16, n_layers, n_outputs, n_output_activations, base_activation) for _ in range(ais)]
        for _ in range(generations):
            for ai in best_networks:
                runCompleted = False
                while not runCompleted:
                    output = ai.forward(getInputs())
                    if not n_output_activations == None:
                        for i in range(n_output_activations):
                            output[i] = n_output_activations[i].forward(output[i])
                    keys = []
                    if output[0] == 1: keys.append('w')
                    if output[1] == 1: keys.append('s')
                    if output[2] == 1: keys.append('a')
                    if output[3] == 1: keys.append('d')

                    if not keys == []: pyautogui.press(keys)
        pass
    def train(self, n_inputs: int, n_layers: int, n_outputs: int, n_output_activations: list[Activation], base_activation: Activation=Activation, runs: int=100):
        """Trains the AI by running a single neural network multiple times (the runs input). Deviates slightly if the reward is greater than 0. Will try back propagation to make good adjustments it the reward is good."""
        pass
