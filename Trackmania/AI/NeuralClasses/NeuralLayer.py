from ActivationClasses.Activation import Activation
import numpy as np

class NeuralLayer():
    def __init__(self, n_inputs: int, n_outputs: int, activation: Activation, activations: list[Activation]=None, weights: list[list[float]] = None, biases: list[list[float]] = None):
        self.multiActivations = False
        self.activation = activation
        self.activations = activations
        if activations != None:
            if len(activations) == n_outputs:
                self.multiActivations = True
            else:
                raise ValueError("Mismatch between requested outputs ("+ str(n_outputs) + ") and requested activations (" + str(len(activations)) + ")")
        if weights == None: self.weights = 0.1 * np.random.randn(n_inputs, n_outputs)
        else: self.weights = weights
        if biases == None: self.biases = np.random.randn(1, n_outputs)
        else: self.biases = biases
        self.nInputs = n_inputs
        self.nOutputs = n_outputs
    def forward(self, inputs, use_activation=True):
        if use_activation:
            if not self.multiActivations: self.output = self.activation.forward(np.dot(inputs, self.weights) + self.biases)
            else:
                outputs = []
                for o in range(len(self.biases)):
                    output = self.biases[o]
                    for w in range(len(self.weights[o])):
                        output += self.weights[0][w]*inputs[w]
                    outputs.append(self.activations[o].forward(output))
                self.output = outputs

        else: self.output = np.dot(inputs, self.weights) + self.biases
    def train(self, m: float = 0.05):
        self.weights += m * np.random.randn(self.nInputs, self.nOutputs)
        self.output += m * np.random(1, self.nOutputs)
