from AI.ActivationClasses.Activations import *
import numpy as np

def activationFromString(string: str):
    if string == "BinaryStep": return BinaryStepActivation()
    elif string == "Sigmoid": return SigmoidActivation()
    elif string == "Hyperbolic": return HyperTangActivation()
    else: return Activation()

class NeuralLayer():
    @classmethod
    def fromData(nl, data):
        if data["activations"] == "None":
            layer = nl(data["n_inputs"], data["n_outputs"], activationFromString(data["activation"]), None, isFromData=True)
            layer.weights = np.array(data["weights"])
            layer.biases = np.array([data["biases"]])
            return layer
        else:
            layer = nl(data["n_inputs"], data["n_outputs"], None, [activationFromString(a) for a in data["activations"]], isFromData=True)
            layer.weights = np.array(data["weights"])
            layer.biases = np.array([data["biases"]])
            return layer
    def __init__(self, n_inputs: int, n_outputs: int, activation: Activation, activations: list[Activation]=None, weights = None, biases = None, isFromData: bool = False):
        if weights is None:
            weights = []
        if biases is None:
            biases = []
        if not isFromData:
            self.multiActivations: bool = False
            self.activation: Activation = activation
            self.activations: list[Activation] | None = activations
            if activations != None:
                if len(activations) == n_outputs:
                    self.multiActivations: bool = True
                else:
                    raise ValueError("Mismatch between requested outputs ("+ str(n_outputs) + ") and requested activations (" + str(len(activations)) + ")")
            self.weights = 0.1 * np.random.randn(n_inputs, n_outputs)
            self.biases = np.random.randn(1, n_outputs)
            self.nInputs: int = n_inputs
            self.nOutputs: int = n_outputs
        else:
            self.nInputs = n_inputs
            self.nOutputs = n_outputs
            self.weights = weights
            self.biases = biases
            if not activations == None:
                self.multiActivations = True
                self.activations = activations
            else:
                self.multiActivations = False
                self.activation = activation
    def forward(self, inputs, inputLayer: bool = False):
        self.input = np.array(inputs)
        if not inputLayer: ainputs = inputs[0]
        else: ainputs = inputs
        if not self.multiActivations: poutput = self.activation.forward(np.dot(ainputs, self.weights) + self.biases)
        else:
            outputs = []
            for o in range(len(self.biases[0])):
                output = self.biases[0][o]
                for w in range(self.nInputs):
                    output += self.weights[w][o]*ainputs[0][w]
                outputs.append(self.activations[o].forward(output))
            poutput = outputs

        self.output = [poutput, np.dot(ainputs, self.weights) + self.biases]
    def backward(self, error, learning_rate: float):
        delta = None
        if self.multiActivations:
            delta = error[0] * self.activations[0].derivative(self.output[0])
            for i in range(len(self.output)-1):
                delta = np.concatenate([delta, error[i+1] * self.activations[i+1].derivative(self.output[i+1])], axis=1)
        else:
            delta = error * np.array(self.activation.derivative(self.output))
        self.weights += self.input.T.dot(delta) * learning_rate
        self.biases += np.sum(delta, axis=0, keepdims=True) * learning_rate
        return delta
    def train(self, m: float = 0.05):
        self.pWeights = self.weights.copy()
        self.pBiases = self.biases.copy()
        self.weights += m * np.random.randn(self.nInputs, self.nOutputs)
        self.biases += m * np.random.randn(1, self.nOutputs)
    def revert(self):
        self.weights = self.pWeights.copy()
        self.biases = self.pBiases.copy()
    def distributionPropagation(self, delta: np.ndarray = None, learning_rate: float = 1, inputLayer: bool = False, nextLayer = None) -> np.ndarray:
        delta *= self.activation.derivative(self.output[1])
        updateWeight = np.dot(self.input[0].T if not inputLayer else np.array([self.input]).T, delta)
        updateBiases = delta
        self.weights -= learning_rate * updateWeight
        self.biases -= learning_rate * updateBiases
        return np.dot(delta, self.weights.T)