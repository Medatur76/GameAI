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
    def __init__(self, n_inputs: int, n_outputs: int, activation: Activation, activations: list[Activation]=None, weights = [], biases = [], isFromData: bool = False):
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
    def forward(self, inputs, use_activation=True):
        self.input = np.array(inputs)
        if use_activation:
            if not self.multiActivations: self.output = self.activation.forward(np.dot(inputs, self.weights) + self.biases)
            else:
                outputs = []
                for o in range(len(self.biases[0])):
                    output = self.biases[0][o]
                    for w in range(self.nInputs):
                        output += self.weights[w][o]*inputs[0][w]
                    outputs.append(self.activations[o].forward(output))
                self.output = outputs

        else: self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, error):
        delta = None
        if self.multiActivations:
            delta = error[0] * self.activations[0].derivative(self.output[0])
            for i in range(len(self.output)-1):
                delta = np.concatenate([delta, error[i+1] * self.activations[i+1].derivative(self.output[i+1])], axis=1)
        else:
            delta = error * self.activation.derivative(self.output)
        self.weights += self.input.T.dot(delta)
        self.biases += np.sum(delta, axis=0, keepdims=True)
        return delta
    def train(self, m: float = 0.05):
        self.pWeights = self.weights.copy()
        self.pBiases = self.biases.copy()
        self.weights += m * np.random.randn(self.nInputs, self.nOutputs)
        self.biases += m * np.random.randn(1, self.nOutputs)
    def revert(self):
        self.weights = self.pWeights.copy()
        self.biases = self.pBiases.copy()
