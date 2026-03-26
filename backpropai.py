import numpy as np
from alive_progress import alive_bar
import math

class Activation:
    @staticmethod
    def forward(x):
        return x
    @staticmethod
    def derivative(x):
        return 1
class Softplus(Activation):
    @staticmethod
    def forward(x):
        return np.log(np.add(1, np.pow(math.e, x)))
    @staticmethod
    def derivative(x):
        return np.divide(1, np.add(1, np.pow(math.e, np.multiply(-1, x))))

class NeuralLayer:
    def __init__(self, n_inputs: int, n_outputs: int, activation: Activation):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.weights = np.multiply(np.random.randn(n_inputs, n_outputs), np.sqrt(2.0 / n_inputs))
        self.biases = np.zeros((1, n_outputs))
        self.activation = activation
    def forward(self, inputs) -> None:
        self.inputs = np.array(inputs)
        self.output = self.activation.forward(np.dot(self.inputs, self.weights) + self.biases)
    def backward(self, error, lError = True):
        if lError: layerError = error.dot(self.weights.T)
        else: layerError = error
        delta = np.multiply(layerError, self.activation.derivative(self.output))
        print(f"{delta.shape=}\n{self.inputs.shape=}")
        self.weights += self.inputs.T.dot(delta)
        return np.array(delta)
class NeuralNetwork:
    def __init__(self, n_layers: int, n_inputs: int, n_outputs: int, activation: Activation = Activation):
        self.layers: list[NeuralLayer] = [];
        for i in range(n_layers): self.layers.append(NeuralLayer(n_inputs+i*2, n_inputs+(i+1)*2, activation))
        self.layers.append(NeuralLayer(n_inputs+(n_layers)*2, n_outputs, activation))
    def forward(self, inputs):
        self.layers[0].forward(inputs)
        for i in range(len(self.layers)-1): self.layers[i+1].forward(self.layers[i].output)
        return self.layers[len(self.layers)-1].output
    def backward(self, error):
        lastError = self.layers[len(self.layers)-1].backward(error, False)
        for layer in reversed(self.layers[:-1]): lastError = layer.backward(lastError)

if __name__ == '__main__':
    nn = NeuralNetwork(4, 2, 1)
    iterations = 200

    with alive_bar(iterations, title="Training!") as bar:
        while iterations != 0:
            x = np.random.randint(51, size=(2))
            nn.backward(pow(nn.forward(x)[0][0] - sum(x), 2))
            iterations -= 1
