from Library.Activations import Activation
import numpy as np

class NeuralLayer:
    def __init__(self, n_inputs: int, n_outputs: int, activation: Activation):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.weights = np.multiply(np.random.randn(n_inputs, n_outputs), np.sqrt(2.0 / n_inputs), dtype=np.float128)
        self.biases = np.zeros((1, n_outputs))
        self.activation = activation
    def forward(self, inputs) -> None:
        self.inputs = np.array(inputs, dtype=np.float128)
        if self.inputs.ndim == 1:
            self.inputs = self.inputs.reshape(1, -1)
        self.output = self.activation.forward(np.dot(self.inputs, self.weights) + self.biases)
    def train(self, m: float = 0.05):
        self.pWeights = self.weights.copy()
        self.pBiases = self.biases.copy()
        self.weights += m * np.random.randn(self.n_inputs, self.n_outputs)
        self.biases += m * np.random.randn(1, self.n_outputs)
    def revert(self):
        self.weights = self.pWeights
        self.biases = self.pBiases
    def backward(self, error, learning_rate):
        error = np.array(error, dtype=np.float128)
        if error.ndim == 0:
            error = np.full(self.output.shape, error, dtype=np.float128)
        else:
            try:
                error = np.broadcast_to(error, self.output.shape).astype(np.float128, copy=False)
            except ValueError as exc:
                raise ValueError(f"Error shape {error.shape} cannot be broadcast to output shape {self.output.shape}") from exc
        delta = np.multiply(error, self.activation.derivative(self.output), dtype=np.float128)
        self.weights -= np.multiply(learning_rate, self.inputs.T.dot(delta), dtype=np.float128)
        self.biases -= np.multiply(learning_rate, np.sum(delta, axis=0, keepdims=True), dtype=np.float128)
        return delta.dot(self.weights.T)
class NeuralNetwork:
    def __init__(self, n_layers: int, n_inputs: int, n_outputs: int, activation: Activation = Activation):
        self.layers: list[NeuralLayer] = [];
        for i in range(n_layers): self.layers.append(NeuralLayer(n_inputs+i*2, n_inputs+(i+1)*2, activation))
        self.layers.append(NeuralLayer(n_inputs+(n_layers)*2, n_outputs, activation))
    def forward(self, inputs):
        self.layers[0].forward(inputs)
        for i in range(len(self.layers)-1): self.layers[i+1].forward(self.layers[i].output)
        return self.layers[len(self.layers)-1].output
    def train(self, m: float = 0.05) -> None:
        for layer in self.layers: layer.train(m)
    def revert(self):
        for layer in self.layers: layer.revert()
    def backward(self, error, learning_rate: float = 1e-7):
        for layer in reversed(self.layers): error = layer.backward(error, learning_rate)