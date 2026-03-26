import numpy as np
from alive_progress import alive_bar
import math

def activation_indenity(x):
    return x
def activation_indenity_derv(x):
    return 1
def activation_tanh(x):
    x = np.array(x, dtype=np.float128)
    return np.divide(np.subtract(np.pow(math.e,x),np.pow(math.e, np.multiply(-1,x))),np.add(np.pow(math.e,x),np.pow(math.e, np.multiply(-1,x))))
def activation_tanh_derv(x):
    return np.subtract(1,np.pow(activation_tanh(x),2))
def activation_softplus(x):
    x = np.array(x, dtype=np.float128)
    return np.log(np.add(1, np.pow(math.e, x)))
def activation_softplus_derv(x):
    return np.divide(1, np.add(1, np.pow(math.e,np.multiply(-1, x))))

class NeuralLayer:
    def __init__(self, n_inputs: int, n_outputs: int, activation):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.weights = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_outputs))
        self.activation = activation
    def forward(self, inputs):
        self.output = self.activation(np.dot(np.array(inputs), self.weights) + self.biases)
    def train(self, m: float = 0.05):
        self.pWeights = self.weights.copy()
        self.pBiases = self.biases.copy()
        self.weights += m * np.random.randn(self.n_inputs, self.n_outputs)
        self.biases += m * np.random.randn(1, self.n_outputs)
    def revert(self):
        self.weights = self.pWeights
        self.biases = self.pBiases
class NeuralNetwork:
    def __init__(self, n_layers: int, n_inputs: int, n_outputs: int, activation):
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


if __name__ == '__main__':
    training_inputs = [
        [1, 2],
        [2, 1],
        [3, 4],
        [4, 3],
        [5, 6],
        [10, 10],
        [39, 5]
    ]

    iterations = 750000 * len(training_inputs)

    nn = NeuralNetwork(4, 2, 1, activation_softplus)
    lastScore = math.inf

    with alive_bar(iterations, title="Training!") as bar:
        for i in range(iterations):
            #if not i%500000: training_inputs.append(np.random.randint(1, 400, (2)))
            nn.train()
            #score = np.sum(np.abs(np.subtract(nn.forward(training_inputs), list([sum(i)] for i in training_inputs))))
            x = np.random.randint(51, size=(2))
            score = abs(nn.forward(x)[0][0] - sum(x))
            if (score > lastScore): nn.revert()
            else: lastScore = score
            bar()
    
    for inputs in training_inputs:
        print(f"Input: {inputs} Output: {nn.forward(inputs)[0][0]} Expected: {sum(inputs)}")
    print(f"Final score: {np.sum(np.abs(np.subtract(nn.forward(training_inputs), list([sum(i)] for i in training_inputs))))}")

    while 1:
        a = input("First number: ")
        if (a == "exit"): break
        else: num1 = int(a)
        num2 = int(input("Second number: "))
        print(f"Sum: {nn.forward([num1, num2])[0][0]}")