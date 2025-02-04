from NeuralClasses.NeuralLayer import NeuralLayer
from ActivationClasses.Activation import Activation

class NeuralNetwork(): 
    def __init__(self, n_inputs: int, n_layers: int, n_outputs: int, output_activations: list[Activation], base_activation: Activation=Activation):
        self.layers: list[NeuralLayer] = [];
        for i in range(n_layers): self.layers.append(NeuralLayer(n_inputs+i*2, n_inputs+(i+1)*2, base_activation))
        self.layers.append(NeuralLayer(n_inputs+(n_layers)*2, n_outputs, None, output_activations))
    def forward(self, inputs, use_final_activation=True):
        self.layers[0].forward(inputs)
        for i in range(len(self.layers)-2): self.layers[i+1].forward(self.layers[i].output)
        if use_final_activation: self.layers[len(self.layers)-1].forward(self.layers[len(self.layers)-2].output)
        else: self.layers[len(self.layers)-1].forward(self.layers[len(self.layers)-2].output, use_activation=False)
        return self.layers[len(self.layers)-1].output
    def train(self, m: float = 0.05):
        for layer in self.layers: layer.train(m)
    def revert(self):
        for layer in self.layers: layer.revert()