from NeuralClasses.NeuralLayer import NeuralLayer
from ActivationClasses.Activation import Activation
from typing_extensions import TypeAlias
from typing import Literal
from ActivationClasses.BinaryStep import BinaryStepActivation
from ActivationClasses.Sigmoid import SigmoidActivation

preset: TypeAlias = Literal["Yosh"]

class NeuralNetwork(): 
    @classmethod
    def fromFile(nn, file: any):
        return nn(file["n_inputs"],file["n_layers"], file["n_outputs"], file["output_activations"], file["base_activation"])
    @classmethod
    def fromPreset(nn, preset: preset):
        if preset == "Yosh":
            return NeuralNetwork(16, 2, 4, [BinaryStepActivation, BinaryStepActivation, BinaryStepActivation, BinaryStepActivation], SigmoidActivation, [NeuralLayer(16, 64, SigmoidActivation), NeuralLayer(64, 16, SigmoidActivation), NeuralLayer(16, 4, BinaryStepActivation, [BinaryStepActivation, BinaryStepActivation, BinaryStepActivation, BinaryStepActivation])])
    def __init__(self, n_inputs: int, n_layers: int, n_outputs: int, output_activations: list[Activation], base_activation: Activation=Activation, layers: list[NeuralLayer]=None):
        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.output_activations = output_activations
        self.base_activation = base_activation
        self.layers: list[NeuralLayer] = [];
        if layers == None:
            for i in range(n_layers): self.layers.append(NeuralLayer(n_inputs+i*2, n_inputs+(i+1)*2, base_activation))
            self.layers.append(NeuralLayer(n_inputs+(n_layers)*2, n_outputs, None, output_activations))
        else:
            self.layers = layers
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
    def save(self):
        file = open("Racer.nn", "w")
        activations = "\"output_activations\": ["
        for a in range(len(self.output_activations)):
            if a == 0:
                activations += self.output_activations[a].toString()
            else:
                activations += ", " + self.output_activations[a].toString()
        activations += "]"
        file.write("\"NeuralNetwork\": { \"n_inputs\": " + str(self.n_inputs) +", \"n_layers\": " + str(self.n_layers) +", \"n_outputs\": " + str(self.n_outputs) + ", " + activations + ", \"base_activation\": " + self.base_activation.toString() + " }")