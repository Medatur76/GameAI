from AI.NeuralClasses.NeuralLayer import NeuralLayer
from AI.ActivationClasses.Activations import *
from typing_extensions import TypeAlias
from typing import Literal
import json

preset: TypeAlias = Literal["Yosh", "Racer"]

class NeuralNetwork(): 
    @classmethod
    def fromFile(nn, file_path: str):
        data = json.load(open(file_path, "r"))["NeuralNetwork"]
        return nn(None, None, None, None, None, [NeuralLayer.fromData(layer) for layer in data["Layers"]], fromFile = True, preset = data["preset"])
    @classmethod
    def fromPreset(nn, preset: preset):
        if preset == "Yosh":
            return nn(16, 2, 4, [BinaryStepActivation, BinaryStepActivation, BinaryStepActivation, BinaryStepActivation], SigmoidActivation, [NeuralLayer(False, 16, 64, SigmoidActivation), NeuralLayer(False, 64, 16, SigmoidActivation), NeuralLayer(False, 16, 4, BinaryStepActivation, [BinaryStepActivation, BinaryStepActivation, BinaryStepActivation, BinaryStepActivation])], preset)
        elif preset == "Racer":
            return nn.fromFile("Racer.nn")
    def __init__(self, n_inputs: int, n_layers: int, n_outputs: int, output_activations: list[Activation], base_activation: Activation=Activation, layers: list[NeuralLayer]=None, preset: preset = None, fromFile: bool = False):
        self.preset: preset | None = preset
        if not fromFile:
            self.layers: list[NeuralLayer] = [];
            if layers == None:
                for i in range(n_layers): self.layers.append(NeuralLayer(n_inputs+i*2, n_inputs+(i+1)*2, base_activation))
                self.layers.append(NeuralLayer(n_inputs+(n_layers)*2, n_outputs, None, output_activations))
            else:
                self.layers = layers
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
        data = "{\r\n\t\"NeuralNetwork\": {\r\n\t\t"
        if not self.preset == None:
            data += "\"preset\": \"" + self.preset + "\","
        else:
            pass
        data += "\r\n\t\t\"Layers\": ["
        for layer in self.layers:
            data += "\r\n\t\t\t{\r\n\t\t\t\t\"n_inputs\": " + str(layer.nInputs) + ",\r\n\t\t\t\t\"n_outputs\": " + str(layer.nOutputs) + ","
            if layer.multiActivations:
                data += "\r\n\t\t\t\t\"activations\": ["
                for activ in layer.activations:
                    data += f"\r\n\t\t\t\t\t\"{activ.toString()}\","
                data = data[:-1] + "\r\n\t\t\t\t],"
            else:
                data += "\r\n\t\t\t\t\"activation\": \"" + layer.activation.toString() + "\",\r\n\t\t\t\t\"activations\": \"None\","
            data += "\r\n\t\t\t\t\"weights\": ["# + str(layer.weights) + ", \r\n\t\t\t\t\"biases\": " + str(layer.biases) + "\r\n\t\t\t},"
            for i in layer.weights:
                data += "\r\n\t\t\t\t\t["
                for n in i:
                    data += "\r\n\t\t\t\t\t\t" + str(n) + ","
                data = data[:-1] + "\r\n\t\t\t\t\t],"
            data = data[:-1] + "\r\n\t\t\t\t],\r\n\t\t\t\t\"biases\": ["
            for i in layer.biases[0]:
                data += "\r\n\t\t\t\t\t" + str(i) + ","
            data = data[:-1] + "\r\n\t\t\t\t]\r\n\t\t\t},"
        data = data[:-1] + "\r\n\t\t]\r\n\t}\r\n}"
        file.write(data)
    def backpropagate(self, deriv):
        hiddenLayersReversed = self.layers.copy()
        hiddenLayersReversed.reverse()
        hiddenLayersReversed = hiddenLayersReversed[:-1]