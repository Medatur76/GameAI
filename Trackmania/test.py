from AI.NeuralClasses.NeuralLayer import NeuralLayer

from AI.ActivationClasses.Sigmoid import SigmoidActivation

layer = NeuralLayer(3, 5, activation=SigmoidActivation)

layer.forward([1, 0, 0])
print(layer.output)
layer.forward([0, 0, 0])
print(layer.output)