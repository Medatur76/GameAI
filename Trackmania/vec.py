from AI.NeuralClasses.NeuralNetwork import NeuralNetwork
from AI.ActivationClasses.Activations import SigmoidActivation, Activation

nn = NeuralNetwork(3, 5, 2, output_activations=[Activation], base_activation=SigmoidActivation)

# error: 2 * (final_output - target)