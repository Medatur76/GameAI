from AI.NeuralClasses.NeuralNetwork import NeuralNetwork
from AI.ActivationClasses.Activations import SigmoidActivation, Activation
from alive_progress import alive_bar
import numpy as np

nn = NeuralNetwork(3, 5, 2, output_activations=[Activation], base_activation=SigmoidActivation)

# error: 2 * (final_output - target)

# Example inputs for the network
x = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]])
# Define a target value for the final output (e.g., supervised learning target)
target = np.array([1, 0, 0, 1, 0, 0, 0, 1])

iterations = 1000000

with alive_bar(iterations, title="Training!") as bar:
    for _ in range(iterations):
        i = np.random.randint(len(x))
        nn.forward(x[i])
        nn.distributionPropagation(target[i], 0.1)
        bar()

print("\nFinal Outputs: ")
for a, i in zip(x, target):
    print(f"Inputs: {a} AI Output: {nn.forward(a)} Expected Output: {i}")