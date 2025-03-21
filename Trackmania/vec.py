from AI.NeuralClasses.NeuralNetwork import NeuralNetwork
from AI.ActivationClasses.Activations import *
from alive_progress import alive_bar
import numpy as np
import math

# Set random seed for reproducibility.
#np.random.seed(42)

nn = NeuralNetwork(3, 6, 2, output_activations=[Activation], base_activation=SigmoidActivation)

# error: 2 * (final_output - target)

# Example inputs for the network
x = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]])
# Define a target value for the final output (e.g., supervised learning target)
target = np.array([1, 0, 0, 1, 0, 0, 0, 1])

iterations = 1000 * len(x) * 8
#iterations = 1

with alive_bar(iterations, title="Training!") as bar:
    for i in range(iterations):
        output = nn.forward(x[i%(len(x))])[1][-1]
        output[1] = math.e**np.log(np.exp(output[1]))
        nn.distributionPropagation((np.random.normal(output[0], output[1]) - target[i%(len(x))]) * 2, 0.001)
        if i%200==0:
            l = []
            for b, c in zip(x, target):
                o = nn.forward(b)[0][-1]
                d = np.random.normal(o[0], math.e**np.log(np.exp(o[1])))
                l.append((d - c)**2)
            l = np.array(l)
            print(f"Loss = {l.sum()}")
        bar()

print("\nFinal Outputs: ")
for a, i in zip(x, target):
    print(f"Inputs: {a} AI Output: {nn.forward(a)[0][0]} Expected Output: {i}")