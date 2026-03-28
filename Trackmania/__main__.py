from Library.NeuralClasses import NeuralNetwork
import Library.Activations as Activations
from alive_progress import alive_bar
import numpy as np
import math
import warnings
warnings.filterwarnings("error")

# Set random seed for reproducibility.
np.random.seed(42)

def stable_exp(x, clip_value=20):
    return math.e**np.log(np.exp(np.clip(x, 1e-8, clip_value)))

nn = NeuralNetwork(7, 3, 2, Activations.HyperbolicTangent)

# error: 2 * (final_output - target)

# Example inputs for the network
x = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]])
# Define a target value for the final output (e.g., supervised learning target)
target = np.array([1, 0, 0, 1, 0, 0, 0, 1])

iterations = 50000 * len(x)
#iterations = 400

with alive_bar(iterations, title="Training!") as bar:
    for i in range(iterations):
        output = nn.forward(x[i%len(x)])[0]
        g = stable_exp(output[1])
        nn.backward((np.random.normal(output[0], g) - target[i%len(x)]) * 2)
        bar()

print("\nFinal Outputs: ")
for a, i in zip(x, target):
    print(f"Inputs: {a} AI Output: {nn.forward(a)} Expected Output: {i}")

print(f"Inputs: [3 2 5] AI Output: {nn.forward(np.array([3,2,5]))[0]}")