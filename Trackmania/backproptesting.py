from AI.NeuralClasses.NeuralNetwork import NeuralNetwork
from AI.ActivationClasses.Activations import *
from alive_progress import alive_bar
import numpy as np

training_data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
output_data = np.array([
    [0],
    [1],
    [1],
    [0]
])

nn = NeuralNetwork(2, 3, 1, base_activation=SigmoidActivation)

iterations = 1000000

with alive_bar(iterations, title='Training the AI!') as bar:
    # Train the network
    for i in range(iterations):

        # Forward propagation
        output = np.array(nn.forward(training_data))
    
        nn.backpropagate(output_data-output)

        bar()

# Print the results
print(nn.forward(training_data))