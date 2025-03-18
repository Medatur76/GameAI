from AI.NeuralClasses.NeuralNetwork import NeuralNetwork
from AI.ActivationClasses.Activations import *
from alive_progress import alive_bar
import numpy as np
import math

training_data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
output_data = np.array([
    [0, 0],
    [1, 0],
    [1, 0],
    [0, 0]
])

nn = NeuralNetwork(2, 6, 2, base_activation=SigmoidActivation)

iterations = 500000

learningRate = 1

with alive_bar(iterations, title='Training the AI!', bar='filling') as bar:
    # Train the network
    for _ in range(iterations):

        # Forward propagation
        output = np.array(nn.forward(training_data))

        actions = []

        for array in output:
            actions.append(np.random.normal(array[0], math.e**np.log(np.exp(array[1])), 1))

        nn.backpropagate(output_data-output, learning_rate=learningRate)

        bar()

# Print the results
print(f"This method is EXTREMELY inconsistent so you might want to run it a couple times. Anyway heres your data!\nLearning Rate: {learningRate}\nExpected Outputs: {[str(i[0]) for i in output_data]}\nActual Output: {[str(i[0]) for i in actions]}\nError: {[str(i[0]) for i in -1*(output_data-actions)]}")