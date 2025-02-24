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
    [0],
    [1],
    [1],
    [0]
])

NeuralNetwork(2, 6, 2, base_activation=SigmoidActivation).save("Vector")

errorList = []

nn = NeuralNetwork.fromFile("Vector")

iterations = 200000

learningRate = 1

with alive_bar(iterations, title='Training the AI!') as bar:
    # Train the network
    for _ in range(iterations):

        # Forward propagation
        output = np.array(nn.forward(training_data))

        actions = []

        for array in output:
            actions.append(np.random.normal(array[0], math.e**np.log(np.exp(array[1])), 1))

        nn.backpropagate(output_data-actions, learning_rate=learningRate)

        bar()

# Print the results
print(f"This method is EXTREMELY inconsistent so you might want to run it a couple times. Anyway heres your data!\nLearning Rate: {learningRate}\nExpected Outputs: {[str(i[0]) for i in output_data]}\nActual Output: {[str(i[0]) for i in actions]}\nError: {[str(i[0]) for i in -1*(output_data-actions)]}")

errorList.append(np.mean([abs(i) for i in -1*(output_data-actions)]))

nn = NeuralNetwork.fromFile("Vector")

learningRate = 0.1

with alive_bar(iterations, title='Training the AI!') as bar:
    # Train the network
    for _ in range(iterations):

        # Forward propagation
        output = np.array(nn.forward(training_data))

        actions = []

        for array in output:
            actions.append(np.random.normal(array[0], math.e**np.log(np.exp(array[1])), 1))

        nn.backpropagate(output_data-actions, learning_rate=learningRate)

        bar()

# Print the results
print(f"Learning Rate: {learningRate}\nExpected Outputs: {[str(i[0]) for i in output_data]}\nActual Output: {[str(i[0]) for i in actions]}\nError: {[str(i[0]) for i in -1*(output_data-actions)]}")

errorList.append(np.mean([abs(i) for i in -1*(output_data-actions)]))

nn = NeuralNetwork.fromFile("Vector")

learningRate = 0.01

with alive_bar(iterations, title='Training the AI!') as bar:
    # Train the network
    for _ in range(iterations):

        # Forward propagation
        output = np.array(nn.forward(training_data))

        actions = []

        for array in output:
            actions.append(np.random.normal(array[0], math.e**np.log(np.exp(array[1])), 1))

        nn.backpropagate(output_data-actions, learning_rate=learningRate)

        bar()

# Print the results
print(f"Learning Rate: {learningRate}\nThis method is EXTREMELY inconsistent so you might want to run it a couple times. Anyway heres your data!\nExpected Outputs: {[str(i[0]) for i in output_data]}\nActual Output: {[str(i[0]) for i in actions]}\nError: {[str(i[0]) for i in -1*(output_data-actions)]}")

errorList.append(np.mean([abs(i) for i in -1*(output_data-actions)]))

nn = NeuralNetwork.fromFile("Vector")

learningRate = 0.001

with alive_bar(iterations, title='Training the AI!') as bar:
    # Train the network
    for _ in range(iterations):

        # Forward propagation
        output = np.array(nn.forward(training_data))

        actions = []

        for array in output:
            actions.append(np.random.normal(array[0], math.e**np.log(np.exp(array[1])), 1))

        nn.backpropagate(output_data-actions, learning_rate=learningRate)

        bar()

# Print the results
print(f"Learning Rate: {learningRate}\nThis method is EXTREMELY inconsistent so you might want to run it a couple times. Anyway heres your data!\nExpected Outputs: {[str(i[0]) for i in output_data]}\nActual Output: {[str(i[0]) for i in actions]}\nError: {[str(i[0]) for i in -1*(output_data-actions)]}")

errorList.append(np.mean([abs(i) for i in -1*(output_data-actions)]))

rateList = [1, 0.1, 0.01, 0.001]

sortedRates = [[rate, error] for error, rate in sorted(zip(errorList, rateList), key=lambda x: x[0])]

print(f"The rates can be sorted into {[str(i[0]) for i in sortedRates]} with errors {[str(i[1]) for i in sortedRates]}")