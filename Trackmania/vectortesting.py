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

nn = NeuralNetwork(2, 5, 2, base_activation=HyperTangActivation)

iterations = 200000

with alive_bar(iterations, title='Training the AI!') as bar:
    # Train the network
    for i in range(iterations):

        r_i = np.random.randint(0, len(training_data)-1)

        nn.train()

        # Forward propagation
        output = np.array(nn.forward(training_data))

        actions = []

        for array in output:
            actions.append(np.random.normal(array[0], math.e**np.log(np.exp(array[1])), 1))

        nn.backpropagate(output_data-output, learning_rate=learningRate)

        bar()

actions = []

for a in output: actions.append([np.random.normal(a[0], 10**a[1], 1)[0]])

# Print the results
print(f"This method is EXTREMELY inconsistent so you might want to run it a couple times. Anyway heres your data!\nExpected outputs: {[str(i[0]) for i in output_data]}\nVector Output: {actions}\nActual Output: {output}")
#print(actions, output_data-actions, output)