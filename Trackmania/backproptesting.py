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

iterations: int = 200000

with alive_bar(iterations, title='Training the AI!') as bar:
    # Train the network
    for _ in range(iterations):

        # Forward propagation
        output = np.array(nn.forward(training_data))
    
        nn.backpropagate(output_data-output)

        bar()

# Print the results
print(f"This method is EXTREMELY inconsistent so you might want to run it a couple times. Anyway heres your data!\nExpected outputs: {[str(i[0]) for i in output_data]}\nActual Output: {[str(i[0]) for i in BinaryStepActivation.forward(np.array(nn.forward(training_data))-0.5)]}")
print(output, output_data-output)