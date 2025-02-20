import numpy as np
from alive_progress import alive_bar
import math

# Define the sigmoid activation function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
  #return 1 + np.log(math.e**x)

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
  return x * (1 - x)
  #return 1/(1+(math.e**(-x)))

def relu(x):
  return np.maximum(0, x)

# Define the derivative of the ReLU function
def relu_derivative(x):
  return np.where(x > 0, 1, 0)

# Define the input data
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

# Define the output data
y = np.array([[0], [1], [1], [0]])

xl = len(X)

# Initialize the weights and biases
weights0 = np.random.rand(len(X[0]), xl)
weights1 = np.random.rand(xl, xl)
weights2 = np.random.rand(xl, len(y[0]))
biases0 = np.random.rand(1, xl)
biases1 = np.random.rand(1, xl)
biases2 = np.random.rand(1, len(y[0]))

iterations = 10000

with alive_bar(iterations, title='Training the AI!', length=20, bar='fish') as bar:
    # Train the network
    for i in range(iterations):

        # Forward propagation
        layer0 = X
        layer1 = relu(np.dot(layer0, weights0) + biases0)
        layer2 = relu(np.dot(layer1, weights1) + biases1)
        layer3 = relu(np.dot(layer2, weights2) + biases2)

        # Backpropagation
        layer3_error = y - layer3
        layer3_delta = layer3_error * sigmoid_derivative(layer3)

        layer2_error = layer3_delta.dot(weights2.T)
        layer2_delta = layer2_error * sigmoid_derivative(layer2)

        layer1_error = layer2_delta.dot(weights1.T)
        layer1_delta = layer1_error * sigmoid_derivative(layer1)

        # Update weights and biases
        weights2 += layer2.T.dot(layer3_delta)
        biases2 += np.sum(layer3_delta, axis=0, keepdims=True)

        weights1 += layer1.T.dot(layer2_delta)
        biases1 += np.sum(layer2_delta, axis=0, keepdims=True)

        weights0 += layer0.T.dot(layer1_delta)
        biases0 += np.sum(layer1_delta, axis=0, keepdims=True)

        bar()

# Print the results
print(layer3)