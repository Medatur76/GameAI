import numpy as np
from alive_progress import alive_bar

layers = [1, 2, 3, 4]
outputLayer = layers[-1:][0]
layers = layers[:-1]
layers.reverse()
print(outputLayer, layers)

# Define the activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def binary_step(x):
    return np.where(x >= 0, 1, 0)

def binary_step_derivative(x):
    return np.zeros_like(x)

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

iterations = 1000000

with alive_bar(iterations, title='Training the AI!') as bar:
  # Train the network
  for i in range(iterations):

    # Forward propagation
    layer0 = X
    layer1 = sigmoid(np.dot(layer0, weights0) + biases0)

    # Apply different activations to layer2 outputs
    #layer2_part1 = sigmoid(np.dot(layer1, weights1[:,:2]) + biases1[:,:2])
    #layer2_part2 = binary_step(np.dot(layer1, weights1[:,2:]) + biases1[:,2:])
    layer2 = sigmoid(np.dot(layer1, weights1) + biases1)
    
    # Concatenate parts
    #layer2 = np.concatenate([layer2_part1, layer2_part2], axis=1)
    
    layer3 = sigmoid(np.dot(layer2, weights2) + biases2)

    # Backpropagation
    layer3_error = y - layer3
    layer3_delta = layer3_error * sigmoid_derivative(layer3)

    layer2_error = layer3_delta.dot(weights2.T)
    layer2_delta = layer2_error * sigmoid_derivative(layer2)
    
    # Split errors for different parts of layer2
    #layer2_error_part1 = layer2_error[:,:2] * sigmoid_derivative(layer2_part1)
    #layer2_error_part2 = layer2_error[:,2:] * binary_step_derivative(layer2_part2)
    
    # Concatenate deltas
    #layer2_delta = np.concatenate([layer2_error_part1, layer2_error_part2], axis=1)
    
    layer1_error = layer2_delta.dot(weights1.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    # Update weights and biases
    weights2 += layer2.T.dot(layer3_delta)
    biases2 += np.sum(layer3_delta, axis=0, keepdims=True)

    #weights1[:,:2] += layer1.T.dot(layer2_delta[:,:2])
    #weights1[:,2:] += layer1.T.dot(layer2_delta[:,2:])
    #biases1[:,:2] += np.sum(layer2_delta[:,:2], axis=0, keepdims=True)
    #biases1[:,2:] += np.sum(layer2_delta[:,2:], axis=0, keepdims=True)
    weights1 += layer1.T.dot(layer2_delta)
    biases1 += np.sum(layer2_delta, axis=0, keepdims=True)

    weights0 += layer0.T.dot(layer1_delta)
    biases0 += np.sum(layer1_delta, axis=0, keepdims=True)

    bar()

# Print the results
print(binary_step(layer3-0.5))