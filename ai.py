import numpy as np

# --- 1. Define the Neural Network Structure ---
def initialize_network(input_size, hidden_size1, hidden_size2, output_size):
    """Initializes the weights and biases for the network."""
    network = {}
    network['W1'] = np.random.randn(input_size, hidden_size1) * 0.01  # Weights layer 1
    network['b1'] = np.zeros((1, hidden_size1))                 # Biases layer 1
    network['W2'] = np.random.randn(hidden_size1, hidden_size2) * 0.01 # Weights layer 2
    network['b2'] = np.zeros((1, hidden_size2))                # Biases layer 2
    network['W3'] = np.random.randn(hidden_size2, output_size) * 0.01 # Weights layer 3
    network['b3'] = np.zeros((1, output_size))               # Biases layer 3
    return network

# --- 2. Activation Function (ReLU) ---
def relu(x):
    """Rectified Linear Unit activation function."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU for backpropagation."""
    return (x > 0).astype(float)

# --- 3. Forward Propagation ---
def forward_propagation(network, inputs):
    """Performs forward propagation through the network."""
    W1, b1, W2, b2, W3, b3 = network['W1'], network['b1'], network['W2'], network['b2'], network['W3'], network['b3']

    Z1 = np.dot(inputs, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = Z3 # No activation on the final layer for simplicity.

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}
    return A3, cache

# --- 4. Loss and Normal Distribution ---
def calculate_loss(predictions, expected):
    """Calculates the loss using normal distribution and comparing to expected value."""
    # Apply normal distribution to each output
    normal_dist_values = np.array([np.random.normal(pred[0], pred[1]) for pred in predictions])
    loss = np.mean((normal_dist_values - expected)**2) # Mean Squared Error
    return loss, normal_dist_values

# --- 5. Backpropagation ---
def backward_propagation(network, cache, inputs, predictions, expected, normal_dist_values):
    """Performs backpropagation to update weights and biases."""
    m = inputs.shape[0] # Number of training examples
    W3, W2, W1 = network['W3'], network['W2'], network['W1']
    A3, A2, A1, Z3, Z2, Z1 = cache['A3'], cache['A2'], cache['A1'], cache['Z3'], cache['Z2'], cache['Z1']

    # Calculate the gradient of the loss with respect to the output
    dZ3 = (normal_dist_values - expected).reshape(-1,1) * (predictions - normal_dist_values.reshape(-1,2)) #simple approximation.
    dW3 = (1 / m) * np.dot(A2.T, dZ3)
    db3 = (1 / m) * np.sum(dZ3, axis=0, keepdims=True)

    dZ2 = np.dot(dZ3, W3.T) * relu_derivative(Z2)
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

    dZ1 = np.dot(dZ2, W2.T) * relu_derivative(Z1)
    dW1 = (1 / m) * np.dot(inputs.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

    gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}
    return gradients

# --- 6. Update Parameters ---
def update_parameters(network, gradients, learning_rate):
    """Updates the network's parameters using the gradients."""
    network['W1'] -= learning_rate * gradients['dW1']
    network['b1'] -= learning_rate * gradients['db1']
    network['W2'] -= learning_rate * gradients['dW2']
    network['b2'] -= learning_rate * gradients['db2']
    network['W3'] -= learning_rate * gradients['dW3']
    network['b3'] -= learning_rate * gradients['db3']
    return network

# --- 7. Train the Network ---
def train(inputs, expected_values, hidden_size1, hidden_size2, learning_rate, epochs):
    """Trains the neural network."""
    input_size = inputs.shape[1]
    output_size = 2 # 2 outputs for mean and standard deviation of normal distribution.
    network = initialize_network(input_size, hidden_size1, hidden_size2, output_size)

    for epoch in range(epochs):
        predictions, cache = forward_propagation(network, inputs)
        loss, normal_dist_values = calculate_loss(predictions, expected_values)
        gradients = backward_propagation(network, cache, inputs, predictions, expected_values, normal_dist_values)
        network = update_parameters(network, gradients, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return network

# --- Example Usage ---
inputs = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]) # Example inputs
expected_values = np.array([0.5, 0.7, 0.9]) # Example expected values

trained_network = train(inputs, expected_values, hidden_size1=4, hidden_size2=4, learning_rate=0.01, epochs=1000)

# Example prediction
example_input = np.array([[0.7, 0.8]])
prediction, _ = forward_propagation(trained_network, example_input)
final_value = np.random.normal(prediction[0][0], prediction[0][1])
print(f"Prediction: {prediction}, Final Value: {final_value}")