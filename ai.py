import numpy as np

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_output):
    """Derivative of the sigmoid function given its output."""
    return sigmoid_output * (1 - sigmoid_output)

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        # First hidden layer parameters.
        self.W1 = np.random.randn(input_dim, hidden_dim1) * 0.1
        self.b1 = np.zeros((1, hidden_dim1))
        # Second hidden layer parameters.
        self.W2 = np.random.randn(hidden_dim1, hidden_dim2) * 0.1
        self.b2 = np.zeros((1, hidden_dim2))
        # Output layer parameters.
        self.W3 = np.random.randn(hidden_dim2, output_dim) * 0.1
        self.b3 = np.zeros((1, output_dim))

    def forward(self, X):
        """
        Forward pass for a mini-batch of inputs X with shape (batch_size, input_dim).
        The network architecture is:
          Input -> Hidden Layer 1 (sigmoid) -> Hidden Layer 2 (sigmoid) ->
          Output Layer (sigmoid) -> [Interpretation into mu and sigma] -> Sampling.
        """
        self.X = X  # (batch_size, input_dim)

        # First hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1         # (batch_size, hidden_dim1)
        self.a1 = sigmoid(self.z1)                     # (batch_size, hidden_dim1)

        # Second hidden layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2     # (batch_size, hidden_dim2)
        self.a2 = sigmoid(self.z2)                     # (batch_size, hidden_dim2)

        # Output layer
        self.z3 = np.dot(self.a2, self.W3) + self.b3     # (batch_size, output_dim)
        #self.a3 = sigmoid(self.z3)                     # (batch_size, output_dim)
        self.a3 = self.z3

        # Interpret outputs:
        #   a3[:, 0] is mu and a3[:, 1] (after exponentiation) is sigma.
        self.mu = self.a3[:, 0]                        # (batch_size,)
        self.sigma = np.exp(self.a3[:, 1])              # (batch_size,)

        # Sample final output for each example.
        batch_size = X.shape[0]
        self.final_output = np.random.normal(self.mu, self.sigma, size=batch_size)
        return self.final_output

    def backward(self, target, learning_rate):
        """
        Backward pass for the mini-batch.
        'target' can be a scalar or an array of shape (batch_size,).
        Returns the average loss over the mini-batch.
        """
        batch_size = self.X.shape[0]
        # Compute Mean Squared Error loss per example and then average.
        loss = np.mean((self.final_output - target)**2)

        # Derivative of loss with respect to final output.
        dL_dfinal = 2 * (self.final_output - target) / batch_size

        # Heuristic gradient approximations for the stochastic sampling:
        # d(final_output)/d(mu) = 1, and approximately:
        # d(final_output)/d(sigma) ~ (final_output - mu) / sigma.
        eps = 1e-8  # small constant to prevent division by zero
        dL_dmu = dL_dfinal
        dL_dsigma = dL_dfinal * (self.final_output - self.mu) / (self.sigma + eps)

        # Now, a3 is the activated output, with:
        # mu = a3[:, 0] and sigma = exp(a3[:, 1]), so:
        # d(mu)/d(a3[:, 0]) = 1 and d(sigma)/d(a3[:, 1]) = exp(a3[:, 1]) = sigma.
        dL_da3 = np.zeros_like(self.a3)
        dL_da3[:, 0] = dL_dmu
        dL_da3[:, 1] = dL_dsigma * self.sigma

        # Backpropagate through the sigmoid activation at the output layer.
        dL_dz3 = dL_da3 #* sigmoid_derivative(self.a3)  # (batch_size, output_dim)

        # Gradients for the output layer parameters.
        dL_dW3 = np.dot(self.a2.T, dL_dz3)              # (hidden_dim2, output_dim)
        dL_db3 = np.sum(dL_dz3, axis=0, keepdims=True)    # (1, output_dim)

        # Backpropagate into second hidden layer.
        dL_da2 = np.dot(dL_dz3, self.W3.T)                # (batch_size, hidden_dim2)
        dL_dz2 = dL_da2 * sigmoid_derivative(self.a2)     # (batch_size, hidden_dim2)

        dL_dW2 = np.dot(self.a1.T, dL_dz2)                # (hidden_dim1, hidden_dim2)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)      # (1, hidden_dim2)

        # Backpropagate into first hidden layer.
        dL_da1 = np.dot(dL_dz2, self.W2.T)                # (batch_size, hidden_dim1)
        dL_dz1 = dL_da1 * sigmoid_derivative(self.a1)     # (batch_size, hidden_dim1)

        dL_dW1 = np.dot(self.X.T, dL_dz1)                 # (input_dim, hidden_dim1)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)      # (1, hidden_dim1)

        # Update all parameters.
        self.W3 -= learning_rate * dL_dW3
        self.b3 -= learning_rate * dL_db3
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1

        return loss

# -----------------------------
# Example usage:
# -----------------------------
if __name__ == "__main__":
    np.random.seed(42)  # for reproducibility

    # Hyperparameters.
    input_dim = 2
    hidden_dim1 = 8      # first hidden layer neurons
    hidden_dim2 = 3      # second hidden layer neurons (as requested)
    output_dim = 2
    learning_rate = 0.01
    num_iterations = 100000
    batch_size = 32

    # Create the neural network.
    nn = NeuralNetwork(input_dim, hidden_dim1, hidden_dim2, output_dim)
    
    # For demonstration, we use a fixed target (or an array of targets).
    target = 0.5

    for i in range(num_iterations):
        # Generate a mini-batch of random inputs.
        X_batch = np.random.randn(batch_size, input_dim)
        # Forward pass.
        outputs = nn.forward(X_batch)
        # Backward pass.
        loss = nn.backward(target, learning_rate)
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss:.6f}")
