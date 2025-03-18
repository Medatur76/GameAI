import numpy as np
import math

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_output):
    """Derivative of the sigmoid function given its output."""
    return sigmoid_output * (1 - sigmoid_output)

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Initialize weights and biases randomly.
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.random.randn(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.random.randn(output_dim)

    def forward(self, x):
        """Forward pass: computes hidden activations and two outputs (mu and sigma),
           then samples final output using np.random.normal."""
        self.x = x  # Store input for backpropagation
        # Hidden layer using sigmoid activation
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        # Output layer: two values (mu and sigma parameter)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # Apply sigmoid activation on the output layer
        # self.a2 = sigmoid(self.z2)
        self.a2 = self.z2
        self.mu = self.a2[0]
        # Ensure sigma is positive using exponentiation
        self.sigma = np.exp(self.a2[1])
        # Sample final output from a normal distribution with mean mu and std sigma
        self.final_output = np.random.normal(self.mu, self.sigma)
        return self.final_output

    def backward(self, target, learning_rate):
        """
        Backpropagation with heuristic gradient approximations for the stochastic sampling.
        Computes gradients and updates weights.
        """
        # Compute Mean Squared Error loss
        # loss = (self.final_output - target) ** 2
        # Derivative of loss with respect to the final output.
        dL_dfinal = 2 * (self.final_output - target)
        
        # Heuristic gradient approximations:
        # For mu: d(final)/d(mu) = 1, so:
        dL_dmu = dL_dfinal
        # For sigma: approximate derivative using:
        # d(final)/d(sigma) ~ (final_output - mu) / sigma
        dL_dsigma = dL_dfinal * (self.final_output - self.mu) / (self.sigma + 1e-8)
        
        # Now, a2 is produced from z2 via a sigmoid.
        # mu = a2[0] and sigma = exp(a2[1]). Thus:
        #   d(mu)/d(a2[0]) = 1, and 
        #   d(sigma)/d(a2[1]) = exp(a2[1]) = sigma.
        #
        # So the gradients with respect to a2 are:
        dL_da2 = np.zeros_like(self.a2)
        dL_da2[0] = dL_dmu  # for mu
        dL_da2[1] = dL_dsigma * self.sigma  # chain rule for sigma
        
        # Backprop through the sigmoid activation on z2:
        # dL_dz2 = dL_da2 * sigmoid_derivative(self.a2)
        dL_dz2 = dL_da2
        
        # Gradients for the output layer weights and biases:
        dL_dW2 = np.outer(self.a1, dL_dz2)
        dL_db2 = dL_dz2
        
        # Propagate gradient into the hidden layer:
        dL_da1 = np.dot(self.W2, dL_dz2)
        dL_dz1 = dL_da1 * sigmoid_derivative(self.a1)
        dL_dW1 = np.outer(self.x, dL_dz1)
        dL_db1 = dL_dz1
        
        # Update weights and biases:
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1

def ab(x: np.array, nn: NeuralNetwork) -> None:
    if x[1] == 0:
        print(f"{x[0]} XOR {x[2]} returns {nn.forward(x):.4f}")
    elif x[1] == 1:
        print(f"{x[0]} AND {x[2]} returns {nn.forward(x):.4f}")

def ch(l: np.array) -> bool:
    return (l[0] > 1 and l[1] < 0 and l[2] < 0 and l[3] > 1 and l[4] < 0 and l[5] < 0 and l[6] < 0 and l[7] > 1)

# -----------------------------
# Example usage:
# -----------------------------
if __name__ == "__main__":
    iterations = 5000000

    # Create a neural network with 3 inputs, one hidden layer with 4 neurons, and 2 outputs.
    nn = NeuralNetwork(input_dim=3, hidden_dim=4, output_dim=2)
    
    # Example inputs for the network
    x = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]])
    # Define a target value for the final output (e.g., supervised learning target)
    target = np.array([1, 0, 0, 1, 0, 0, 0, 1])
    learning_rate = 0.01

    # Train the network for a number of iterations
    for i in range(iterations):
        a = np.random.randint(len(x))
        # Forward pass: compute the final output after sampling
        output = nn.forward(x[a])
        
        # Compute squared error loss: (output - target)^2
        # loss = (output - target[a]) ** 2
        
        # Derivative of the squared loss with respect to the final output:
        # d(loss)/d(output) = 2*(output - target)
        
        # Backpropagate the error and update weights
        nn.backward(target[a], learning_rate)

        # Optionally print the loss every 100 iterations
        if (i*100)%iterations == 0:
            print(f"Iteration {i}, Loss: {np.sum(([nn.forward(a) for a in x] - target)**2):.4f}")
    print(f"Final output: ")
    for b in x:
        ab(b, nn)
    print(ch([nn.forward(a) for a in x]))