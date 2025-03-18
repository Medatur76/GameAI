import numpy as np
import math

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Initialize weights and biases randomly.
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.random.randn(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.random.randn(output_dim)

    def forward(self, x) -> float:
        """Forward pass: computes hidden activations and two outputs (mu and sigma)."""
        self.x = x  # Store input for backprop
        # Hidden layer activation
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        # Output layer: two values (mu and sigma)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.mu = self.z2[0]
        self.sigma = math.e**np.log(np.exp(self.z2[1]))  # Ensure positive standard deviation

        # Direct sampling from normal distribution
        self.final_output = np.random.normal(self.mu, self.sigma)
        return self.final_output

    def backward(self, target: float = 0, learning_rate: float = 1.0) -> None:
        """
        Backpropagation through heuristic estimation of gradients.
        This approximates the gradient of the stochastic sampling operation.
        """
        
        # Estimate gradients using heuristic:
        # d(loss)/d(final_output) = 2 * (final_output - target)
        dL_dfinal = 2 * (self.final_output - target)
        
        # Heuristic backpropagation:
        # Since final_output is directly sampled, we approximate:
        dL_dmu = dL_dfinal  # Since mu directly affects final_output
        dL_dsigma = dL_dfinal * (self.final_output - self.mu) / (self.sigma + 1e-8)  # Approximate gradient w.r.t sigma

        # Gradients for z2 (output layer pre-activation values)
        dL_dz2_0 = dL_dmu
        dL_dz2_1 = dL_dsigma * self.sigma  # Since sigma = exp(z2[1]), we scale the gradient

        dL_dz2 = np.array([dL_dz2_0, dL_dz2_1])

        # Gradients for W2 and b2
        dL_dW2 = np.outer(self.a1, dL_dz2)
        dL_db2 = dL_dz2

        # Backprop into hidden layer
        dL_da1 = np.dot(self.W2, dL_dz2)
        dL_dz1 = dL_da1 * (1 - np.tanh(self.z1) ** 2)  # Derivative of tanh
        dL_dW1 = np.outer(self.x, dL_dz1)
        dL_db1 = dL_dz1

        # Gradient descent update
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
    # iterations = 1000000

    # Create a neural network with 3 inputs, one hidden layer with 4 neurons, and 2 outputs.
    nn = NeuralNetwork(input_dim=3, hidden_dim=4, output_dim=2)
    
    # Example inputs for the network
    x = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]])
    # Define a target value for the final output (e.g., supervised learning target)
    target = np.array([1, 0, 0, 1, 0, 0, 0, 1])
    learning_rate = 0.01

    i = 0

    # Train the network for a number of iterations
    while not ch([nn.forward(a) for a in x]):
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
        if (i%100) == 0:
            print(f"Iteration {i}, Loss: {np.mean(([nn.forward(a) for a in x] - target)**2):.4f}")
        i += 1
    print(f"Final output: ")
    for b in x:
        ab(b, nn)
    print(ch([nn.forward(a) for a in x]))