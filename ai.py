import numpy as np
import math

class SigmoidActivation():
    @staticmethod
    def forward(inputs):
        if not isinstance(inputs, list) and not isinstance(inputs, np.ndarray):
            return 1/(1+math.e**(-round(inputs, 6)))
        else:
            return np.array([SigmoidActivation.forward(i) for i in inputs])
    @staticmethod
    def toString():
        return "Sigmoid"
    @staticmethod
    def derivative(inputs):
        if not isinstance(inputs, list):
            return SigmoidActivation.forward(inputs)*(1-SigmoidActivation.forward(inputs))
        else:
            return np.array([SigmoidActivation.derivative(i) for i in inputs])

def clamp(low: float, num: float, high: float):
    if num < low: return low
    elif num > high: return high
    else: return num

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        Initialize the neural network.
        layer_sizes: list of integers specifying the size of each layer.
                    For example, [2, 4, 6, 5, 4, 2] means:
                    Input layer: 2 neurons,
                    Hidden layer1: 4 neurons,
                    Hidden layer2: 6 neurons,
                    Hidden layer3: 5 neurons,
                    Hidden layer4: 4 neurons,
                    Output layer: 2 neurons (for μ and σ).
        """
        self.num_layers = len(layer_sizes)
        self.weights = []
        self.biases = []
        
        # Initialize weights with He initialization and biases with zeros.
        for i in range(self.num_layers - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, x):
        """
        Perform a forward pass through the network.
        Returns a list of activations and the corresponding pre-activation values (zs).
        """
        activations = [x]
        zs = []  # Store all the linear combinations
        a = x
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            # For the final layer, use a linear output.
            if i == len(self.weights) - 1:
                a = z  
            else:
                a = SigmoidActivation.forward(z)
            activations.append(a)
            zs.append(a)
        return activations, zs

    def backward(self, activations, zs, delta_vector):
        """
        Backpropagate the error given the gradient vector at the output layer.
        delta_vector: gradient of the loss with respect to the output layer (shape: 1x2)
        Returns gradients for weights and biases.
        """
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)
        
        # Backpropagation from the output layer backward.
        for l in range(len(self.weights) - 1, -1, -1):
            a_prev = activations[l]  # activation from previous layer
            grads_w[l] = np.dot(a_prev.T, delta_vector)
            grads_b[l] = delta_vector  # For a single example, this is fine
            
            if l > 0:
                # Propagate error through the activation function of the previous layer.
                delta_vector = np.dot(delta_vector, self.weights[l].T) * SigmoidActivation.derivative(zs[l-1])
                
        return grads_w, grads_b

    def update_params(self, grads_w, grads_b, lr):
        """Update weights and biases using gradient descent."""
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads_w[i]
            self.biases[i] -= lr * grads_b[i]

# Function to perform one training step using negative log-likelihood loss.
def train_step(nn, x, expected, lr=0.01):
    """
    Performs a training step.
    The network outputs two values: μ and σ.
    σ is transformed as: sigma_corrected = exp(σ) to enforce positivity.
    Loss is defined as: L = σ + (expected - μ)^2/(2*exp(2σ))
    """
    # Forward pass.
    activations, zs = nn.forward(x)
    output = activations[-1]  # Final layer output: shape (1,2)
    
    # Separate outputs: first element is μ, second is σ (before transformation).
    mu = output[0, 0]
    sigma = output[0, 1]
    
    # Transform sigma to ensure positivity.
    sigma_corrected = np.exp(sigma)

    a = 1e-8
    
    # Compute negative log likelihood loss (ignoring constant terms).
    loss = sigma + (expected - mu)**2 / (2 * np.exp(2 * (sigma + a)))
    
    # Compute gradients with respect to μ and σ.
    grad_mu = -(expected - mu) / (np.exp(2 * (sigma + a)))
    grad_sigma = 1 - (expected - mu)**2 / (np.exp(2 * (sigma + a)))
    
    # Form the gradient vector for the final layer.
    delta_vector = np.array([[grad_mu, grad_sigma]])
    
    # Backpropagate the error.
    grads_w, grads_b = nn.backward(activations, zs, delta_vector)
    nn.update_params(grads_w, grads_b, lr)
    
    return loss, mu, sigma, sigma_corrected

# Example usage:
if __name__ == '__main__':
    # Set random seed for reproducibility.
    np.random.seed(42)
    
    # Define network architecture: [input, layer1, layer2, layer3, layer4, output]
    layer_sizes = [3, 4, 6, 5, 7, 2]
    nn = NeuralNetwork(layer_sizes)
    
    # Example inputs for the network
    x = [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]]
    # Define a target value for the final output (e.g., supervised learning target)
    target = np.array([1, 0, 0, 1, 0, 0, 0, 1])
    
    # Training loop.
    epochs = 1000000
    for epoch in range(epochs):
        loss, mu, sigma, sigma_corrected = train_step(nn, np.array([x[epoch%(len(x))]]), target[epoch%(len(x))], lr=0.001)
        if (epoch + 1) % 20000 == 0:
            Output = []
            for a in x:
                Output.append(nn.forward([a])[0][-1][0])
            Output = np.array(Output)
            print(f"Epoch: {epoch+1:4d} {Output=}")
    
    # Final forward pass for demonstration.
    print("\nFinal outputs:")
    for i, o in zip(x, target):
        p = nn.forward(np.array([i]))[0][-1][0]
        p[1] = np.exp(p[1])
        print(f"Input: {i} output: {np.random.normal(p[0], p[1])} Expected output: {o}")