import numpy as np
import math

def activation(x):
    return np.tanh(x)

def activation_derivative(x):
    return 1.0 - np.tanh(x)**2

class NeuralLayer:
    def __init__(self, input_size, output_size, act=None, act_deriv=None):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = act
        self.activation_deriv = act_deriv

    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.weights) + self.biases
        self.output = self.activation(self.z) if self.activation is not None else self.z
        return self.output

    def backward(self, delta, lr):
        delta_z = delta * self.activation_deriv(self.z) if self.activation is not None else delta
        grad_w = np.dot(self.input.T, delta_z)
        grad_b = delta_z
        self.weights -= lr * grad_w
        self.biases -= lr * grad_b
        return np.dot(delta_z, self.weights.T)

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            if i < len(layer_sizes) - 2:
                self.layers.append(NeuralLayer(layer_sizes[i], layer_sizes[i+1], act=activation, act_deriv=activation_derivative))
            else:
                self.layers.append(NeuralLayer(layer_sizes[i], layer_sizes[i+1], act=None, act_deriv=None))

    def forward(self, x):
        activations = [x]
        for layer in self.layers:
            x = layer.forward(x)
            activations.append(x)
        return activations

    def backward(self, activations, error, lr):
        delta = np.array([[error, error]])
        for layer in reversed(self.layers):
            delta = layer.backward(delta, lr)

def train_step(nn, x, expected, lr=0.01):
    activations = nn.forward(x)
    output = activations[-1]
    mu, sigma = output[0, 0], output[0, 1]
    sigma_corrected = math.e**np.log(np.exp(sigma))
    sampled_output = np.random.normal(mu, sigma_corrected)
    error = (sampled_output - expected) * 2
    nn.backward(activations, error, lr)
    loss = 0.5 * (sampled_output - expected) ** 2
    return loss, mu, sigma, sigma_corrected, sampled_output

if __name__ == '__main__':
    np.random.seed(42)
    layer_sizes = [2, 4, 6, 5, 4, 2]
    nn = NeuralNetwork(layer_sizes)
    x = np.array([[0.5, -0.2]])
    expected_output = 1.2
    epochs = 100000
    for epoch in range(epochs):
        loss, mu, sigma, sigma_corrected, sampled = train_step(nn, x, expected_output, lr=0.01)
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1:4d}: Loss = {loss:.4f}, mu = {mu:.4f}, sigma_raw = {sigma:.4f}, sigma_corrected = {sigma_corrected:.4f}, sampled = {sampled:.4f}")
    final_output = nn.forward(x)[-1]
    mu_final, sigma_final = final_output[0, 0], final_output[0, 1]
    sigma_corrected_final = math.e**np.log(np.exp(sigma_final))
    final_sample = np.random.normal(mu_final, sigma_corrected_final)
    print("\nFinal outputs:")
    print("mu =", mu_final, "sigma_raw =", sigma_final, "sigma_corrected =", sigma_corrected_final, "sampled output =", final_sample)
