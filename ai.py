import numpy as np
import math
import warnings
warnings.filterwarnings('error')

def activation(x):
    return np.tanh(x)

def activation_derivative(x):
    return 1.0 - np.tanh(x)**2

def custom_activation(x):
    if not isinstance(x, list) and not isinstance(x, np.ndarray):
        if x <= -1: return 0
        x += 1
        return -(((x**-math.e - x)/(x**math.e + x**-math.e))/1.35) + 0.7408
    else:
        return np.array([custom_activation(a) for a in x])

def custom_activation_derivative(x):
    if not isinstance(x, list) and not isinstance(x, np.ndarray):
        return (463/625) - ((20*((1/((x+1)**math.e)-x-1)))/(27*(((x+1)**math.e)+(1/((x+1)**math.e)))))
    else:
        return np.array([custom_activation_derivative(a) for a in x])

def stable_exp(x, clip_value=20):
    return math.e**np.log(np.exp(np.clip(x, None, clip_value)))

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
        delta_z = delta * (self.activation_deriv(self.z) if self.activation is not None else 1)
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
                self.layers.append(NeuralLayer(layer_sizes[i], layer_sizes[i+1], act=custom_activation, act_deriv=custom_activation_derivative))
            else:
                self.layers.append(NeuralLayer(layer_sizes[i], layer_sizes[i+1], act=None, act_deriv=None))

    def forward(self, x):
        activations = [x]
        for layer in self.layers:
            x = layer.forward(x)
            activations.append(x)
        return activations

    def backward(self, activations, delta, lr):
        for layer in reversed(self.layers):
            delta = layer.backward(delta, lr)

def train_step(nn, x, expected, lr=0.01):
    activations = nn.forward(x)
    output = activations[-1]
    mu, sigma = output[0, 0], output[0, 1]
    sigma_corrected = stable_exp(sigma)
    sampled_output = np.random.normal(mu, sigma_corrected)
    error = (sampled_output - expected) * 2
    loss = 0.5 * (sampled_output - expected)**2
    delta = np.array([[error, error/5]])
    nn.backward(activations, delta, lr)
    return loss, mu, sigma, sigma_corrected, sampled_output

if __name__ == '__main__':
    np.random.seed(42)  
    layer_sizes = [3, 4, 6, 5, 4, 2]
    nn = NeuralNetwork(layer_sizes)
    x_train: np.ndarray[list[int]] = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]])
    y_train = np.array([0, 1, 1, 0, 0, 0, 0, 1])
    epochs = 100000
    for epoch in range(epochs):
        epoch_loss = 0
        for x_val, y_val in zip(x_train, y_train):
            x_val = x_val.reshape(1, 3)
            loss, mu, sigma, sigma_corrected, sampled = train_step(nn, x_val, y_val, lr=0.01)
            epoch_loss += loss
        if (epoch + 1) % 5000 == 0:
            print(f"Epoch {epoch+1:4d}: Avg Loss = {epoch_loss/4:.4f}")
    print("\nXOR Predictions:")
    for x_val, y_val in zip(x_train, y_train):
        x_val = x_val.reshape(1, 3)
        output = nn.forward(x_val)[-1]
        mu, sigma = output[0, 0], output[0, 1]
        sigma_corrected = stable_exp(sigma)
        print(f"Input: {x_val.flatten()}, Predicted: {np.random.normal(mu, sigma_corrected):.4f} ({mu:.4f}), Target: {y_val}")
