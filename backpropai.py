from alive_progress import alive_bar
from Library.NeuralClasses import NeuralNetwork
import Library.Activations as Activations

if __name__ == '__main__':
    nn = NeuralNetwork(4, 2, 1, Activations.HyperbolicTangent)
    iterations = 1000000

    x = []

    Y = []

    with alive_bar(iterations, title="Training!") as bar:
        for i in range(iterations):
            output = nn.forward(x[i%len(x)])[0][0]
            loss_gradient = 2 * (output - Y[i%len(x)])  # Derivative of MSE loss
            nn.backward(loss_gradient)
            bar()
            iterations -= 1
    
    while 1:
