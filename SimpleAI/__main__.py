from alive_progress import alive_bar
from Library.NeuralClasses import NeuralNetwork
import Library.Activations as Activations
import numpy as np

if __name__ == '__main__':
    nn = NeuralNetwork([1, 16, 16, 1], Activations.Sigmoid)
    iterations = 2000000

    rng = np.random.default_rng()

    x = rng.random(size=(40, 1)) * np.pi

    Y = np.sin(x)

    with alive_bar(iterations, title="Training!") as bar:
        for i in range(iterations):
            output = nn.forward(x[i%len(x)])[0][0]
            loss_gradient = 2 * (output - Y[i%len(x)][0])  # Derivative of MSE loss
            nn.backward(loss_gradient, 1e-4)
            bar()
    for i, inputs in enumerate(x):
        print(f"Input: {inputs} Output: {nn.forward(inputs)[0][0]} Expected: {Y[i][0]}")
    print(
        f"Final score: {np.sum(np.abs(np.subtract(nn.forward(x), Y)))}")

    while 1:
        a = input("First number: ")
        if a == "exit":
            break
        else:
            num1 = float(a)
        #num2 = int(input("Second number: "))
        print(f"Sum: {nn.forward([num1])}")