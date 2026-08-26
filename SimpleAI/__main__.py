from alive_progress import alive_bar
from Library.NeuralClasses import NeuralNetwork
import Library.Activations as Activations
import numpy as np

if __name__ == '__main__':
    nn = NeuralNetwork(4, 2, 1, Activations.Activation)
    iterations = 1000000

    x = [
        [1, 2],
        [2, 1],
        [3, 4],
        [4, 3],
        [5, 6],
        [10, 10],
        [39, 5]
    ]

    Y = list(map(lambda j: j[0] + j[1], x))

    with alive_bar(iterations, title="Training!") as bar:
        for i in range(iterations):
            output = nn.forward(x[i%len(x)])[0][0]
            loss_gradient = 2 * (output - Y[i%len(x)])  # Derivative of MSE loss
            nn.backward(loss_gradient)
            bar()
    for inputs in x:
        print(f"Input: {inputs} Output: {nn.forward(inputs)[0][0]} Expected: {sum(inputs)}")
    print(
        f"Final score: {np.sum(np.abs(np.subtract(nn.forward(x), Y)))}")

    while 1:
        a = input("First number: ")
        if a == "exit":
            break
        else:
            num1 = int(a)
        num2 = int(input("Second number: "))
        print(f"Sum: {nn.forward([num1, num2])}")