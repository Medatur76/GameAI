from NeuralClasses.NeuralNetwork import NeuralNetwork

#nn = NeuralNetwork.fromPreset("Yosh")
nn = NeuralNetwork.fromFile("Racer.nn")

nn.save()