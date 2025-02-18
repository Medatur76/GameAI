# Trackmania AI
This is an AI tuned to play the racing game Trackmania 2020. The AI takes in 16 inputs, 15 wall distances and the speed, and it will output 4 values of 1 or 0. These control forward, left, backward, and right driving control. To get from 16 inputs to 4 outputs the AI runs a series of small, tuned calculations that take the data and turn it into directions using a combination of weights and biases. But how do we get these weights and biases? This will take use to the next section: [Training](#training)

## Training

Training is required for any AI to function properly. Training automates the process of determining the weights and biases of the AI. This AI comes with two ways to train the AI:

1: [I still cant remember the name lol](#i-still-cant-remember-the-name-lol)  
2: [Generational Training](#generational-training)

### I still cant remember the name lol

```Python
def train(self, n_layers: int=None, output_activations: list[Activation]=None, base_activation: Activation=Activation, runs: int=100, preset: preset = None):
    if preset == None:
        nn = NeuralNetwork(16, n_layers, 4, output_activations, base_activation)
    else:
        nn = NeuralNetwork.fromPreset(preset)
    lastScore = 0
    pydirectinput.press('del')
    if runs == None or runs == -1:
        running = True
        self.currentAI = 0
        while running:
            self.currentAI += 1
            score = self.run(nn, str(self.currentAI-1))
            if (score > lastScore): lastScore = score
            else: nn.revert()
    else:
        for r in range(runs):
            self.currentAI = r+1
            score = self.run(nn, str(r))
            if (score > lastScore): lastScore = score
            else: nn.revert()
    return nn
```

This is the most simple way to train any AI.  
The process is simple. Run the AI and test the output for a score. If the score is better or worse than the last one (depending on which one you want), save this new score and run again. If the score is better or worse than the last one (depending on which one you dont want), make a slight adjustment to the AI weights and biases and run again.  
As the loop repeats over and over and over again the score gets better and better, training the AI.

Now that our AI is trained we can [run](#running) it!

### Generational Training

```Python
def genTrain(self, n_layers: int=None, output_activations: list[Activation]=None, base_activation: Activation=Activation, generations: int=1, ais: int = 1, nnpreset: preset = None):
    if nnpreset == None: bestNetworks = [NeuralNetwork(16, n_layers, 4, output_activations, base_activation) for _ in range(ais*100)]
    else: bestNetworks = [NeuralNetwork.fromPreset(nnpreset) for _ in range(ais*100)]
    pydirectinput.press('del')
    for g in range(generations):
        self.currentGen = g+1
        scores: list[float] = []
        for n in range(len(bestNetworks)):
            self.currentAI = n+1
            scores.append(self.run(bestNetworks[n], f"G{str(g)}A{str(n)}"))
        sortedAis = [ai for _, ai in sorted(zip(scores, bestNetworks), key=lambda x: x[0], reverse=True)]
        bestNetworks.clear()
        returnedAis = ((len(sortedAis)*5)/100).__round__()
        multiplesOfAi = (100/returnedAis).__round__()
        bestNetworks5 = sortedAis[:returnedAis].copy()
        for n in bestNetworks5:
            for _ in range(multiplesOfAi):
                bestNetworks.append(n)
    return bestNetworks[0]
```

This is another, more complex way to train the AI. If you can grasp the concept it wil be very simple to understand.
The whole training system is based on a number of generations or groups. In each generation, there is a group of AIs with slightly different weight and bias combinations. All of these are run and sorted by their score. After they are all run, a certain number or percent of them are chosen to move on to the next generation. These chosen few are multiplied a few times and each given slightly different weights and biases and the process repeats. After all of the generations have been run, the best AI is returned.
Now that our AI is trained we can [run](#running) it!

## Running

```Python
while getInputs()[1][2] and not getInputs()[1][1]:
    data, _ = getInputs()

    output = bestRacer.forward(data)
    keys = []
    if output[0] == 1: keys.append('w')
    if output[1] == 1: keys.append('s')
    if output[2] == 1: keys.append('a')
    if output[3] == 1: keys.append('d')

    if not keys == []: self.pressKeys(keys)
```

Now that we have the best AI (``bestRacer``) we can run a simple script to get our inputs, process them, and execute the outputs until the user says for it to stop