from Trackmania.AI.Training.Inputs import getInputs
from Library.Activations import Activation
from Library.NeuralClasses import NeuralNetwork
import numpy as np
import time, pydirectinput, math

def seqPressKeys(keys: list[str]):
        for key in keys:
            pydirectinput.press(key)

class Training():
    nextUpKeys: list[str] = []
    isGen = False
    currentGen = -1
    currentAI = 0
    startTime = 0
    speed = 0
    position = [0.0, 0.0, 0.0]
    def getAIG(self):
        return [self.currentGen, self.currentAI]
    def pressKeys(self, keys: list[str]):
        for key in keys:
            try:
                self.nextUpKeys.remove(key)
                keys.remove(key)
            except:
                pass
        for key in self.nextUpKeys:
            pydirectinput.keyUp(key)
        self.nextUpKeys.clear()
        for key in keys:
            pydirectinput.keyDown(key)
        self.nextUpKeys = keys.copy()
    def run(self, ai: NeuralNetwork, current: str):
        possibleEnd = False
        endTicks = 0
        ai.train(0.25)
        runCompleted = False
        score: float = 0.0
        lastSpeed = 0
        lastZ = getInputs()[1][0][2]
        self.startTime = time.time()
        end_time = self.startTime + 20
        while not runCompleted and time.time() < end_time and not possibleEnd:
            inputs, gameData = getInputs()
            self.speed = inputs[15]
            self.position: list[float] = gameData[0]
            if gameData[1]:
                score += 1000000
                runCompleted = True
                continue
            if self.speed.__round__() == 0:
                if endTicks <= 10:
                    endTicks += 1
                else:
                    possibleEnd = True
                    score -= 1000000
            else:
                endTicks = 0
            if self.position[1] < 10:
                score -= 1000000
                possibleEnd = True

            if self.position[0] > 770 or self.position[0] < 643:
                #score += ((self.position[2] - lastZ)*2)-abs(self.position[2] - lastZ)
                if (self.position[2] - lastZ)/abs(self.position[2] - lastZ) == 1:
                    score += self.position[2] - lastZ
                else:
                    score += (self.position[2] - lastZ)*3
            else:
                #score += ((lastZ - self.position[2])*2)-abs(self.position[2] - lastZ)
                if (lastZ - self.position[2])/abs(lastZ - self.position[2]) == 1:
                    score += lastZ - self.position[2]
                else:
                    score += (lastZ - self.position[2])*3
            lastZ = self.position[2]

            if self.speed > lastSpeed:
                score += (self.speed - lastSpeed)
            output = ai.forward(inputs)
            keys = []
            if output[0] == 1: 
                keys.append('w')
            if output[1] == 1: 
                keys.append('s')
            if output[2] == 1: 
                keys.append('a')
            if output[3] == 1: 
                keys.append('d')

            if not keys == []: self.pressKeys(keys)
        pydirectinput.press(['r', 'up', 'up', 'down'])
        time.sleep(0.1)
        pydirectinput.typewrite(current, 0.1)
        pydirectinput.press(['down', 'enter', 'enter', 'del'])
        return score
    def genTrain(self, n_layers: int=None, output_activations: list[Activation]=None, base_activation: type[Activation]=Activation, generations: int=1, ais: int = 1):
        """
        Trains the AI by running a number of random neural networks (the ais input times 100) and calculates each of their accumulative reward. 
        It collects the top 5% neural networks, multiplies equally to match the ais number times 100, then slightly modifies each one. 
        This is repeated a number of times based on the generations input. 
        After the last generation is process, the best ai is returned.
        """
        
        self.isGen = True
        print("Waiting")
        while not getInputs()[1][2]: time.sleep(0.1)
        print("Started!")
        bestNetworks = [NeuralNetwork(16, n_layers, 4, output_activations, base_activation) for _ in range(ais*100)]
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

    def train(self, n_layers: int=None, output_activations: list[Activation]=None, base_activation: type[Activation]=Activation, runs: int=100):
        """Trains the AI by running a single neural network multiple times (the runs input). Deviates slightly if the reward is greater than the last reward. Will try back propagation to make good adjustments it the reward is good."""
        print("Waiting")
        while not getInputs()[1][2]: time.sleep(0.1)
        print("Started!")
        nn = NeuralNetwork(16, n_layers, 4, output_activations, base_activation)
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
    def trainPPO(self, n_layers: int, driver_activation: type[Activation]=Activation, coach_activation: type[Activation]=Activation, episodes: int=100, epotches: int=20):
        """PPO Training for the network."""
        print("Waiting")
        while not getInputs()[1][2]: time.sleep(0.1)
        print("Started!")
        
        #decayFactor = 0.95

        driver = NeuralNetwork(n_layers, 16, 4, driver_activation)
        coach = NeuralNetwork(n_layers, 16, 1, coach_activation)

        for e in range(episodes):
            episode: list[tuple] = []
            nextInput, gameData = getInputs()
            self.startTime = time.time()
            end_time = self.startTime + 20
            lastSpeed = 0
            lastZ = gameData[0][2]
            end = False
            while time.time() < end_time and not end:
                #runTime = time.time() - self.startTime
                score = 0
                state = nextInput
                outputs = np.clip(driver.forward(nextInput)[0], a_min=1e-7, a_max=1-1e-7)

                probabilities = 1 / (1 + np.exp(-outputs))

                action = np.asarray(np.random.binomial(
                    1,
                    probabilities
                ), dtype=np.longdouble)

                log_prob = bernoulli_log_prob(
                    action,
                    outputs
                )

                keys = []
                if action[0]:
                    keys.append('w')
                if action[1]:
                    keys.append('s')
                if action[2]:
                    keys.append('a')
                if action[3]:
                    keys.append('d')

                if not keys == []: self.pressKeys(keys)

                time.sleep(0.01)

                nextInput, gameData = getInputs()
                self.speed = nextInput[15]

                delta_z = gameData[0][2] - lastZ

                if gameData[0][0] > 770 or gameData[0][0] < 643:
                    if delta_z >= 0:
                        score += delta_z
                    else:
                        score += delta_z * 3
                else:
                    if delta_z <= 0:
                        score -= delta_z
                    else:
                        score -= delta_z * 3
                
                lastZ = gameData[0][2]
                
                if self.speed > lastSpeed:
                    score += (self.speed - lastSpeed)
                lastSpeed = self.speed

                if gameData[0][1] < 10:
                    score -= 1000
                    end = True
                #score += nextInput[15]
                #score *= decayFactor**(((runTime*100).__round__())/100)

                episode.append((state, action, score, log_prob))
            pydirectinput.press(['r', 'up', 'up', 'down'])
            time.sleep(0.1)
            pydirectinput.typewrite(f"Ep{e}", 0.1)
            pydirectinput.press(['down', 'enter', 'enter'])
            returns = np.zeros(len(episode), dtype=np.longdouble)

            G = np.longdouble(0.0)
            epsilon = 0.2

            for i in reversed(range(len(episode))):
                G = episode[i][2] + np.longdouble(0.99) * G
                returns[i] = G
            values = np.zeros(len(episode), dtype=np.longdouble)

            for i, frame in enumerate(episode):
                values[i] = coach.forward(frame[0])[0].item()

            advantages = np.array(returns - values)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for _ in range(epotches):
                for i, frame in enumerate(episode):
                    coach.backward(2 * (coach.forward(frame[0])[0] - returns[i]))
                    
                    state, action, _, old_log_prob = episode[i]
                    outputs = np.clip(driver.forward(state)[0], a_min=-709-1e-7, a_max=709-1e-7)
                    probabilities = 1 / (1 + np.exp(-outputs))
                    new_log_prob = bernoulli_log_prob(action, outputs)
                    ratio = np.exp(new_log_prob - old_log_prob)
                    advantage = advantages[i]

                    #something = np.minimum(ratio * advantage,np.clip(ratio, 1 - epsilon, 1 + epsilon, dtype=np.longdouble) * advantage)

                    unclipped = ratio * advantage
                    clipped = np.clip(ratio, 1 - epsilon, 1 + epsilon) * advantage

                    surrogate_gradient = np.where(
                        unclipped <= clipped,
                        unclipped,
                        0
                    )

                    actor_gradient = (
                        -surrogate_gradient
                        * (action - probabilities)
                    )
                
                    driver.backward(actor_gradient, 1e-2)

                
            pydirectinput.press('del')
        return driver

#AI stuffs
def bernoulli_log_prob(action, logits):
    return np.sum(
        action * -np.logaddexp(0, -logits)
        + (1 - action) * -np.logaddexp(0, logits)
    )

# error(x)
# |  \      ____
# |   \    /    \
# |    \__/      |
# |               \_/
# ------------------- x
#        ^         ^
# local minimum   global minimum
# minimums are points of low error
# there are two types; local (smaller minimums that solve the immediate problems)
# and global (solutions to everything)
# the gradient points DOWN the slop of error(x) always
# so if i start at x = 0, and i go down, i end up in the local minimum forever
# if i start at the middle peak, i go down to the global minimum
# everything we do in machine learning is to help the AI find the global loss (or error) minimum
# easiest way: to kick us out of the minimum;: increase the learning rate, the rate at which we descent d the slop
# we want the ai to overshoot itself out of the minimum
