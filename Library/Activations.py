import numpy as np
import math

class Activation:
    @staticmethod
    def forward(x):
        return x
    @staticmethod
    def derivative(_):
        return 1
class Softplus(Activation):
    @staticmethod
    def forward(x):
        return np.log(np.add(1, np.pow(math.e, x)))
    @staticmethod
    def derivative(x):
        return np.divide(1, np.add(1, np.pow(math.e, np.multiply(-1, x))))
class Sigmoid(Activation):
    @staticmethod
    def forward(x):
        return np.divide(1, np.add(1, np.pow(math.e, np.multiply(-1, x))))
    @staticmethod
    def derivative(x):
        sig = Sigmoid.forward(x)
        return np.multiply(sig, np.subtract(1, sig))
class HyperbolicTangent(Activation):
    @staticmethod
    def forward(x):
        return np.divide(np.subtract(np.pow(math.e,x),np.pow(math.e,np.negative(x))), np.add(np.pow(math.e,x), np.pow(math.e,np.negative(x))))
    @staticmethod
    def derivative(x):
        return np.subtract(1, np.pow(HyperbolicTangent.forward(x), 2))
#Might work but im not sure
class BinaryStep(Activation):
    @staticmethod
    def forward(x):
        return np.where(x < 0, 0, 1)
    @staticmethod
    def derivative(x):
        return np.zeros_like(x)