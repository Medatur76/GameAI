from AI.ActivationClasses.Activation import Activation
import math
from numpy import ndarray
class SigmoidActivation(Activation):
    @staticmethod
    def forward(inputs):
        if not isinstance(inputs, list) and not isinstance(inputs, ndarray):
            return 1/(1+math.e**(-inputs))
        else:
            return [1/(1+math.e**(-i)) for i in inputs]
    @staticmethod
    def toString():
        return "Sigmoid"
    @staticmethod
    def derivative(inputs):
        if not isinstance(inputs, list):
            return inputs*(1-inputs)
        else:
            return [i*(1-i) for i in inputs]