from AI.ActivationClasses.Activation import Activation
import math
from numpy import ndarray
class SigmoidActivation(Activation):
    @staticmethod
    def forward(inputs):
        if not isinstance(inputs, list) and not isinstance(inputs, ndarray):
            return 1/(1+math.e**(-round(inputs, 6)))
        else:
            return [SigmoidActivation.forward(i) for i in inputs]
    @staticmethod
    def toString():
        return "Sigmoid"
    @staticmethod
    def derivative(inputs):
        if not isinstance(inputs, list) and not isinstance(inputs, ndarray):
            return SigmoidActivation.forward(inputs)*(1-SigmoidActivation.forward(inputs))
        else:
            return [SigmoidActivation.derivative(i) for i in inputs]