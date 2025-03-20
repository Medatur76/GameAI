from AI.ActivationClasses.Activation import Activation
import math
from numpy import ndarray

class HyperTangActivation(Activation):
    @staticmethod
    def forward(inputs):
        if isinstance(inputs, list) or isinstance(inputs, ndarray):
            return [HyperTangActivation.forward(x) for x in inputs]
        else:
            if abs(inputs) >= 6.5:
                return 1 * (inputs/abs(inputs))
            else:
                return ((math.e**inputs) - (math.e**(-inputs)))/((math.e**inputs)+(math.e**(-inputs)))
    @staticmethod
    def toString() -> str:
        return "Hyperbolic"
    @staticmethod
    def derivative(inputs):
        if not isinstance(inputs, list) and not isinstance(inputs, ndarray):
            return 1-(HyperTangActivation.forward(inputs)**2)
        else:
            return [HyperTangActivation.derivative(i) for i in inputs]
