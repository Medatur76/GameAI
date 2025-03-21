from AI.ActivationClasses.Activation import Activation
from math import e
from numpy import ndarray, tanh, array

class HyperTangActivation(Activation):
    @staticmethod
    def forward(inputs):
        #if isinstance(inputs, list) or isinstance(inputs, ndarray):
        #    return [HyperTangActivation.forward(x) for x in inputs]
        #else:
        #    inputs = HyperTangActivation.clamp(inputs)
        #    return ((math.e**inputs) - (math.e**(-inputs)))/((math.e**inputs)+(math.e**(-inputs)))
        return tanh(inputs)
    @staticmethod
    def toString() -> str:
        return "Hyperbolic"
    @staticmethod
    def derivative(inputs):
        if not isinstance(inputs, list) and not isinstance(inputs, ndarray):
            return 1-(HyperTangActivation.forward(inputs)**2)
        else:
            return array([HyperTangActivation.derivative(i) for i in inputs])
