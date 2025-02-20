from AI.ActivationClasses.Activation import Activation
import math

class HyperTangActivation(Activation):
    @staticmethod
    def forward(inputs):
        if not isinstance(inputs, list):
            return ((math.e**inputs) - (math.e**(-inputs)))/((math.e**inputs)+(math.e**(-inputs)))
        else:
            return [((math.e**x) - (math.e**(-x)))/((math.e**x)+(math.e**(-x))) for x in inputs]
    @staticmethod
    def toString() -> str:
        return "Hyperbolic"
    @staticmethod
    def derivative(inputs):
        if not isinstance(inputs, list):
            return 1-(inputs**2)
        else:
            return [1-(i**2) for i in inputs]