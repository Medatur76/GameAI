import math

class HyperTangActivation(Activation):
    def forward(inputs):
        if not isinstance(inputs, list):
            return ((math.e**inputs) - (math.e**(-inputs)))/((math.e**inputs)+(math.e**(-inputs)))
        else:
            return [((math.e**x) - (math.e**(-x)))/((math.e**x)+(math.e**(-x))) for x in inputs]
    def toString() -> str:
        return "Hyperbolic"