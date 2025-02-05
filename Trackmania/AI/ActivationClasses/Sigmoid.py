from ActivationClasses.Activation import Activation
import math

class SigmoidActivation(Activation):
    def forward(inputs):
        if not isinstance(inputs, list):
            return 1/(1+math.e**inputs)
        if len(inputs) == 1:
            return 1/(1+math.e**inputs[0])
        else:
            return [1/(1+math.e**i) for i in inputs]
        
    def toString():
        return "Sigmoid"