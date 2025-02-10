from ActivationClasses.Activation import Activation
import math

class SigmoidActivation(Activation):
    @staticmethod
    def forward(inputs):
        if not isinstance(inputs, list):
            return 1/(1+math.e**(-inputs))
        else:
            return [1/(1+math.e**(-i)) for i in inputs]
        
    @staticmethod
    def toString():
        return "Sigmoid"