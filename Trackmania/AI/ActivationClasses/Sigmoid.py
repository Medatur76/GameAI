from AI.ActivationClasses.Activation import Activation
from math import e
from numpy import ndarray, array
class SigmoidActivation(Activation):
    @staticmethod
    def forward(inputs):
        if not isinstance(inputs, list) and not isinstance(inputs, ndarray):
            return 1/(1+e**(-SigmoidActivation.clamp(inputs)))
        else:
            return array([SigmoidActivation.forward(i) for i in inputs])
    @staticmethod
    def toString():
        return "Sigmoid"
    @staticmethod
    def derivative(inputs):
        if not isinstance(inputs, list) and not isinstance(inputs, ndarray):
            return SigmoidActivation.forward(inputs)*(1-SigmoidActivation.forward(inputs))
        else:
            return array([SigmoidActivation.derivative(i) for i in inputs])