from AI.ActivationClasses.Activation import Activation
from numpy import ndarray, zeros_like, where

class BinaryStepActivation(Activation):
    @staticmethod
    def forward(inputs):
        if not isinstance(inputs, list) and not isinstance(inputs, ndarray):
            return where(inputs > 0, 1, 0)
        else:
            outputs: list[int] = []
            for i in inputs:
                outputs.append(where(i > 0, 1, 0))
            return outputs
    @staticmethod
    def toString():
        return "BinaryStep"
    @staticmethod
    def derivative(input):
        return zeros_like(input)
