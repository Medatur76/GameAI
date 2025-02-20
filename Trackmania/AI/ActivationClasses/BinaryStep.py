from AI.ActivationClasses.Activation import Activation
import numpy as np

class BinaryStepActivation(Activation):
    @staticmethod
    def forward(inputs):
        if not isinstance(inputs, list):
            output = 0
            if inputs > 0: output = 1
            return output
        else:
            outputs: list[int] = []
            for i in inputs:
                output = 0
                if i > 0: output = 1
                outputs.append(output)
            return outputs
    @staticmethod
    def toString():
        return "BinaryStep"
    @staticmethod
    def derivative(_):
        return 0