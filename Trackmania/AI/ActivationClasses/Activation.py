from ActivationClasses.BinaryStep import BinaryStepActivation
from ActivationClasses.HyperbolicTanget import HyperTangActivation
from ActivationClasses.Sigmoid import SigmoidActivation

class Activation():
    @staticmethod
    def forward(inputs):
        return inputs
    @staticmethod
    def toString() -> str:
        return "Activation"