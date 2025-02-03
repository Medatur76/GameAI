from Training.Training import Training
from ActivationClasses.BinaryStep import BinaryStepActivation
from ActivationClasses.Sigmoid import SigmoidActivation

# Inputs:
# 15 - Distance
# 1 - Speed

nn = Training.genTrain(10, [BinaryStepActivation, BinaryStepActivation, SigmoidActivation, SigmoidActivation])