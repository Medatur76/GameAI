from ActivationClasses.Activation import Activation

class BinaryStepActivation(Activation):
    def forward(self, inputs):
        if len(inputs) == 1:
            output = 0
            if i > 0: output = 1
            return output
        else:
            outputs: list[int] = []
            for i in inputs:
                output = 0
                if i > 0: output = 1
                outputs.append(output)
            return outputs