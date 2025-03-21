from numpy import ndarray, array

class Activation():
    @staticmethod
    def forward(inputs):
        if not isinstance(inputs, list) and not isinstance(inputs, ndarray):
            return Activation.clamp(inputs)
        else: return array([Activation.forward(a) for a in inputs])
    @staticmethod
    def toString() -> str:
        return "Activation"
    @staticmethod
    def derivative(_):
        return 1
    @staticmethod
    def clamp(num):
        if abs(num) > 2**5-10: return (2**5-10)*(abs(num)/num)
        else: return round(num, 6)