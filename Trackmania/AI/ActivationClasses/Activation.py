class Activation():
    @staticmethod
    def forward(inputs):
        return inputs
    @staticmethod
    def toString() -> str:
        return "Activation"
    @staticmethod
    def derivative(_):
        return 1