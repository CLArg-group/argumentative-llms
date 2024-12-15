import math

class EulerBasedInfluence:
    def __init__(self) -> None:
        pass

    def compute_strength(self, weight, aggregate):
        return 1 - (1-weight**2) / (1 + weight * math.exp(aggregate))

    def __str__(self) -> str:
        return __class__.__name__