class LinearInfluence:
    def __init__(self, conservativeness) -> None:
        self.conservativeness = conservativeness

    def compute_strength(self, weight, aggregate):
        strength = weight
        if (aggregate > 0):
            strength += aggregate * (1-weight)/self.conservativeness
        else:
            strength += aggregate*weight/self.conservativeness

        return strength

    def __str__(self) -> str:
        return __class__.__name__ + f"({self.conservativeness})"
