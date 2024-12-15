class QuadraticMaximumInfluence:
    def __init__(self, conservativeness) -> None:
        self.conservativeness = conservativeness

    def compute_strength(self, weight, aggregate):
        strength = weight

        scaled_aggregate = aggregate / self.conservativeness
        h = scaled_aggregate**2 / (1 + scaled_aggregate**2)

        if (aggregate > 0):
            strength += h * (1 - weight)
        else:
            strength -= h * weight

        return strength

    def __str__(self) -> str:
        return __class__.__name__ + f"({self.conservativeness})"
