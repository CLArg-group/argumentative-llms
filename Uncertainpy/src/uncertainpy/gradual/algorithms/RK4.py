from .Approximator import Approximator


class RK4(Approximator):
    def __init__(self, ads, time=0, arguments=[], argument_strength=[], attacker=[], supporter=[], name="") -> None:
        super().__init__(ads, time, arguments, argument_strength, attacker, supporter)
        self.name = "RK4"

    def perform_iteration(self, delta, epsilon):

        k1 = self.ads.compute_derivative_at(self.ads.argument_strength)

        y2 = []
        for i in range(len(self.ads.arguments)):
            y2.append(self.ads.argument_strength[i] + delta / 2 * k1[i])

        k2 = self.ads.compute_derivative_at(y2)

        y3 = []
        for i in range(len(self.ads.arguments)):
            y3.append(self.ads.argument_strength[i] + delta / 2 * k2[i])

        k3 = self.ads.compute_derivative_at(y3)

        y4 = []
        for i in range(len(self.ads.arguments)):
            y4.append(self.ads.argument_strength[i] + delta * k3[i])

        k4 = self.ads.compute_derivative_at(y4)

        max_derivative = 0

        for i in range(len(self.ads.arguments)):
            d = (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6
            abs_derivative = abs(d)

            if abs_derivative > max_derivative:
                max_derivative = abs_derivative

            self.ads.argument_strength[i] += delta * d

        return max_derivative

    def __str__(self) -> str:
        return __class__.__name__
