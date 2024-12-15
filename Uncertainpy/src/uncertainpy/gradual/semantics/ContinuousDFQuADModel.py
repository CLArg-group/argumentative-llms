from .Model import Model


class ContinuousDFQuADModel(Model):
    def __init__(self, aggregation=None, influence=None, BAG=None, approximator=None, arguments=..., argument_strength=..., attacker=..., supporter=..., name="") -> None:
        super().__init__(BAG, approximator, aggregation, influence, arguments, argument_strength, attacker, supporter, name)
        self.name = __class__.__name__

    def compute_derivative_at(self, state):
        derivatives = []

        for i in range(len(self.arguments)):

            support_energy = 1
            for a in self.attacker[i]:
                support_energy *= (1 - state[a])

            attack_energy = 1
            for s in self.supporter[i]:
                attack_energy *= (1 - state[s])

            geometric_energy = support_energy - attack_energy
            weight = self.arguments[i].initial_weight
            derivative = weight
            if geometric_energy > 0:
                derivative += (1 - weight) * geometric_energy

            elif geometric_energy < 0:
                derivative += weight * geometric_energy

            derivative -= state[i]
            derivatives.append(derivative)

        return derivatives

    def __repr__(self) -> str:
        return super().__repr__(__name__)

    def __str__(self) -> str:
        return super().__str__(__name__)
