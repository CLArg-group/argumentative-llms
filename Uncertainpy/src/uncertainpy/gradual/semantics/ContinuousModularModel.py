from .Model import Model


class ContinuousModularModel(Model):
    def __init__(self, aggregation=None, influence=None, BAG=None, approximator=None, arguments=..., argument_strength=..., attacker=..., supporter=..., name="") -> None:
        super().__init__(BAG, approximator, aggregation, influence, arguments, argument_strength, attacker, supporter, name)
        self.name = __class__.__name__

    def compute_derivative_at(self, state):
        derivatives = []

        for i in range(len(self.arguments)):
            aggregate_strength = self.aggregation.aggregate_strength(self.attacker[i], self.supporter[i], state)
            derivative = self.influence.compute_strength(self.arguments[i].initial_weight, aggregate_strength)
            derivative -= state[i]

            derivatives.append(derivative)

        return derivatives

    def __repr__(self) -> str:
        return super().__repr__(__name__)

    def __str__(self) -> str:
        return super().__str__(__name__)
