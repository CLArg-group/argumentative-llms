class Model:
    def __init__(self, BAG=None, approximator=None, aggregation=None, influence=None, arguments=[], argument_strength=[], attacker=[], supporter=[], name="") -> None:
        self.BAG = BAG
        self.approximator = approximator
        self.aggregation = aggregation
        self.influence = influence
        self.arguments = arguments
        self.argument_strength = argument_strength
        self.attacker = attacker
        self.supporter = supporter
        self.name = name

    def solve(self, delta, epsilon, verbose=True, generate_plot=False):
        if type(verbose) != bool:
            raise TypeError("verbose must be a boolean")

        if type(generate_plot) != bool:
            raise TypeError("generate_plot must be a boolean")

        if (type(delta) != float and type(delta) != int):
            raise TypeError("delta must be a float or integer")

        if (type(epsilon) != float and type(epsilon) != int):
            raise TypeError("epsilon must be a float or integer")

        if self.approximator is None:
            raise AttributeError("Model does not have approximator attached")

        if self.BAG is None:
            raise AttributeError("Model does not have BAG attached")

        result = self.approximator.approximate_solution(delta, epsilon, verbose, generate_plot)
        return result

    def __repr__(self, name) -> str:
        return f"{name}({self.BAG}, {self.approximator}, {self.arguments}, {self.argument_strength}, {self.attacker}, {self.supporter})"

    def __str__(self, name) -> str:
        return f"{name} - BAG: {self.BAG}, Approximator: {self.approximator}, Arguments: {self.arguments}, Argument strength: {self.argument_strength}, Attacker: {self.attacker}, Supporter: {self.supporter})"
