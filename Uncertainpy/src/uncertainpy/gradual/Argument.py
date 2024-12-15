class Argument:
    def __init__(self, name, arg, initial_weight, strength=None, attackers=None, supporters=None):
        self.name = name
        self.arg = arg
        self.initial_weight = initial_weight
        self.strength = strength
        self.attackers = attackers
        self.supporters = supporters
        self.parent = None

        if type(initial_weight) != int and type(initial_weight) != float:
            raise TypeError("initial_weight must be of type integer or float")

        if strength is None:
            self.strength = initial_weight

        if attackers is None:
            self.attackers = []

        if supporters is None:
            self.supporters = []

    def get_name(self):
        return self.name

    def get_arg(self):
        return self.arg

    def add_attacker(self, attacker):
        self.attackers.append(attacker)

    def add_supporter(self, supporter):
        self.supporters.append(supporter)

    def add_parent(self, parent):
        self.parent = parent

    def get_initial_weight(self):
        return self.initial_weight

    def reset_initial_weight(self, weight):
        self.initial_weight = weight

    def __repr__(self) -> str:
        return (f"Argument: {self.arg}, initial weight: {self.initial_weight}, strength: {self.strength}, attackers:"
                f"{self.attackers}, supporters: {self.supporters}")

    def __str__(self) -> str:
        return (f"Argument: {self.arg}, initial weight: {self.initial_weight}, strength: {self.strength}, attackers:"
                f"{self.attackers}, supporters: {self.supporters}")
        # return f"Argument(name={self.name}, weight={self.initial_weight}, strength={self.strength})"

    def _to_shallow_dict(self):
        return {
            'name': self.name,
            'argument': self.arg,
            'initial_weight': self.initial_weight,
            'strength': self.strength,
        }

    @classmethod
    def _from_shallow_dict(cls, d):
        return cls(
            d['name'],
            d['argument'],
            d['initial_weight'],
            strength=d['strength'],
        )
