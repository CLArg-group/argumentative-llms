class Attack:
    def __init__(self, attacker, attacked) -> None:
        self.attacker = attacker
        self.attacked = attacked

    def get_attacker(self):
        return self.attacker

    def get_attacked(self):
        return self.attacked

    def __repr__(self) -> str:
        return f"Attack({self.attacker}, {self.attacked})"

    def __str__(self) -> str:
        return f"Attack by {self.attacker} to {self.attacked}"
