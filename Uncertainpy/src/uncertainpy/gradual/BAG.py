import os
import re
import string
from .Argument import Argument
from .Support import Support
from .Attack import Attack


class BAG:

    def __init__(self, path=None):
    
        self.arguments = {}
        self.attacks = []
        self.supports = []
        
        self.path = path

        if (path is None):
            pass
        else:
            with open(os.path.abspath(path), "r") as f:
                for line in f.readlines():
                    k_name = line.split("(")[0]
                    if k_name in string.whitespace:
                        pass
                    else:
                        k_args = re.findall(rf"{k_name}\((.*?)\)", line)[0].replace(" ", "").split(",")
                        if k_name == "arg":
                            argument = Argument(k_args[0], float(k_args[1]), None, [], [])
                            self.arguments[argument.name] = argument

                        elif k_name == "att":
                            attacker = self.arguments[k_args[0]]
                            attacked = self.arguments[k_args[1]]
                            self.add_attack(attacker, attacked)

                        elif k_name == "sup":
                            supporter = self.arguments[k_args[0]]
                            supported = self.arguments[k_args[1]]
                            self.add_support(supporter, supported)

    def add_attack(self, attacker, attacked):
        if type(attacker) != Argument:
            raise TypeError("attacker must be of type Argument")

        if type(attacked) != Argument:
            raise TypeError("attacked must be of type Argument")

        if attacker.name in self.arguments:
            attacker = self.arguments[attacker.name]
        else:
            self.arguments[attacker.name] = attacker

        if attacked.name in self.arguments:
            attacked = self.arguments[attacked.name]
        else:
            self.arguments[attacked.name] = attacked
            
        attacked.add_attacker(attacker)

        self.attacks.append(Attack(attacker, attacked))

    def add_support(self, supporter, supported):
        if type(supporter) != Argument:
            raise TypeError("supporter must be of type Argument")

        if type(supported) != Argument:
            raise TypeError("supported must be of type Argument")

        if supporter.name in self.arguments:
            supporter = self.arguments[supporter.name]
        else:
            self.arguments[supporter.name] = supporter

        if supported.name in self.arguments:
            supported = self.arguments[supported.name]
        else:
            self.arguments[supported.name] = supported

        supported.add_supporter(supporter)

        self.supports.append(Support(supporter, supported))

    def reset_strength_values(self):
        for a in list(self.arguments.values()):
            a.strength = a.initial_weight

    def get_arguments(self):
        return list(self.arguments.values())

    def __str__(self) -> str:
        return f"BAG set to read from {self.path} with arguments: {self.arguments}, attacks: {self.attacks} and supports: {self.supports}"

    def __repr__(self) -> str:
        return f"BAG({self.path}) Arguments: {self.arguments} Attacks: {self.attacks} Supports: {self.supports}"

    def to_dict(self):
        return {
            'arguments': {
                n: a._to_shallow_dict()
                for n, a in self.arguments.items()
            },
            'attacks': [
                [a.attacker.name, a.attacked.name]
                for a in self.attacks
            ],
            'supports': [
                [s.supporter.name, s.supported.name]
                for s in self.supports
            ],
        }

    @classmethod
    def from_dict(cls, d):
        bag = cls()
        arguments = {
            n: Argument._from_shallow_dict(a)
            for n, a in d['arguments'].items()
        }
        for attacker_name, attacked_name in d['attacks']:
            bag.add_attack(arguments[attacker_name], arguments[attacked_name])
        for supporter_name, supported_name in d['supports']:
            bag.add_support(arguments[supporter_name], arguments[supported_name])
        return bag
