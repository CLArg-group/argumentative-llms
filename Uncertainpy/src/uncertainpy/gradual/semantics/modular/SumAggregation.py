class SumAggregation:
    def __init__(self) -> None:
        pass

    def aggregate_strength(self, attackers, supporters, state):
        aggregate = 0
        for a in attackers:
            aggregate -= state[a]

        for s in supporters:
            aggregate += state[s]

        return aggregate
    
    def __str__(self) -> str:
        return __class__.__name__
