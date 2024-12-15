from uncertainpy.propositional.semantics import BooleanInterpretation
from uncertainpy.propositional.syntax import Formula
from uncertainpy.propositional.syntax import Conditional

class ProbabilityDist:

    def __init__(self, interp: BooleanInterpretation, probs):
        self.interp = interp
        self.probs = probs
    
    def __str__(self):
        str = ""
        for i in range(len(self.probs)):
            str = str + f"{self.interp.int_to_map(i)}: {self.probs[i]}\n"
        return str
    
    def computeProb(self, f):
        
        prob = 0
        if isinstance(f, Formula):
            for i in range(len(self.probs)):
                if self.interp.satisfies(i, f):
                    prob += self.probs[i]
            return prob
        
        if isinstance(f, Conditional):
            num = 0
            denomin = 0
            for i in range(len(self.probs)):
                if (f.a==None or self.interp.satisfies(i, f.a)):
                    denomin += self.probs[i]
                    if self.interp.satisfies(i, f.b):
                        num += self.probs[i]
            
            if denomin>0:
                return num/denomin
            else:
                return 'Undefined (condition has probability 0)'  
            
        return None