class Formula:
    """Super class for all formulas"""
    
    def satisfied_by(self, interpretation):
        """tests if formula is satisfied by interpretation"""
        return None
    
    def negate(self):
        return Negation(self)
    
    def simplify(self):
        """Try to simplify the formula"""
        pass
    
    
class Atom(Formula):
    """Super class for all atomic formulas, including BooleanAtoms and Variables"""
    pass   


class BooleanAtom(Atom):
    """Boolean Atom"""
    
    def __init__(self, name):
        self.name = name
        
        
    def satisfied_by(self, interpretation):
        """tests if formula is satisfied by interpretation"""
        return interpretation[self]
    
        
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
   

class Variable(Atom):
    pass



class Negation(Formula):
    
    def __init__(self, f):
        self.f = f
        
    def satisfied_by(self, interpretation):
        """tests if formula is satisfied by interpretation"""
        return not self.f.satisfied_by(interpretation)
        
    def __str__(self):
        return "!"+self.f.__str__()
    

class Conjunction(Formula):
    
    def __init__(self, *f):
        self.conjuncts = f
        
    def satisfied_by(self, interpretation):
        """tests if formula is satisfied by interpretation"""
        for c in self.conjuncts:
            if not c.satisfied_by(interpretation):
                return False
        return True 
        
        
    def __str__(self):
        str = "("
        for f in self.conjuncts:
            str = str + f.__str__() +" * "
        return str[:-3]+")"
    
    
class Disjunction(Formula):
    
    def __init__(self, *f):
        self.disjuncts = f
        
    def satisfied_by(self, interpretation):
        """tests if formula is satisfied by interpretation"""
        for d in self.disjuncts:
            if d.satisfied_by(interpretation):
                return True
        return False 
        
    def __str__(self):
        str = "("
        for f in self.disjuncts:
            str = str + f.__str__() +" + "
        return str[:-3]+")"
        
class Conditional:
    """Represents a conditional of the form (b|a)[l,u]"""

    def __init__(self, b, a, l, u):
        self.b = b
        self.a = a
        self.l = l
        self.u = u
        
    def verified_by(self, interpretation):
        """tests if conditional is verified (both a and b are satisfied)"""
        return self.a.satisfied_by(interpretation) and self.b.satisfied_by(interpretation)
        
    def falsified_by(self, interpretation):
        """tests if conditional is falsified (a is satisfied, but b is falsified)"""
        return self.a.satisfied_by(interpretation) and not self.b.satisfied_by(interpretation)
    
    def __str__(self):
        if self.a == None:
            str = f"({self.b})["
        else:
            str = f"({self.b}|{self.a})["
        if self.l == self.u:
            str += f"{self.l}]"
        else:
            str += f"{self.l}, {self.u}]"
            
        return str