from uncertainpy.propositional.syntax import Formula
  

class BooleanInterpretation:
    """Manages interpretations for languages over boolean atoms. Interpretations are represented by
    integers for memory efficiency (too many atoms may result in overflow)"""
    
    def __init__(self, atoms):
        """Expects a sequence of atoms that is to be interpreted"""
        self.atoms = atoms
        
    def satisfies(self, i: int, f: Formula):
        """Tests if interpretation i represented by an integer satisfies formula f"""
        truthmap = self.int_to_map(i)
        return f.satisfied_by(truthmap)
        
    def noWorlds(self):
        return pow(2,len(self.atoms))
        
        
    def int_to_map(self, i: int):
        """Translates integer representation of interpretation to mapping representation"""
        truthmap = {}
        for a in self.atoms:
            v = i%2
            truthmap[a] = (v==1)
            i = (i-v)/2
            
        return truthmap