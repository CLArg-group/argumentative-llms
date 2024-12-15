
def computeStrengthValues(bag, agg_f, inf_f):
    """
    Computes strength values in acyclic BAGs using a topological ordering and forward propagation
    of the base scores. If the graph contains cycles, None will be returned
    """
    
    order = computeTopOrder(bag)
    if order == None:
        return None
    
    strength = {arg:arg.initial_weight for arg in order}
    
    for arg in order:
        agg = agg_f.aggregate_strength(arg.attackers, arg.supporters, strength)
        s = inf_f.compute_strength(arg.initial_weight, agg)
        
        arg.strength = s
        strength[arg] = s
        
    return strength
    
    

    
def computeTopOrder(bag):
    """
    Compute topological order for given bag or return None if bag is cyclic.
    """

    args = bag.arguments.values()

    #compute topological order
    indeg = {arg:0 for arg in args}

    #store indegree and parents
    attacks = {arg:[] for arg in args}
    supports = {arg:[] for arg in args}

    for att in bag.attacks:
        indeg[att.get_attacked()] += 1
        attacks[att.get_attacker()].append(att.get_attacked())
    for sup in bag.supports:
        indeg[sup.get_supported()] += 1
        supports[sup.get_supporter()].append(sup.get_supported())

    #determine source arguments
    source_args = []
    for arg in args:
        if indeg[arg] == 0:
            source_args.append(arg)

    #build up order
    order = []

    while(len(source_args) > 0):

        arg = source_args.pop(0)
        order.append(arg)

        #update children
        for c in attacks[arg]:
            indeg[c] -= 1
            if indeg[c]==0:
                source_args.append(c)

        for c in supports[arg]:
            indeg[c] -= 1
            if indeg[c]==0:
                source_args.append(c)

    #if node is missing in order, the bag must be cyclic
    if len(order) != len(args):   
        print(f"Graph contains cycles. Found partial topological order {[arg.name for arg in order]}.")
        return None
          
    return order
    
    


