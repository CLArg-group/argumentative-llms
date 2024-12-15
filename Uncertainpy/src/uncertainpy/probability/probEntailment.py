import scipy.sparse as sp
import numpy as np
import gurobipy as gp

from gurobipy import GRB
from uncertainpy.probability.distribution import ProbabilityDist


class ProbEntailmentEngine:

    def __init__(self):
        self.model = None
        self.ints = None
        self.x = None
        self.noVars = 0
        self.noWorlds = 0

    def createConstraints(self, kb, ints):
        """
        Takes a knowledge base kb and a BooleanInterpretation object ints
        to build up the constraints for probabilistic entailemt 
        """

        self.ints = ints
        self.model = gp.Model("Probabilistic Entailment")

        # partition conditionals based on their nature
        condsl = []  # only lower bound
        condsu = []  # only upper bound
        condsp = []  # point probabilities
        for cond in kb:
            if cond.l == cond.u:
                if cond.l != None:
                    condsp.append(cond)
            # now we have cond.l != cond.u:
            elif cond.l == None:
                condsu.append(cond)
            elif cond.u == None:
                condsl.append(cond)
            else:
                condsl.append(cond)
                condsu.append(cond)

        # create non-negative variables (probabilities, slack variables, auxiliary variable for conditional queries)
        self.noWorlds = ints.noWorlds()
        self.noVars = self.noWorlds + len(condsl) + len(condsu) + 1
        self.x = self.model.addMVar(shape=self.noVars, name="Vars")

        # create conditional constraints
        A = sp.dok_matrix((len(condsp) + len(condsl) + len(condsu), self.noVars), dtype=np.float32)

        # create point probability constraints
        rows = 0
        for cond in condsp:
            print(f"Create point constraint for {cond}")
            for i in range(self.noWorlds):
                if cond.a == None or ints.satisfies(i, cond.a):
                    if ints.satisfies(i, cond.b):
                        A[rows, i] = 1 - cond.l
                    else:
                        A[rows, i] = -cond.l

            rows += 1

        # create lower constraints
        slack_i = self.noWorlds
        for cond in condsl:
            print(f"Create lower constraint for {cond}")
            for i in range(self.noWorlds):
                if cond.a == None or ints.satisfies(i, cond.a):
                    if ints.satisfies(i, cond.b):
                        A[rows, i] = 1 - cond.l
                    else:
                        A[rows, i] = -cond.l

            A[rows, slack_i] = -1
            rows += 1
            slack_i += 1

        # create upper constraints
        for cond in condsu:
            print(f"Create upper constraint for {cond}")
            for i in range(self.noWorlds):
                if cond.a == None or ints.satisfies(i, cond.a):
                    if ints.satisfies(i, cond.b):
                        A[rows, i] = 1 - cond.u
                    else:
                        A[rows, i] = -cond.u
            A[rows, slack_i] = 1

            rows += 1
            slack_i += 1

        if(rows > 0):
            rhs = np.zeros(rows)
            self.model.addConstr(A @ self.x == rhs, name="Conditionals")

        # create fractional normalization constraint
        A = np.ones(self.noVars, dtype=np.float32)
        A[-1] = -1
        A[self.noWorlds:-1] = 0
        self.model.addConstr(A @ self.x == 0, name="Fractional normalization")
        #print(f"Fractional normalization vector: {A}")

    def computeBounds(self, query):
        """
        Takes Conditional query and initializes query.l and query.u with
        lower and upper bounds computed by solving the probabilistic entailment problem.
        """

        # create query normalization constraint
        A = sp.dok_matrix((1, self.noVars), dtype=np.float32)
        for i in range(self.noWorlds):
            if query.a == None or self.ints.satisfies(i, query.a):
                A[0, i] = 1
        self.model.addConstr(A @ self.x == 1, name="Query normalization")
        #print(f"Query normalization vector: {A.toarray()}")

        # create query objective
        A = sp.dok_matrix((1, self.noVars), dtype=np.float32)
        for i in range(self.noWorlds):
            if (query.a == None or self.ints.satisfies(i, query.a)) and self.ints.satisfies(i, query.b):
                A[0, i] = 1
        #print(f"Query vector: {A.toarray()}")

        # compute lower and upper bound
        self.model.setObjective(A @ self.x, GRB.MINIMIZE)
        self.model.optimize()
        query.l = self.model.objVal
        #pl = ProbabilityDist(self.ints, self.x.X[:self.noWorlds]/self.x.X[-1])

        self.model.setObjective(A @ self.x, GRB.MAXIMIZE)
        self.model.optimize()
        query.u = self.model.objVal
        print(f"kb |= {query}")
        #pu = ProbabilityDist(self.ints, self.x.X[:self.noWorlds]/self.x.X[-1])

        return query
