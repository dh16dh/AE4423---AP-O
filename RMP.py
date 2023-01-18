# Import relevant modules
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from gurobipy import Model, GRB, LinExpr, quicksum
from Parameters import Parameters


class RMP:
    def __init__(self):
        self.parameter_set = Parameters()

        # Define Sets
        self.N = self.parameter_set.N  # set of nodes  [k]
        self.K = self.parameter_set.K  # set of aircraft types
        self.L = self.parameter_set.F  # set of flights
        self.P = self.parameter_set.Pi  # set of all passenger itineraries (paths)
        self.R = self.parameter_set.Pr
        self.G = self.parameter_set.G  # set of ground arcs  [k]
        self.TC = self.parameter_set.TC  # set of unique time cuts   [k]
        self.NG = self.parameter_set.NG  # set of flight and ground arcs intercepted by the time cut  [k, tc]
        self.O = self.parameter_set.O  # flight arcs originating at node n in fleet k   [k, n]
        self.I = self.parameter_set.I  # flight arcs terminating at node n in fleet k   [k, n]
        self.n_plus = self.parameter_set.n_plus  # ground arcs originating at any node n    n+[k, n]
        self.n_min = self.parameter_set.n_minus  # ground arcs terminating at any node n   n-[k, n]

        self.ac = self.parameter_set.AC  # number of aircraft in fleet of type k   [k]
        self.cost = self.parameter_set.c  # operating cost of AC type k for flight i   .loc[i, k]
        self.s = self.parameter_set.s  # number of seats for aircraft type k   [k]
        self.fare = self.parameter_set.fare  # average fare for itinerary p   [p]

        self.D = self.parameter_set.D  # daily unconstrained demand for itinerary p   [p]
        self.Q = self.parameter_set.Q  # daily unconstrained demand on flight (leg) i   .loc[i]

        self.b = self.parameter_set.b  # recapture rate of a pax that desired itinerary p and is allocated to r   .loc[p, r]
        self.delta = self.parameter_set.delta  # if flight i is in itinerary p [i, p]

    def rmp_model(self):
        # Initialise gurobipy model
        model = Model("RMP")

        # Define Decision Variables
        f = {}  # 1 if flight arc i is assigned to aircraft type k, 0 otherwise [i, k]
        y = {}  # number of aircraft of type k on the ground arc a [a, k]
        t = {}  # number of passengers that would like to travel on itinerary p and are reallocated to itin. r [p, r]

        # Add Variables to Objective Function
        for i in self.L:
            for k in self.K:
                f[i, k] = model.addVar(obj=self.cost.loc[i, k], vtype=GRB.BINARY)
                y[i, k] = model.addVar(ub=0)

        for p in self.P:
                t[p] = model.addVar(obj=self.fare[p[0]] - (self.b.loc[p] * self.fare[p[1]]), vtype=GRB.INTEGER)

        for k in self.K:
            for a in self.G[k]:
                y[a, k] = model.addVar(vtype=GRB.INTEGER)
                f[a, k] = model.addVar(ub=0)

        model.update()
        model.setObjective(model.getObjective(), GRB.MINIMIZE)
        print("Objective Function Defined")

        # Define Constraints
        for i in self.L:
            model.addConstr(quicksum(f[i, k] for k in self.K) == 1, name='C1')
        print("Added Constraint: C1")
        for k in self.K:
            for n in self.N[k]:
                model.addConstr(
                    y[self.n_plus[k, n], k] + quicksum(f[i, k] for i in self.O[k, n]) - y[self.n_min[k, n], k] - quicksum(
                        f[i, k] for i in self.I[k, n]) == 0, name='C2')
        print("Added Constraint: C2")
        for k in self.K:
            for tc in self.TC[k]:
                model.addConstr(quicksum(y[a, k] + f[a, k] for a in self.NG[k, tc]) <= self.ac[k], name='C3')
        print("Added Constraint: C3")
        for i in self.L:
            model.addConstr(quicksum(int(self.s[k]) * f[i, k] for k in self.K) +
                            quicksum(self.delta[i, p[0]] * t[p] for p in self.P) -
                            quicksum(self.delta[i, p[0]] * float(self.b.loc[p]) * t[p] for p in self.P) >= int(self.Q.loc[i]),
                            name='C4')
        print("Added Constraint: C4")
        for p in list(self.R.keys()):
            model.addConstr(quicksum(t[p, r] for r in self.R[p]) <= self.D[p], name='C5')
        print("Added Constraint: C5")

        model.write('model.lp')

        model.update()

        model.optimize()
        status = model.status

        if status == GRB.Status.UNBOUNDED:
            print('The model cannot be solved because it is unbounded')

        elif status == GRB.Status.OPTIMAL or True:
            f_objective = model.objVal
            print('***** RESULTS ******')
            print('\nObjective Function Value: \t %g' % f_objective)

        elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
            print('Optimization was stopped with status %d' % status)


if __name__ == '__main__':
    relax_model = RMP().rmp_model()
