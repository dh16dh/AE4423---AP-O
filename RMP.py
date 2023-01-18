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
        self.P = self.parameter_set.P  # set of all passenger itineraries (paths)
        self.Gk = self.parameter_set.G  # set of ground arcs  [k]
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
        self.Q = self.parameter_set.Q  # daily unconstrained demand on flight (leg) i   [i]

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
                f[i, k] = model.addVar(obj=self.cost[k][i], vtype=GRB.BINARY)

        for p in self.P:
            for r in self.P:
                if p == r:
                    continue
                t[p, r] = model.addVar(obj=self.fare[p] - (self.b.loc[p, r] * self.fare[r]), vtype=GRB.INTEGER)

        model.update()
        model.setObjective(model.getObjective(), GRB.MINIMIZE)

        # Define Constraints
        for i in self.L:
            model.addConstr(quicksum(f[i, k] for k in self.K) == 1, name='C1')

        for k in self.K:
            for n in self.N[k]:
                model.addConstr(
                    y[self.n_plus[k, n], k] + quicksum(f[i, k] for i in self.O[k, n]) - y[self.n_min[k, n], k] - quicksum(
                        f[i, k] for i in self.I[k, n]) == 0, name='C2')

        for t in self.TC:
            for k in self.K:
                print(self.NG[k, t])
                model.addConstr(quicksum(y[a, k] + f[a, k] for a in self.NG[k, t]) <= self.ac[k], name='C3')

        for i in self.L:
            model.addConstr(quicksum(self.s[k] * f[i, k] for k in self.K) +
                            quicksum(quicksum(delta[i, p] * t[p, r] for p in self.P) for r in self.P) -
                            quicksum(quicksum(delta[i, p] * b[r, p] * t[r, p] for p in self.P) for r in self.P) > Q[i],
                            name='C4')

        for p in self.P:
            model.addConstr(quicksum(t[p, r] for r in self.P) <= self.D[p], name='C5')

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
