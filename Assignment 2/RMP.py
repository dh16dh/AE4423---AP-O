# Import relevant modules
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from gurobipy import Model, GRB, LinExpr, quicksum
import Parameters

class RMP:
    def __init__(self):
        self.parameter_set = Parameters()
        
        # Define Sets
        self.N = 0 # set of airports
        self.K = 0 # set of aircraft types
        self.L = 0 # set of flights
        self.P = 0 # set of all passenger itineraries (paths)
        self.Gk = 0 # set of ground arcs
        self.TC = 0 # set of unique time cuts
        self.NGk = 0 # set of flight and ground arcs intercepted by the time cut
        self.O = 0 # flight arcs originating at node n in fleet k
        self.I = 0 # flight arcs terminiating at node n in fleet k
        self.ni = 0 # ground arcs originating at any node n
        self.nf = 0 # ground arcs terminating at any node n
        
        
        self.d = 0 # distance of flight i [i]
        self.ac = 0 # number of aircraft in fleet of type k [k]
        self.cost = 0 # operating cost of AC type k for flight i [i, k]
        self.s = 0 # number of seats for aircraft type k [k]
        self.fare = 0 # average fare for itineray p [p]   
        
        self.D = 0 # daily unconstrained demand for itinerary p [p]
        self.Q = 0 # daily unconstrained demand on flight (leg) i [i]
        
        self.b = 0 # recapture rate of a pax that desired itinerary p and is allocated to r
        
        
        # Create binary matrix (dict) for flight leg as part of itinerary
        self.delta = {}
        for i in self.L:
            
            if i in p:    
                self.delta[i, p] = 1
            
            else:
                self.delta[i, p] = 0
        
    
    
    def rmp_model(self):
        # Initialise gurobipy model
        model = Model("RMP")
        
        # Define Decision Variables
        f = {} # 1 if flight arc i is assigned to aircraft type k, 0 otherwise [i, k]
        y = {} # number of aircraft of type k on the ground arc a [a, k]
        t = {} # number of passengers that would like to travel on itinerary p and are reallocated to iternary r [p, r]
        
        # Add Variables to Objective Function
        for i in self.L:
            for k in self.K:
                f[i, k] = model.addVar(obj=self.cost[k][i], vtype=GRB.CONTINIOUS)
        
        for p in self.P:
            for r in self.P:
                t[p, r] = model.addVar(obj=fare[p] - (b[p, r] * fare[r]), vtype=GRB.INTEGER)
        
        model.update()
        model.setObjective(model.getObjective(), GRB.MINIMIZE)
                
                 
        # Define Constraints
        for i in self.L:
            model.addConstr(quicksum(f[i, k] for k in self.K) == 1, name='C1')
          
        for n in self.Nk:
            for k in self.K:
                model.addConstr(y[n, k] + quicksum(f[i, k] - y[n, k] for i in self.O) - quicksum(f[i, k] for i in self.I) == 0, name='C2')   
        
        for cut in self.TC:
            for k in self.K:
                model.addConstr(quicksum(y[a, k] + f[a, k] for a in self.NGk) <= self.ac, name='C3')
        
        for i in self.L:
            model.addConstr(quicksum(self.s[k] * f[i, k] for k in self.K) + 
                            quicksum(quicksum(delta[i, p] * t[p, r] for p in self.P) for r in self.P) - 
                            quicksum(quicksum(delta[i, p] * b[r, p] * t[r, p] for p in self.P) for r in self.P) > Q[i], name='C4')
           
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
    pass