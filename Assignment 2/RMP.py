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
        self.N = self.parameter_set.airport_data.index.to_list()
        self.K = self.parameter_set.aircraft_data.index.to_list()
        self.F = 0
        self.Gk = 0
        self.P = 0
        self.Pp = 0
        self.NGk = 0
        self.O = 0
        self.I = 0
        self.ni = 0
        self.nf = 0
        
        
        
        
        # Fares
        
        
        
        # Define Cost Parameters
        self.Cost = 0 # k, i
        
        
        
        # Define Constraint Parameters
        
        self.d = self.parameter_set.distance_matrix # i
        self.s = self.parameter_set.seat_list # k
        
        
      
    def rmp_model(self):
        # Initialise gurobipy model
        model = Model("RMP")
        
        # Define Decision Variables
        f = {}
        y = {}
        x = {}
        
        
        # Add Variables to Objective Function
        for i in self.L:
            for k in self.K:
                f[i, k] = model.addVar(obj=self.cost[k][i], vtype=GRB.CONTINIOUS)
        
        for p in self.P:
            for r in self.P:
                pass
        
        model.update()
        model.setObjective(model.getObjective(), GRB.MINIMIZE)
                
        
                
        # Define Constraints
        for i in self.L:
            model.addConstr(quicksum(f[i, k] for k in self.K) == 1, name='C1')
            
        
        
        for k in self.K:
            model.addConstr(quicksum(y[a, k] + f[a, k] for a in self.NGk), name='C3')
            
        
        
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
         


