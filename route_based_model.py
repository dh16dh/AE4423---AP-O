import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gurobipy import Model, GRB, LinExpr, quicksum
from demand_forecast import DemandForecast
from leg_based_model import Parameters
from routes import routes


class RouteBasedModel:
    def __init__(self):
        self.parameter_set = Parameters()

        # Airport Lookup DB

        # Define Sets
        self.N = self.parameter_set.airport_data.index.to_list()
        self.K = self.parameter_set.aircraft_data.index.to_list()
        self.R = routes['route']
        self.Rid = routes.index.to_list()
        self.S = routes['subsequent']
        self.P = routes['precedent']
        self.Pairs = routes['pairs']
        self.RouteRange = routes['range']
        self.ICAOList = routes['ICAOs']

        # Define Revenue Parameters
        self.Yield = self.parameter_set.yield_matrix.replace(np.inf, 0)  # ij
        self.d = self.parameter_set.distance_matrix  # ij
        # Define Cost Parameters
        self.C_Lk = self.parameter_set.lease_cost  # k
        self.C_Xk = self.parameter_set.operating_cost  # k
        self.C_Tk = self.parameter_set.time_cost  # k
        self.C_Fk = self.parameter_set.fuel_cost  # k

        # Define Constraint Parameters
        self.q = self.parameter_set.demand_matrix  # ij
        self.g = self.parameter_set.g_values  # i, j
        self.s = self.parameter_set.seat_list  # k
        self.sp = self.parameter_set.speed_list  # k
        self.LF = self.parameter_set.LF  # const
        self.LTO = self.parameter_set.TAT  # k
        self.BT = self.parameter_set.BT  # k
        self.Range = self.parameter_set.Range
        self.AP_rwy = self.parameter_set.AP_rwy  # i/j
        self.AC_rwy = self.parameter_set.AC_rwy  # k

        # Create binary matrix for range constraint
        self.a = {}
        for r in self.Rid:
            for k in self.K:
                if self.RouteRange[r] < self.Range[k]:
                    self.a[r, k] = 1
                else:
                    self.a[r, k] = 0

        self.delta = {}
        for r in self.Rid:
            for i in self.ICAOList[r]:
                for j in self.ICAOList[r]:
                    if (i, j) in self.Pairs[r]:
                        self.delta[i, j, r] = 1
                    else:
                        self.delta[i, j, r] = 0

        print(self.delta)        

        # Create binary matrix for runway constraint
        self.rwy = {}
        for i in self.N:
            for k in self.K:
                if self.AC_rwy[k] <= self.AP_rwy[i]:
                    self.rwy[i, k] = 1
                else:
                    self.rwy[i, k] = 0

    def network_fleet_model(self):

        #global self
        model = Model("NFM")

        # Define Decision Variables
        x = {}
        w = {}
        z = {}
        AC = {}

        # Add Variables to Objective Function
        for r in self.R:
            for i in self.N:
                for j in self.N:
                    x[i, j, r] = model.addVar(obj=routes['yield'][r] * routes['range'][r], lb=0, vtype=GRB.INTEGER)
                    w[i, j, r] = model.addVar(obj=routes['yield'][r] * routes['range'][r] * 0.9, lb=0, vtype=GRB.INTEGER)
                    for k in self.K:
                        z[r, k] = model.addVar(
                            obj=-((1 - 0.3 * (1 - self.g[i]) - 0.3 * (1 - self.g[j])) * (self.C_Xk[k] + self.d[i][j] *
                                                                                         (self.C_Tk[k] + self.C_Fk[
                                                                                             k]))),
                            lb=0, vtype=GRB.INTEGER)
        for k in self.K:
            AC[k] = model.addVar(obj=-self.C_Lk[k], lb=0, vtype=GRB.INTEGER)
            # Currently adding number of aircraft as DV but not part of OF
        model.update()
        model.setObjective(model.getObjective(), GRB.MAXIMIZE)

        # Define Constraints
        for r in self.R:
            for i in self.N:
                for j in self.N:
                    model.addConstr(quicksum(x[i, j, r] + quicksum(w[i, j, r, n] for n in self.R) for r in self.R) <= self.q[i][j],
                                    name='C1')
                    model.addConstr(w[i, j, r] <= self.q[i][j] * self.g[i] * self.g[j], name='C1*')
                    model.addConstr(x[i, j, r] + quicksum(w[i, m] * (1 - self.g[j]) for m in self.N) +
                                    quicksum(w[m, j] * (1 - self.g[i]) for m in self.N) <=
                                    quicksum(z[i, j, k] * self.s[k] * self.LF for k in self.K), name='C2')

            model.addConstr(quicksum(x['LIRA', m, r] for m in self.S) + quicksum(quicksum(quicksum(w[p, m, r, n] for m in self.S for p in self.N for n in self.R))) <= quicksum(z[r, k] * self.s[k] * self.LF for k in self.K), name = 'C3')
            model.addConstr(quicksum(x[m, 'LIRA', r] for m in self.P) + quicksum(quicksum(quicksum(w[m, p, r, n] for n in self.R for m in self.P for p in self.N))) <= quicksum(z[r, k] * self.s[k] * self.LF for k in self.K), name='C4')
            model.addConstr(quicksum(x[i, m, r] for m in self.S) + quicksum(x[m, j, i, r] for m in self.P) + quicksum(quicksum(quicksum(w[p, m, n, r] for n in self.R for p in self.P for m in self.S))) + quicksum(quicksum(quicksum(w[p, m, n, r] for n in self.R for p in self.N for m in self.S))) <= quicksum(z[r, k] * self.s[k] * self.LF), name='C5')

            for k in self.K:
                model.addConstr(quicksum(z[r, k] * (routes['range'][r] / self.sp[k] + self.LTO[k] * routes['TATfactor'] + self.chargetime[k]) for r in self.R) <= self.BT[k] * 7 * AC[k], name='C6')
        for i in self.N:
            for j in self.N:
                for k in self.K:
                    model.addConstr(z[r, k] <= self.a[r, k] * 999, name='C7')
                    model.addConstr(z[r, k] <= self.rwy[i, k] * self.rwy[j, k] * 999, name='RWY')

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

        result = pd.DataFrame(columns=['Origin', 'Destination', 'Frequency', 'AC Type', 'Direct Flow', 'Transfer Flow'])

        for i in self.N:
            for j in self.N:
                for k in self.K:
                    if z[r, k].X > 0:
                        new_row = pd.DataFrame([[i, j, z[r, k].X, k, x[i, j, r].X, w[i, j, r].X]],
                                               columns=['Origin', 'Destination', 'Frequency', 'AC Type',
                                                        'Direct Flow', 'Transfer Flow'])
                        result = pd.concat([result, new_row], ignore_index=True)
        for i in self.N:
            for j in self.N:
                if w[i, j, r].X > 0:
                    new_row = pd.DataFrame([[i, j, np.NaN, np.NaN, x[i, j, r].X, w[i, j, r].X]],
                                           columns=['Origin', 'Destination', 'Frequency', 'AC Type',
                                                    'Direct Flow', 'Transfer Flow'])
                    # result = pd.concat([result, new_row], ignore_index=True)
        print('Fleet')
        for k in self.K:
            print('Leasing', k, ':', AC[k].X)

        return result

if __name__ == '__main__':
    aircraft_path = 'Groups_data/Aircraft_info.csv'
    airport_path = "Groups_data/Group_17_Airport_info.csv"
    demand_path = "Groups_data/Group_17_Demand.csv"
    distance_path = "Groups_data/Group_17_Distances.csv"
    annual_growth_path = "Groups_data/Group_17_Annual_growth.csv"

    parameters = Parameters()
    model = RouteBasedModel()

    final_result = model.network_fleet_model()
    final_result.to_csv('Route-Based Results.csv')
    # route_csv = pd.read_csv('Leg-Based Results.csv')
    model.plot_routes(final_result)
    