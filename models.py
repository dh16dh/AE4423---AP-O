"""

"""
import numpy as np
import pandas as pd
from gurobipy import Model, GRB, LinExpr, quicksum
from demand_forecast import DemandForecast


class Parameters:

    def __init__(self, aircraft_data='Groups_data/Aircraft_info.csv',
                 airport_data="Groups_data/Group_17_Airport_info.csv",
                 demand_data="Groups_data/Group_17_Demand.csv",
                 distance_data="Groups_data/Group_17_Distances.csv",
                 annual_growth="Groups_data/Group_17_Annual_growth.csv"):
        self.aircraft_data = pd.read_csv(aircraft_data, index_col=0)
        self.airport_data = pd.read_csv(airport_data, index_col=0)

        forecast_model = DemandForecast(airport_data, demand_data, distance_data, annual_growth)
        forecast_model.calibrate()

        zero_list = list(1 for i in range(len(self.airport_data)))
        zero_list[0] = 0
        bt_list = list(10 for i in range(len(self.aircraft_data)))

        self.fuel_cost_usdgal = 1.42
        self.demand_matrix = forecast_model.forecast_demand()
        self.g_values = pd.Series(zero_list, index=self.airport_data['ICAO Code'], name='g value')
        self.distance_matrix = pd.read_csv(distance_data, index_col=0)
        self.yield_matrix = 5.9 * self.distance_matrix ** -0.76 + 0.043
        self.seat_list = self.aircraft_data['Seats']
        self.speed_list = self.aircraft_data['Speed']
        self.LF = 0.8
        self.TAT = self.aircraft_data['TAT']
        self.BT = pd.Series(bt_list, name='Block Time', index=self.aircraft_data.index)

        self.lease_cost = self.aircraft_data['Lease_c']  # Needs to be multiplied by AC^k (weekly cost per aircraft)
        self.operating_cost = self.aircraft_data['Operating_c']  # Needs to be multiplied by z_{ij} (per flight leg)
        self.time_cost = self.aircraft_data['Time_c'] / self.aircraft_data['Speed']  # Needs to be multiplied by d_{ij}
        self.fuel_cost = self.aircraft_data['Fuel_c'] * self.fuel_cost_usdgal / 1.5  # Needs to be multiplied by d_{ij}


class LegBasedModel:
    def __init__(self):
        parameter_set = Parameters()

        # Define Sets
        self.N = parameter_set.airport_data['ICAO Code'].to_list()
        self.K = parameter_set.aircraft_data.index.to_list()[0:3]

        # Define Revenue Parameters
        self.Yield = parameter_set.yield_matrix.replace(np.inf, 0)  # ij
        self.d = parameter_set.distance_matrix  # ij
        # Define Cost Parameters
        self.C_Lk = parameter_set.lease_cost  # k
        self.C_Xk = parameter_set.operating_cost  # k
        self.C_Tk = parameter_set.time_cost  # k
        self.C_Fk = parameter_set.fuel_cost  # k

        # Define Constraint Parameters
        self.q = parameter_set.demand_matrix  # ij
        self.g = parameter_set.g_values  # i, j
        self.s = parameter_set.seat_list  # k
        self.sp = parameter_set.speed_list  # k
        self.LF = parameter_set.LF  # const
        self.LTO = parameter_set.TAT  # k
        self.BT = parameter_set.BT  # k

    def network_fleet_model(self):

        model = Model("NFM")

        # Define Decision Variables
        x = {}
        w = {}
        z = {}
        AC = {}

        # Add Variables to Objective Function
        for i in self.N:
            for j in self.N:
                x[i, j] = model.addVar(obj=self.Yield[i][j] * self.d[i][j], lb=0, vtype=GRB.INTEGER)
                w[i, j] = model.addVar(obj=self.Yield[i][j] * self.d[i][j] * 0.9, lb=0, vtype=GRB.INTEGER)
                for k in self.K:
                    z[i, j, k] = model.addVar(
                        obj=-((1 - 0.3 * (1 - self.g[i]) - 0.3 * (1 - self.g[j])) * (self.C_Xk[k] + self.d[i][j] *
                        (self.C_Tk[k] + self.C_Fk[k]))), lb=0, vtype=GRB.INTEGER)
        for k in self.K:
            AC[k] = model.addVar(obj= - self.C_Lk[k], lb=0, vtype=GRB.INTEGER)
            # Currently adding number of aircraft as DV but not part of OF
        model.update()
        model.setObjective(model.getObjective(), GRB.MAXIMIZE)

        # Define Constraints
        for i in self.N:
            for j in self.N:
                model.addConstr(x[i, j] + w[i, j] <= self.q[i][j], name='C1')
                model.addConstr(w[i, j] <= self.q[i][j] * self.g[i] * self.g[j], name='C1*')
                model.addConstr(x[i, j] + quicksum(w[i, m] * (1 - self.g[j]) for m in self.N) +
                                quicksum(w[m, j] * (1 - self.g[i]) for m in self.N) <=
                                quicksum(z[i, j, k] * self.s[k] * self.LF for k in self.K), name='C2')
            for k in self.K:
                model.addConstr(quicksum(z[i, j, k] for j in self.N) == quicksum(z[j, i, k] for j in self.N), name='C3')
        for k in self.K:
            model.addConstr(quicksum(quicksum(
                z[i, j, k] * (self.d[i][j] / self.sp[k] + self.LTO[k] * (1 + 0.5 * (1 - self.g[j]))) for j in self.N)
                                     for i in self.N) <= self.BT[k] * 7 * AC[k], name='C4')
            # Range and Budget constraint formulation to be done.
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

        print()
        print("Frequencies:----------------------------------")
        print()
        for i in self.N:
            for j in self.N:
                for k in self.K:
                    if z[i, j, k].X > 0:
                        print(i, 'to', j, z[i, j, k].X, 'with AC', k)
        for k in self.K:
            print(AC[k].X)

class ElectricACModel:
    def __init__(self):
        pass


if __name__ == '__main__':
    aircraft_path = 'Groups_data/Aircraft_info.csv'
    airport_path = "Groups_data/Group_17_Airport_info.csv"
    demand_path = "Groups_data/Group_17_Demand.csv"
    distance_path = "Groups_data/Group_17_Distances.csv"
    annual_growth_path = "Groups_data/Group_17_Annual_growth.csv"

    parameters = Parameters()
    model1 = LegBasedModel()

    model1.network_fleet_model()
