"""

"""

import pandas as pd
from gurobipy import Model, GRB, LinExpr
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
        self.Yield = parameter_set.yield_matrix  # ij
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
                        obj=-(1 - 0.3 * (1 - self.g[i]) - 0.3 * (self.g[j])) * self.C_Xk[k] + self.d[i][j] *
                            (self.C_Tk[k] + self.C_Fk[k]), lb=0, vtype=GRB.INTEGER)
        for k in self.K:
            AC[k] = model.addVar(obj=-self.C_Lk, lb=0, vtype=GRB.INTEGER)

        model.update()
        model.setObjective(model.getObjective(), GRB.MAXIMIZE)


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
