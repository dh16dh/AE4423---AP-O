"""
This file contains the Parameters class that preprocesses parameters required from relevant data files to use in the
linear programming models.
This also contains the LegBasedModel class

Dependencies:
Requires pandas data handling
Requires gurobipy for solution
Requires plotly for route map creation
Requires kaleido to export route map

author: @davidahartong
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gurobipy import Model, GRB, LinExpr, quicksum
from demand_forecast import DemandForecast


class Parameters:

    def __init__(self, aircraft_data='Groups_data/Aircraft_info.csv',
                 airport_data="Groups_data/Group_17_Airport_info.csv",
                 demand_data="Groups_data/Group_17_Demand.csv",
                 distance_data="Groups_data/Group_17_Distances.csv",
                 annual_growth="Groups_data/Group_17_Annual_growth.csv"):
        """
        This class preprocesses data from the relevant CSV files into a usable format for the two mathematical LP models

        :param aircraft_data: Optional path to the aircraft data CSV
        :param airport_data: Optional path to the airport data CSV
        :param demand_data: Optional path to the 2020 demand data CSV
        :param distance_data: Optional path to the distance data CSV
        :param annual_growth: Optional path to the annual growth data CSV
        """
        # Reads aircraft and airport data
        self.aircraft_data = pd.read_csv(aircraft_data, index_col=0)
        self.airport_data = pd.read_csv(airport_data, index_col=1)

        # Passes CSV path information to DemandForecast class to calibrate demand forecast model
        forecast_model = DemandForecast(airport_data, demand_data, distance_data, annual_growth)
        forecast_model.calibrate()

        # Create binary variable list for the g value to check for the hub airport
        zero_list = list(1 for i in range(len(self.airport_data)))
        zero_list[0] = 0
        # Create list for the block time for all aircraft types
        bt_list = list(10 for i in range(len(self.aircraft_data)))

        # Set various callable parameters for use in the LP models.
        self.fuel_cost_usdgal = 1.42
        self.demand_matrix = forecast_model.forecast_demand()
        self.g_values = pd.Series(zero_list, index=self.airport_data.index, name='g value')
        self.distance_matrix = pd.read_csv(distance_data, index_col=0)
        self.yield_matrix = 5.9 * self.distance_matrix ** -0.76 + 0.043
        self.seat_list = self.aircraft_data['Seats']
        self.speed_list = self.aircraft_data['Speed']
        self.LF = 0.8
        self.TAT = self.aircraft_data['TAT'] / 60
        self.ChargeTime = self.aircraft_data['Charging'] / 60
        self.BT = pd.Series(bt_list, name='Block Time', index=self.aircraft_data.index)
        self.Range = self.aircraft_data['Range']
        self.AP_rwy = self.airport_data['Runway (m)']
        self.AC_rwy = self.aircraft_data['Runway']

        self.lease_cost = self.aircraft_data['Lease_c']  # Needs to be multiplied by AC^k (weekly cost per aircraft)
        self.operating_cost = self.aircraft_data['Operating_c']  # Needs to be multiplied by z_{ij} (per flight leg)
        self.time_cost = self.aircraft_data['Time_c'] / self.aircraft_data['Speed']  # Needs to be multiplied by d_{ij}
        self.fuel_cost = self.aircraft_data['Fuel_c'] * self.fuel_cost_usdgal / 1.5  # Needs to be multiplied by d_{ij}

        self.energy_price = 0.07  # Wh/eur
        self.energy_cost = self.energy_price * self.aircraft_data['Energy_c'] / self.aircraft_data[
            'Range']  # Needs to be multiplied by d_{ij}


class LegBasedModel:
    def __init__(self):
        """
        The LegBasedModel class contains all the code for creating, optimising and visualising the leg-based fleet
        network model.

        The network_fleet_model() function which creates, optimises and returns final network.

        The plot_routes() function visualises the final network on a map.
        """
        # Call Parameter class to extract and assign variables
        self.parameter_set = Parameters()

        # Define Sets
        self.N = self.parameter_set.airport_data.index.to_list()
        self.K = self.parameter_set.aircraft_data.index.to_list()[0:3]

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

        # Create binary matrix (dict) for range constraint
        self.a = {}
        for i in self.N:
            for j in self.N:
                for k in self.K:
                    if self.d[i][j] < self.Range[k]:
                        self.a[i, j, k] = 1
                    else:
                        self.a[i, j, k] = 0
        # Create binary matrix (dict) for runway constraint
        self.rwy = {}
        for i in self.N:
            for k in self.K:
                if self.AC_rwy[k] <= self.AP_rwy[i]:
                    self.rwy[i, k] = 1
                else:
                    self.rwy[i, k] = 0

    def plot_routes(self, routes):
        airport_data = self.parameter_set.airport_data
        fig = go.Figure(data=go.Scattergeo())

        fig.update_geos(showcountries=True,
                        showsubunits=True,
                        lataxis_range=[36, 48],
                        lonaxis_range=[6, 20],
                        resolution=50)

        # Add Routes
        for i in range(len(routes.index)):
            fig.add_trace(go.Scattergeo(
                lon=[airport_data['Longitude (deg)'].loc[routes['Origin'].loc[i]],
                     airport_data['Longitude (deg)'].loc[routes['Destination'].loc[i]]],
                lat=[airport_data['Latitude (deg)'].loc[routes['Origin'].loc[i]],
                     airport_data['Latitude (deg)'].loc[routes['Destination'].loc[i]]],
                mode='lines',
                line=dict(
                    width=1,
                    color='red'
                ),
                opacity=routes['Frequency'].loc[i] / routes['Frequency'].max() / 2
            ))

        # Add airports:
        fig.add_trace(go.Scattergeo(
            lon=airport_data['Longitude (deg)'],
            lat=airport_data['Latitude (deg)'],
            text=airport_data['City Name'],
            mode='markers',
            marker=dict(
                size=5,
                color='red'
            )
        ))
        fig.add_trace(go.Scattergeo(
            lon=[airport_data.loc['LIRA']['Longitude (deg)']],
            lat=[airport_data.loc['LIRA']['Latitude (deg)']],
            text=airport_data['City Name']['LIRA'],
            mode='markers',
            marker=dict(
                size=10,
                color='blue'
            )
        ))
        fig.update_layout(showlegend=False)
        fig.write_image('Route_Map (Leg Model).svg')
        fig.show()

    def network_fleet_model(self):
        # Initialise gurobipy model
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
                                                                                     (self.C_Tk[k] + self.C_Fk[k]))),
                        lb=0, vtype=GRB.INTEGER)
        for k in self.K:
            AC[k] = model.addVar(obj=-self.C_Lk[k], lb=0, vtype=GRB.INTEGER)
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
        for i in self.N:
            for j in self.N:
                for k in self.K:
                    model.addConstr(z[i, j, k] <= self.a[i, j, k] * 999, name='C5')
                    model.addConstr(z[i, j, k] <= self.rwy[i, k] * self.rwy[j, k] * 999, name='RWY')

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

        # Create empty result dataframe to append results to
        result = pd.DataFrame(columns=['Origin', 'Destination', 'Frequency', 'AC Type',
                                       'Direct Flow', 'Transfer Flow', 'Capacity', 'LF'])

        # Create List of Routes Flown
        for i in self.N:
            for j in self.N:
                for k in self.K:
                    if z[i, j, k].X > 0:
                        w_total = 0
                        if i == 'LIRA':
                            for m in self.N:
                                w_total += w[m, j].X
                        if j == 'LIRA':
                            for m in self.N:
                                w_total += w[i, m].X
                        Capacity = 0
                        for k1 in self.K:
                            Capacity += self.s[k1] * z[i, j, k1].X
                        LF = (x[i, j].X + w_total) / Capacity
                        new_row = pd.DataFrame([[i, j, z[i, j, k].X, k, x[i, j].X, w_total, Capacity, LF]],
                                               columns=['Origin', 'Destination', 'Frequency', 'AC Type',
                                                        'Direct Flow', 'Transfer Flow', 'Capacity', 'LF'])
                        result = pd.concat([result, new_row], ignore_index=True)
        # KPIs
        # Print fleet composition and calculate utilisation and weekly flights per AC type
        print('Fleet')
        for k in self.K:
            print(k)
            print('Fleet:', AC[k].X)
            hours = 0
            block_time = self.BT[k] * 7 * AC[k].X
            flights = 0

            for i in self.N:
                for j in self.N:
                    flight_hrs = z[i, j, k].X * (self.d[i][j] / self.sp[k] + self.LTO[k] * (1 + 0.5 * (1 - self.g[j])))
                    hours += flight_hrs
                    flights += z[i, j, k].X
            utilization = hours / block_time * 100
            print('Utilisation:', utilization, '%')
            print('Weekly Flights:', flights)
            print()
        # Calculate the total RPK to give an indication of the number of passengers and distance flown as well as the
        # average load factor of all legs (adds all passengers and divides by available seats)
        rpk = 0
        LF = []
        for i in self.N:
            for j in self.N:
                rpk += x[i, j].X * self.d[i][j]
                if x[i, j].X > 0:
                    seats = 0
                    for k in self.K:
                        seats += z[i, j, k].X * self.s[k]
                    pax = x[i, j].X
                    pax_w = 0
                    if i == 'LIRA':
                        for m in self.N:
                            pax_w += w[m, j].X
                    if j == 'LIRA':
                        for m in self.N:
                            pax_w += w[i, m].X
                    pax += pax_w
                    routeLF = pax / seats
                    LF.append(routeLF)
                if w[i, j].X > 0:
                    rpk += w[i, j].X * (self.d[i]['LIRA'] + self.d['LIRA'][j])

        print('Total RPK:', rpk)
        print('AVG Load Factor', np.mean(LF))

        return result


if __name__ == '__main__':
    aircraft_path = 'Groups_data/Aircraft_info.csv'
    airport_path = "Groups_data/Group_17_Airport_info.csv"
    demand_path = "Groups_data/Group_17_Demand.csv"
    distance_path = "Groups_data/Group_17_Distances.csv"
    annual_growth_path = "Groups_data/Group_17_Annual_growth.csv"

    parameters = Parameters()
    model1 = LegBasedModel()

    final_result = model1.network_fleet_model()
    final_result.to_csv('Leg-Based Results.csv')
    # route_csv = pd.read_csv('Leg-Based Results.csv')
    model1.plot_routes(final_result)
