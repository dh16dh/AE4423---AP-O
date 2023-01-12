"""
File contains the RouteBasedModel class. Requires the Parameters class from leg_based_model.py and pre-processed routes
from the routes.py file

Dependencies:
Requires pandas data handling
Requires gurobipy for solution
Requires plotly for route map creation
Requires kaleido to export route map

@author: tijmengodeke
@author: swijnands97
@author: davidahartong
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gurobipy import Model, GRB, LinExpr, quicksum
from leg_based_model import Parameters
from routes import routes


class RouteBasedModel:
    def __init__(self):
        self.parameter_set = Parameters()

        # Airport Lookup DB

        # Define Sets
        self.N = self.parameter_set.airport_data.index.to_list()
        self.K = self.parameter_set.aircraft_data.index.to_list()
        self.Route = routes['route']
        self.R = routes.index.to_list()
        self.R2 = self.R[14::]
        self.S = routes['subsequent']  # r, H,
        self.P = routes['precedent']
        self.Pairs = routes['pairs']
        self.RouteRange = routes['range']
        self.ICAOList = routes['ICAOs']
        self.RouteMinRwy = routes['minrwy']
        self.LTO_Route = routes['tatfactor']

        # Define Revenue Parameters
        self.Yield = self.parameter_set.yield_matrix.replace(np.inf, 0)  # ij
        self.d = self.parameter_set.distance_matrix  # ij
        # Define Cost Parameters
        self.Cost = routes['operatingcosts']  # r, k
        self.Cost_Lease = self.parameter_set.lease_cost  # k

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
        self.ChargeTime = self.parameter_set.ChargeTime  # k

        # Create binary matrix for range constraint
        self.a = {}
        for r in self.R:
            for k in self.K:
                if self.RouteRange[r] < self.Range[k]:
                    self.a[r, k] = 1
                else:
                    self.a[r, k] = 0

        self.delta = {}
        for r in self.R:
            for i in self.N:
                for j in self.N:
                    if (i, j) in self.Pairs[r]:
                        self.delta[i, j, r] = 1
                    else:
                        self.delta[i, j, r] = 0
        # Create binary matrix for runway constraint
        self.rwy = {}
        for r in self.R:
            for k in self.K:
                if self.AC_rwy[k] <= self.RouteMinRwy[r]:
                    self.rwy[r, k] = 1
                else:
                    self.rwy[r, k] = 0

    def plot_routes(self, final_routes):
        airport_data = self.parameter_set.airport_data
        fig = go.Figure(data=go.Scattergeo())

        fig.update_geos(showcountries=True,
                        showsubunits=True,
                        lataxis_range=[36, 48],
                        lonaxis_range=[6, 20],
                        resolution=50)

        # Add Routes
        for i in range(len(final_routes.index)):
            fig.add_trace(go.Scattergeo(
                lon=[airport_data['Longitude (deg)'].loc[final_routes['Origin'].loc[i]],
                     airport_data['Longitude (deg)'].loc[final_routes['Destination'].loc[i]]],
                lat=[airport_data['Latitude (deg)'].loc[final_routes['Origin'].loc[i]],
                     airport_data['Latitude (deg)'].loc[final_routes['Destination'].loc[i]]],
                mode='lines',
                line=dict(
                    width=1,
                    color='red'
                ),
                opacity=final_routes['Frequency'].loc[i] / final_routes['Frequency'].max() / 2
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
        fig.write_image('Route_Map (Route Model).svg')
        fig.show()

    def network_fleet_model(self):

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
                    x[i, j, r] = model.addVar(obj=self.Yield[i][j] * self.d[i][j], lb=0, vtype=GRB.INTEGER)
                    for n in self.R:
                        w[i, j, r, n] = model.addVar(obj=self.Yield[i][j] * self.d[i][j] * 0.9, lb=0, vtype=GRB.INTEGER)
            for k in self.K:
                z[r, k] = model.addVar(obj=-self.Cost[r][k], lb=0, vtype=GRB.INTEGER)
        for k in self.K:
            AC[k] = model.addVar(obj=-self.Cost_Lease[k], lb=0, vtype=GRB.INTEGER)
        model.update()
        model.setObjective(model.getObjective(), GRB.MAXIMIZE)

        # Define Constraints
        for i in self.N:
            for j in self.N:
                model.addConstr(
                    quicksum(x[i, j, r] + quicksum(w[i, j, r, n] for n in self.R) for r in self.R) <= self.q[i][j],
                    name='C1')
                for r in self.R:
                    model.addConstr(x[i, j, r] <= self.q[i][j] * self.delta[i, j, r], name='C1*_1')
                    for n in self.R:
                        model.addConstr(
                            w[i, j, r, n] <= self.q[i][j] * self.delta[i, 'LIRA', r] * self.delta['LIRA', j, n],
                            name='C1*_2')
        for r in self.R:
            model.addConstr(quicksum(x['LIRA', m, r] for m in self.S[r]['LIRA']) + quicksum(
                quicksum(quicksum(w[p, m, n, r] for m in self.S[r]['LIRA']) for p in self.N) for n in
                self.R) <= quicksum(z[r, k] * self.s[k] * self.LF for k in self.K), name='C2_Hm')
        for r in self.R2:
            i = self.S[r]['LIRA'][0]
            j = self.S[r]['LIRA'][1]
            model.addConstr(
                quicksum(x[i, m, r] for m in self.S[r][j]) + quicksum(x[m, j, r] for m in self.P[r][i]) + quicksum(
                    quicksum(w[p, j, n, r] for p in self.N) for n in self.R) + quicksum(
                    quicksum(w[i, p, r, n] for p in self.N) for n in self.R) <= quicksum(
                    z[r, k] * self.s[k] * self.LF for k in self.K), name='C2_spokes')
        for r in self.R:
            i = self.S[r]['LIRA'][-2]
            model.addConstr(quicksum(x[m, 'LIRA', r] for m in self.P[r][i]) + quicksum(
                quicksum(quicksum(w[m, p, r, n] for m in self.P[r][i]) for p in self.N) for n in self.R) <= quicksum(
                z[r, k] * self.s[k] * self.LF for k in self.K), name='C2_mH')
        for k in self.K:
            model.addConstr(quicksum(
                (self.RouteRange[r] / self.sp[k] + self.LTO[k] * self.LTO_Route[r] + self.ChargeTime[k]) * z[r, k] for r
                in self.R) <= 7 * self.BT[k] * AC[k], name='C4')
        for r in self.R:
            for k in self.K:
                model.addConstr(z[r, k] <= self.a[r, k] * 999, name='C7')
                model.addConstr(z[r, k] <= self.rwy[r, k] * 999, name='RWY')

        model.update()

        model.setParam('Timelimit', 900)  # Set Timeout limit to 15 minutes

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

        result = pd.DataFrame(columns=['Origin', 'Destination', 'Route', 'Frequency', 'AC Type',
                                       'Direct Flow', 'Transfer Flow', 'Capacity', 'LF'])

        rpk = 0

        # Create List of Routes Flown
        for r in self.R:
            for k in self.K:
                if z[r, k].X > 0:
                    for apt in range(len(self.Route[r]) - 1):
                        i = self.Route[r][apt]
                        j = self.Route[r][apt + 1]
                        x_total = 0
                        w_total = 0
                        if i == 'LIRA' and j == self.Route[r][1]:
                            for m in self.S[r]['LIRA']:
                                x_total += x['LIRA', m, r].X
                                for p in self.N:
                                    for n in self.R:
                                        w_total += w[p, m, n, r].X
                        if i != 'LIRA' and j != 'LIRA':
                            for m in self.S[r][j]:
                                x_total += x[i, m, r].X
                            for m in self.P[r][i]:
                                x_total += x[m, j, r].X
                            for p in self.N:
                                for n in self.R:
                                    w_total += w[p, j, n, r].X + w[i, p, r, n].X
                        if i == self.Route[r][-2] and j == 'LIRA':
                            for m in self.P[r][self.S[r]['LIRA'][-2]]:
                                x_total += x[m, 'LIRA', r].X
                                for p in self.N:
                                    for n in self.R:
                                        w_total += w[m, p, r, n].X
                        Capacity = 0
                        for k1 in self.K:
                            Capacity += self.s[k1] * z[r, k1].X
                        LF = (x_total + w_total) / Capacity
                        rpk += (x_total + w_total) * self.d[i][j]
                        new_row = pd.DataFrame([[i, j, self.Route[r], z[r, k].X, k, x_total, w_total, Capacity, LF]],
                                               columns=['Origin', 'Destination', 'Route', 'Frequency', 'AC Type',
                                                        'Direct Flow', 'Transfer Flow', 'Capacity', 'LF'])
                        result = pd.concat([result, new_row], ignore_index=True)
        # KPIs
        # Print fleet composition
        print('Fleet')
        for k in self.K:
            print(k)
            print('Fleet:', AC[k].X)
            hours = 0
            block_time = self.BT[k] * 7 * AC[k].X
            flights = 0
            for r in self.R:
                flight_hrs = z[r, k].X * (
                        self.RouteRange[r] / self.sp[k] + self.LTO[k] * self.LTO_Route[r] + self.ChargeTime[k])
                hours += flight_hrs
                if z[r, k].X > 0:
                    flights += len(self.Route[r]) - 1
            utilization = hours / block_time * 100
            print('Utilisation:', utilization, '%')
            print('Weekly Flights:', flights)
            print()
        LF_Avg = result['LF'].mean()
        print('Total RPK:', rpk)
        print('AVG Load Factor:', LF_Avg)
        return result


if __name__ == '__main__':
    aircraft_path = 'Groups_data/Aircraft_info.csv'
    airport_path = "Groups_data/Group_17_Airport_info.csv"
    demand_path = "Groups_data/Group_17_Demand.csv"
    distance_path = "Groups_data/Group_17_Distances.csv"
    annual_growth_path = "Groups_data/Group_17_Annual_growth.csv"

    parameters = Parameters()
    route_model = RouteBasedModel()

    final_result = route_model.network_fleet_model()
    final_result.to_csv('Route-Based Results.csv')
    # route_csv = pd.read_csv('Route-Based Results.csv')
    route_model.plot_routes(final_result)
