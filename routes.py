# Import relevant modules
from leg_based_model import Parameters
import numpy as np
import pandas as pd
import itertools

#pd.set_option("display.max_columns", None)

# Initialize lists
return_routes = []
return_ranges = []
tri_routes = []
tri_ranges = []

# Set parameters from Parameters class
parameters = Parameters()
fixed_operating_costs = parameters.operating_cost
time_based_costs = parameters.time_cost
fuel_cost = parameters.fuel_cost
energy_cost = parameters.energy_cost

# Get aircraft ranges and airport distance matrix
aircraft_ranges = parameters.Range
distances = parameters.distance_matrix
airports = list(distances.index)

# Compute return distances and routes from return fligths from hub
for _ in airports:
    return_routes.append(['LIRA', _, 'LIRA'])

for _ in distances['LIRA']:
    return_ranges.append(_*2)  

# Create DataFrame with return flights from hub
return_flights = pd.DataFrame()
return_flights['route'] = return_routes
return_flights['range'] = return_ranges

# Return routes costs
return_operatingcosts = []
for index, row in return_flights.iterrows():
    x = 0.7 * ((fixed_operating_costs*2) + (time_based_costs + fuel_cost + energy_cost) * row['range'])
    y = x.to_dict()
    return_operatingcosts.append(y)

return_flights['operatingcosts'] = return_operatingcosts

# Create DataFrame with triangular flights from hub
for i in distances:
    for j in distances:
        if j == i or i == 'LIRA' or j == 'LIRA':
            continue
        dist_1 = distances['LIRA'][i]
        dist_2 = distances[i][j]
        dist_3 = distances[j]['LIRA']
        dist_total = dist_1 + dist_2 + dist_3
        tri_routes.append(['LIRA', i , j , 'LIRA'])
        tri_ranges.append(dist_total)

# Create DataFrame with triangular flights from hub
tri_flights = pd.DataFrame()
tri_flights['route'] = tri_routes
tri_flights['range'] = tri_ranges

# Triangular routes costs
tri_operatingcosts = []
for index, row in tri_flights.iterrows():
    disth1 = float(parameters.distance_matrix[row['route'][0]][[row['route'][1]]])
    dist12 = float(parameters.distance_matrix[row['route'][1]][[row['route'][2]]])
    dist2h = float(parameters.distance_matrix[row['route'][2]][[row['route'][3]]])
    x = fixed_operating_costs + (time_based_costs + fuel_cost + energy_cost)*dist12
    y = (0.7*((fixed_operating_costs*2) + (time_based_costs + fuel_cost)*(disth1 + dist2h))) + energy_cost*(disth1 + dist2h)
    z = x + y
    tri_operatingcosts.append(z.to_dict())

tri_flights['operatingcosts'] = tri_operatingcosts

# Concat return and triangular DataFrames
frames = [return_flights, tri_flights]
routes = pd.concat(frames).reset_index()
routes = routes.drop(['index'], axis=1)

# Create unique pairs from list of airports
def od_markets(route):
    l = list(itertools.permutations(route, 2))
    return l

# Set of ICAO
ICAOs = []
for i in routes['route']:
    ICAOs.append(list(set(i)))
routes['ICAOs'] = ICAOs

# Individual pairs
pairs = []
for route in routes['ICAOs']:
    pairs.append(od_markets(route))
routes['pairs'] = pairs

# Drop first row consisting of just [LIRA, LIRA, LIRA]
routes = routes.drop(0)

# Dictionary implementation functions
def subsequent_nodes(route):
    d_sub = {}
    for _ in range(len(route)-1):
        d_sub[route[_]] = route[_+1:len(route)]
    return d_sub

def precedent_nodes(route):
    d_pre = {}
    for _ in range(len(route)-1):
        d_pre[route[_]] = route[:_+1]
    return d_pre

# Subsequent nodes
sub_dicts = []
for route in routes['route']:
    sub_dicts.append(subsequent_nodes(route))
routes['subsequent'] = sub_dicts

# Precedent nodes
pre_dicts = []
for route in routes['route']:
    pre_dicts.append(precedent_nodes(route))
routes['precedent'] = pre_dicts

# TAT factor
def tat_factor(route):
    if len(route) == 3:
        return 2.5
    else:
        return 3.5

tat_fac = []
for route in routes['route']:
    tat_fac.append(tat_factor(route))
routes['tatfactor'] = tat_fac

# Limiting runway length
def limit_runway(route):
    run_lengths = []
    for airport in route:
        run_lengths.append(parameters.AP_rwy[airport])
    return min(run_lengths)

min_run = []
for route in routes['route']:
    min_run.append(limit_runway(route))

routes['minrwy'] = min_run

# print(routes)
