# Import relevant modules
from models import Parameters
import numpy as np
import pandas as pd
import itertools

pd.set_option("display.max_columns", None)

# Initialize lists
return_routes = []
return_ranges = []
tri_routes = []
tri_ranges = []
return_yield = []
tri_yield = []

# Set parameters from Parameters class
parameters = Parameters()

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


# Return route yield
def calc_yield(route):
    return parameters.yield_matrix[route[0]][route[1]]*2

for _ in return_flights['route']:
    return_yield.append(calc_yield(_))

return_flights['yield'] = return_yield

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

# Return triangular yield
def calc_yield(route):
    return parameters.yield_matrix[route[0]][route[1]] + parameters.yield_matrix[route[1]][route[2]] + parameters.yield_matrix[route[2]][route[3]]

for _ in tri_flights['route']:
    tri_yield.append(calc_yield(_))

tri_flights['yield'] = tri_yield

# Concat return and triangular DataFrames
frames = [return_flights, tri_flights]
routes = pd.concat(frames)

# Create unique pairs from list of airports
def od_markets(route):
    l = list(itertools.permutations(route, 2))
    return l

# Set of ICAO
ICAOs = []
for i in routes['route']:
    ICAOs.append(set(i))
routes['ICAOs'] = ICAOs

# Individual pairs
pairs = []
for route in routes['ICAOs']:
    pairs.append(od_markets(route))
routes['pairs'] = pairs

# Drop first row consisting of just [LIRA, LIRA, LIRA]
routes = routes.iloc[1:, :]

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

#print(routes)