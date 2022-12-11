# Import relevant modules
from models import Parameters
import numpy as np
import pandas as pd
import itertools

# Initialize lists
return_routes = []
return_ranges = []
tri_routes = []
tri_ranges = []

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

# Implement routes in routes DataFrame
routes['pairs'] = pairs

#print(od_markets(['LIRA', 'LIBD', 'LIPQ']))

print(routes)
