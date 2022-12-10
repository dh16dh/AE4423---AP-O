# Import relevant modules
from models import Parameters
import numpy as np
import pandas as pd

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

# Compute return distances from hub
for _ in distances['LIRA']:
    return_ranges.append(_*2)

for _ in airports:
    return_routes.append(['LIRA', _, 'LIRA'])

# Create DataFrame with return flights from hub
bi_flights = pd.DataFrame(np.array(return_ranges), np.array(return_routes), columns=['range'])

# Create DataFrame with triangular flights from hub
for i in distances:
    for j in distances:
        if j == i or i == 'LIRA' or j == 'LIRA':
            continue
        dist_1 = distances['LIRA'][i]
        dist_2 = distances[i][j]
        dist_3 = distances[j]['LIRA']
