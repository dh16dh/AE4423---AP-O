"""

"""

import pandas as pd
from demand_forecast import DemandForecast

class Parameters:

    def __init__(self, aircraft_data, airport_data, annual_growth, demand_data, distance_data):
        self.aircraft_data = pd.read_csv(aircraft_data, index_col=0)

        forecast_model = DemandForecast(airport_data, demand_data, distance_data, annual_growth)
        forecast_model.calibrate()

        self.demand_matrix = forecast_model.forecast_demand()
        # g_matrix =
        self.distance_matrix = pd.read_csv(distance_data, index_col=0)
        self.yield_matrix = 5.9 * self.distance_matrix ** -0.76 + 0.043
        self.seat_list = self.aircraft_data['Seats']
        # CASK_matrix =
        self.speed_list = self.aircraft_data['Speed']
        self.LF = 0.8
        # AC =
        # LTO =
        # BT_list =
        # cost_list =



if __name__ == '__main__':
    aircraft_data = 'Groups_data/Aircraft_info.csv'
    airport_data = "Groups_data/Group_17_Airport_info.csv"
    demand_data = "Groups_data/Group_17_Demand.csv"
    distance_data = "Groups_data/Group_17_Distances.csv"
    annual_growth_data = "Groups_data/Group_17_Annual_growth.csv"

    parameters = Parameters(aircraft_data, airport_data, annual_growth_data, demand_data, distance_data)
    print(parameters.demand_matrix)
    print(parameters.distance_matrix)
    print(parameters.yield_matrix)
