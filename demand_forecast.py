"""
Demand forecast for future demand
@author: David Hartong
"""

import pandas as pd
import numpy as np
import statsmodels.api as sms


class DemandForecast:

    def __init__(self, airport_data, demand_data, distance_data, annual_growth):
        self.airport_data = airport_data
        self.demand_data = demand_data
        self.distance_data = distance_data
        self.annual_growth = annual_growth

        self.fuel_cost = 1.42  # USD/gallon
        self.data = pd.DataFrame(columns=['Demand', 'Population', 'GDP', 'Fuel_Dist'])

        self.K = 0
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0

    # Load data from CSV files
    def format_data(self):
        airport_info = pd.read_csv(self.airport_data, index_col=1)
        demand_2020 = pd.read_csv(self.demand_data, index_col=0)
        distances = pd.read_csv(self.distance_data, index_col=0)

        # Filter Data for Population, GDP, and Airport ICAO codes
        pop_list_2020 = airport_info["Population"]
        gdp_list_2020 = airport_info["GDP"]
        airport_names = airport_info.index  # Complete list of airport ICAO codes
        demand_airports = demand_2020.index  # List of airport ICAO codes with known demand

        # Create matrices for each data group
        demand_log = np.log(demand_2020)
        pop_matrix = demand_2020.copy()
        gdp_matrix = demand_2020.copy()
        dst_matrix = np.log(distances.loc[demand_airports][demand_airports] * self.fuel_cost) * -1

        # Fill population and GDP matrices ln(pop_i * pop_j) and ln(GDP_i * GDP_j)
        for i in range(len(pop_list_2020)):
            for j in range(len(pop_list_2020)):
                pop_i = pop_list_2020[i]
                pop_j = pop_list_2020[j]
                gdp_i = gdp_list_2020[i]
                gdp_j = gdp_list_2020[j]
                pop_product = pop_j * pop_j
                gdp_product = gdp_i * gdp_j
                if airport_names[i] in demand_airports and airport_names[j] in demand_airports:
                    pop_matrix.at[airport_names[i], airport_names[j]] = np.log(pop_product)
                    gdp_matrix.at[airport_names[i], airport_names[j]] = np.log(gdp_product)

        # Create lists
        demand_log = demand_log.stack().values
        pop_matrix = pop_matrix.stack().values
        gdp_matrix = gdp_matrix.stack().values
        dst_matrix = dst_matrix.stack().values

        self.data = pd.DataFrame(np.transpose([demand_log, pop_matrix, gdp_matrix, dst_matrix]),
                                 columns=['Demand', 'Population', 'GDP', 'Fuel_Dist'])
        self.data = self.data.replace([np.inf, -np.inf], np.nan).dropna()

        return self.data

    def calibrate(self):
        """
        Calibrates gravity model using OLS. Sets calibration constant and parameters
        based on result.

        Non-Linear model:
        D_ij = k (pop_i * pop_j)^b1 * (GDP_i * GDP_j)^b2 / (fuel_cost * d_ij)^b3
        Linearised model:
        log(D_ij) = log(k) + b1 * log(pop_i * pop_j) + b2 * log(GDP_i * GDP_j) + b3 * -log(fuel_cost * d_ij)
        Y = alpha + beta_1 * X1 + beta_2 * X2 + beta_3 * X3
        """
        Y = self.data['Demand']
        X = self.data[['Population', 'GDP', 'Fuel_Dist']]
        X = sms.add_constant(X)

        est = sms.OLS(Y, X).fit()
        print(est.summary())

        params = est.params

        self.K = np.exp(params['const'])
        self.b1 = params['Population']
        self.b2 = params['GDP']
        self.b3 = params['Fuel_Dist']

    def forecast_population(self):

