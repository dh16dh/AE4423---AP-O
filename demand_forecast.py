"""
Demand forecast for future demand.

@author: David Hartong
"""

import pandas as pd
import numpy as np
import statsmodels.api as sms


class DemandForecast:

    def __init__(self, airport_data, demand_data, distance_data, annual_growth, years=10, fuel_cost=1.42):
        """
        Creates DemandForecast class object, which takes path information for all relevant data files and optional
        fuel cost value

        Use forecast_demand() function to retrieve the forecasted demand matrix for the 15 given airports.

        Optional: Use calibrate() function to calibrate the model to retrieve model parameters as class properties
        (K, b1, b2, b3)


        :param airport_data: Path to airport data CSV
        :type airport_data: str
        :param demand_data: Path to demand data CSV
        :type demand_data: str
        :param distance_data: Path to distance data CSV
        :type distance_data: str
        :param annual_growth: Path to annual growth CSV
        :type annual_growth: str
        :param fuel_cost: Cost of fuel in USD/gallon. By default 1.42
        :type fuel_cost: float
        """
        self.airport_data = pd.read_csv(airport_data, index_col=1)
        self.demand_data = pd.read_csv(demand_data, index_col=0)
        self.distance_data = pd.read_csv(distance_data, index_col=0)
        self.annual_growth = pd.read_csv(annual_growth, index_col=False, header=None).values[0][0]

        self.years = years  # Number of years to forecast
        self.fuel_cost = fuel_cost  # USD/gallon
        self.data = pd.DataFrame(columns=['Demand', 'Population', 'GDP', 'Fuel_Dist'])

        self.K = 0
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0

        np.seterr(invalid='ignore', divide='ignore')

    def format_data(self):
        # Filter Data for Population, GDP, and Airport ICAO codes
        pop_list_2020 = self.airport_data["Population"]
        gdp_list_2020 = self.airport_data["GDP"]
        airport_names = self.airport_data.index  # Complete list of airport ICAO codes
        demand_airports = self.demand_data.index  # List of airport ICAO codes with known demand

        # Create matrices for each data group
        demand_log = np.log(self.demand_data)
        pop_matrix = self.demand_data.copy()
        gdp_matrix = self.demand_data.copy()
        dst_matrix = -1 * np.log(self.distance_data.loc[demand_airports][demand_airports] * self.fuel_cost)

        # Fill population and GDP matrices with ln(pop_i * pop_j) and ln(GDP_i * GDP_j)
        for i in range(len(pop_list_2020)):
            for j in range(len(pop_list_2020)):
                pop_i = pop_list_2020[i]
                pop_j = pop_list_2020[j]
                gdp_i = gdp_list_2020[i]
                gdp_j = gdp_list_2020[j]
                pop_product = pop_i * pop_j
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

    def calibrate(self):
        """
        Calibrates gravity model using OLS. Sets calibration constant and parameters
        based on result.

        Non-Linear model:
            D_{ij} = k (pop_i * pop_j)^{b_1} * (GDP_i * GDP_j)^{b_2} / (fuel_cost * d_ij)^{b_3}
        Linearised model:
            log(D_ij) = log(k) + b1 * log(pop_i * pop_j) + b2 * log(GDP_i * GDP_j) + b3 * -log(fuel_cost * d_ij)

            Y = alpha + beta_1 * X1 + beta_2 * X2 + beta_3 * X3
        """
        self.format_data()

        Y = self.data['Demand']
        X = self.data[['Population', 'GDP', 'Fuel_Dist']]
        X = sms.add_constant(X)

        est = sms.OLS(Y, X).fit()

        params = est.params

        self.K = np.exp(params['const'])
        self.b1 = params['Population']
        self.b2 = params['GDP']
        self.b3 = params['Fuel_Dist']

    def gravity_model(self, data, i, j):
        demand = self.K * (data["Population"][i] * data["Population"][j]) ** self.b1 \
                 * (data["GDP"][i] * data["GDP"][j]) ** self.b2 / (self.fuel_cost * self.distance_data[i][j]) ** self.b3
        return demand

    def forecast_demand(self):
        params = [self.K, self.b1, self.b2, self.b3]
        if all(param == 0 for param in params):
            print("=== Calibrating Model ===")
            self.calibrate()
            print("Parameters Calibrated")
            print("K :", self.K, "\nb1:", self.b1, "\nb2:", self.b2, "\nb3:", self.b3)
        forecasted_data = self.airport_data.copy()
        forecasted_data["Population"] = forecasted_data["Population"] * self.annual_growth ** self.years

        airport_names = self.airport_data.index.to_list()

        forecasted_demand = pd.DataFrame(columns=airport_names, index=airport_names)

        for i in airport_names:
            for j in airport_names:
                if i == j:
                    forecasted_demand[i][j] = 0
                    continue
                forecasted_demand[i][j] = self.gravity_model(forecasted_data, i, j)

        return forecasted_demand


if __name__ == '__main__':
    airport_data_path = "Groups_data/Group_17_Airport_info.csv"
    demand_data_path = "Groups_data/Group_17_Demand.csv"
    distance_data_path = "Groups_data/Group_17_Distances.csv"
    annual_growth_data_path = "Groups_data/Group_17_Annual_growth.csv"

    demand_model = DemandForecast(airport_data_path, demand_data_path, distance_data_path, annual_growth_data_path)
    forecast_demand = demand_model.forecast_demand()
    print(forecast_demand)

    forecast_demand.to_csv('Demand_2030.csv')
