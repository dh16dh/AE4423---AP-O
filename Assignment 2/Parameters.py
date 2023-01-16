import pandas as pd


class Parameters:
    def __init__(self,
                 aircraft_data='Group_data_17/Aircraft_data.csv',
                 flight_data='Group_data_17/Flight_data.csv',
                 itinerary_data='Group_data_17/Itinerary_data.csv',
                 new_flight_data='Group_data_17/New_flight_data.csv',
                 new_itinerary_data='Group_data_17/New_itinerary_data.csv',
                 optional_flight_data='Group_data_17/Optional_flight_data.csv',
                 recapture_rate='Group_data_17/Recapture_rate.csv'
                 ):

        self.aircraft_data = pd.read_csv(aircraft_data, sep=';')
        self.flight_data = pd.read_csv(flight_data, sep=';')
        self.itinerary_data = pd.read_csv(itinerary_data, sep=';')
        self.new_flight_data = pd.read_csv(new_flight_data, sep=';')
        self.new_itinerary_data = pd.read_csv(new_itinerary_data, sep=';')
        self.optional_flight_data = pd.read_csv(optional_flight_data, sep=';')
        self.recapture_rate = pd.read_csv(recapture_rate, sep=';')

        self.N = self.flight_data['ORG'].unique()
        self.K = self.aircraft_data['Type']
        self.L = self.flight_data['Flight Number']
        self.P = self.itinerary_data['Itin No.']

        for k in self.K:
            filtered_flights = self.flight_data[self.flight_data[k] < 1000000]
            for n in self.N:
                destination_set = self.flight_data[self.flight_data['DEST'] == n].drop(columns=['ORG', 'Departure'])
                origin_set = self.flight_data[self.flight_data['ORG'] == n].drop(columns=['DEST', 'Arrival'])
                destination_set.rename(columns={'DEST': 'IATA'}, inplace=True)
                origin_set.rename(columns={'ORG': 'IATA'}, inplace=True)






if __name__ == "__main__":
    parameters = Parameters()

