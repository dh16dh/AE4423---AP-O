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

        self.aircraft_data = pd.read_csv(aircraft_data, index_col=0)
        self.flight_data = pd.read.csv(flight_data, index_col=0)
        self.itinerary_data = pd.read_csv(itinerary_data, index_col=0)
        self.new_flight_data = pd.read.csv(new_flight_data, index_col=0)
        self.new_itinerary_data = pd.read.csv(new_itinerary_data, index_col=0)
        self.optional_flight_data = pd.read.csv(optional_flight_data, index_col=0)
        self.recapture_rate = pd.read.csv(recapture_rate)