import pandas as pd
import numpy as np
from itertools import product
import time


def get_sec(time_str):
    """
    Function to convert time strings into seconds

    :param time_str: A time string in the format 'HH:MM:SS'
    :type time_str: str
    :return: Returns the time as seconds
    """
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def get_hhmmss(time_int):
    """
    Function to convert seconds into regular 'HH:MM:SS' format

    :param time_int: Time in seconds
    :type time_int: int
    :return: Returns time as string in format 'HH:MM:SS'
    """
    hhmmss = time.strftime('%H:%M:%S', time.gmtime(time_int))
    return hhmmss


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

        self.ICAO = self.flight_data['ORG'].unique()
        self.K = self.aircraft_data['Type']
        self.L = self.flight_data['Flight Number']
        self.P = self.itinerary_data['Itin No.']

        destination_nodes = self.flight_data.drop(columns=['ORG', 'Departure']).rename(columns={'DEST': 'ICAO',
                                                                                                'Arrival': 'Time'})
        origin_nodes = self.flight_data.drop(columns=['DEST', 'Arrival']).rename(columns={'ORG': 'ICAO',
                                                                                          'Departure': 'Time'})
        destination_nodes['Type'] = 'Inbound'
        origin_nodes['Type'] = 'Outbound'
        destination_nodes['Flight Number'] = destination_nodes['Flight Number'] + 'i'
        origin_nodes['Flight Number'] = origin_nodes['Flight Number'] + 'o'
        all_nodes = pd.concat([destination_nodes, origin_nodes], ignore_index=True)
        all_nodes['Time'] = all_nodes['Time'].map(get_sec)
        for k in self.K:
            all_nodes[k] = all_nodes[k] < 1000000
        all_nodes = all_nodes.set_index(['Flight Number']).sort_index()

        self.N = list(all_nodes.index)
        self.G = dict()

        ground_links = []
        for AP in self.ICAO:
            ground_nodes = all_nodes[all_nodes['ICAO'] == AP]
            outbound_nodes = ground_nodes[ground_nodes['Type'] == 'Outbound']
            inbound_nodes = ground_nodes[ground_nodes['Type'] == 'Inbound']

            node_combinations = list(product(inbound_nodes.index, outbound_nodes.index))
            for nodes in node_combinations:
                new_ground_link = {'Nodes': nodes,
                                   'ICAO': AP,
                                   'Time In': inbound_nodes.loc[nodes[0]]['Time'],
                                   'Time Out': outbound_nodes.loc[nodes[1]]['Time'],
                                   'A330': all(
                                       [inbound_nodes.loc[nodes[0]]['A330'], outbound_nodes.loc[nodes[1]]['A330']]),
                                   'A340': all(
                                       [inbound_nodes.loc[nodes[0]]['A340'], outbound_nodes.loc[nodes[1]]['A340']]),
                                   'B737': all(
                                       [inbound_nodes.loc[nodes[0]]['B737'], outbound_nodes.loc[nodes[1]]['B737']]),
                                   'B738': all(
                                       [inbound_nodes.loc[nodes[0]]['B738'], outbound_nodes.loc[nodes[1]]['B738']]),
                                   'BUS': all(
                                       [inbound_nodes.loc[nodes[0]]['BUS'], outbound_nodes.loc[nodes[1]]['BUS']])}
                ground_links.append(new_ground_link)
        G = pd.DataFrame(ground_links).set_index('Nodes')
        for k in self.K:
            self.G[k] = list(G[G[k] == True].index)

        time_list = np.unique([G['Time In'], G['Time Out']])
        self.TC = [0]
        for i in range(len(time_list)-1):
            time_cut = np.mean([time_list[i], time_list[i+1]])
            self.TC.append(time_cut)

        print(self.flight_data)


if __name__ == "__main__":
    parameters = Parameters()
    # print(parameters.G)
