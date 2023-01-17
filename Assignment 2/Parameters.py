import pandas as pd
import numpy as np
from itertools import product
import time
import pickle

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

        self.aircraft_data = pd.read_csv(aircraft_data, sep=';', index_col=0)
        self.flight_data = pd.read_csv(flight_data, sep=';', index_col=0)
        self.itinerary_data = pd.read_csv(itinerary_data, sep=';')
        self.new_flight_data = pd.read_csv(new_flight_data, sep=';')
        self.new_itinerary_data = pd.read_csv(new_itinerary_data, sep=';')
        self.optional_flight_data = pd.read_csv(optional_flight_data, sep=';')
        self.recapture_rate = pd.read_csv(recapture_rate, sep=';')

        # Define basic parameters from imported data including list of unique airports, AC type, itineraries,
        # and flight numbers
        self.ICAO = self.flight_data['ORG'].unique()
        self.K = self.aircraft_data.index
        self.P = self.itinerary_data['Itin No.']
        self.F = self.flight_data.index

        # # Create Flight Arcs, such that Fl.No. is the index
        # self.Flight_Arcs = self.flight_data.copy()
        # self.Flight_Arcs = self.Flight_Arcs.set_index(['Flight Number'])
        # self.d = self.Flight_Arcs['']

        self.AC = self.aircraft_data['Units']
        self.s = self.aircraft_data['Seats']
        self.TAT = self.aircraft_data['TAT']

        # Convert departure and arrival times into seconds (int)
        self.flight_data['Departure'] = self.flight_data['Departure'].map(get_sec)
        self.flight_data['Arrival'] = self.flight_data['Arrival'].map(get_sec)

        # Divide arrival and departure nodes into separate node sets
        destination_nodes = self.flight_data.drop(columns=['ORG', 'Departure']).rename(columns={'DEST': 'ICAO',
                                                                                                'Arrival': 'Time'})
        origin_nodes = self.flight_data.drop(columns=['DEST', 'Arrival']).rename(columns={'ORG': 'ICAO',
                                                                                          'Departure': 'Time'})
        destination_nodes['Type'] = 'Inbound'
        origin_nodes['Type'] = 'Outbound'

        destination_nodes = destination_nodes.reset_index().rename(columns={'index': 'Flight Number'})
        origin_nodes = origin_nodes.reset_index().rename(columns={'index': 'Flight Number'})

        # Combine arrival and departure nodes to create list of all nodes.
        all_nodes = pd.concat([destination_nodes, origin_nodes], ignore_index=True)
        all_nodes['Index'] = all_nodes['ICAO']
        all_nodes = all_nodes.set_index(['Index', 'Time']).sort_index()

        # Define set of all nodes, outbound nodes, inbound nodes, outbound flight arcs, inbound flight arcs
        self.N = dict()
        self.N_O = dict()
        self.N_I = dict()
        self.O = dict()
        self.I = dict()
        k_flights = dict()
        k_outbound = dict()
        k_inbound = dict()

        # Determine set of all nodes, outbound nodes, inbound nodes for each AC type,
        # and outbound/inbound DataFrames
        for k in self.K:
            filtered_flights = all_nodes[all_nodes[k] < 1000000]
            self.N[k] = list(filtered_flights.index.unique())
            k_flights[k] = filtered_flights

            outbound_flights = filtered_flights[filtered_flights['Type'] == 'Outbound']
            inbound_flights = filtered_flights[filtered_flights['Type'] == 'Inbound']

            k_outbound[k] = outbound_flights
            k_inbound[k] = inbound_flights

            self.N_O[k] = list(outbound_flights.index.unique())
            self.N_I[k] = list(inbound_flights.index.unique())

        # Determine set of outbound and inbound flight arcs per node and per AC type
        for k in self.K:
            for n in self.N_O[k]:
                df = k_outbound[k].loc[[n]]
                flight_arc = list(df['Flight Number'])
                self.O[k, n] = flight_arc
            for n in self.N_I[k]:
                df = k_inbound[k].loc[[n]]
                flight_arc = list(df['Flight Number'])
                self.I[k, n] = flight_arc

        # Define set of ground links and set of ground links for DataFrame
        self.G = dict()
        ground_link_set = []

        for k in self.K:
            gl_k_set = []
            for ap in self.ICAO:
                nodes_ap = [item for item in self.N[k] if item[0] == ap]
                for i in range(len(nodes_ap)):
                    node_1 = nodes_ap[i-1]
                    node_2 = nodes_ap[i]
                    ground_link = ap + str(node_1[1]) + '-' + str(node_2[1])
                    new_gl = {'Ground Link': ground_link,
                              'ICAO': ap,
                              'T1': node_1[1],
                              'T2': node_2[1],
                              'Type': k}
                    ground_link_set.append(new_gl)
                    gl_k_set.append(ground_link)
            self.G[k] = gl_k_set
        self.Ground_Links = pd.DataFrame(ground_link_set).set_index(['Ground Link'])

        print(self.Ground_Links)

        # Define originating and terminating ground links sets
        self.n_plus = dict()
        self.n_minus = dict()

        # Determine originating and terminating ground links for AC type and node
        for k in self.K:
            ground_links = self.Ground_Links[self.Ground_Links['Type'] == k]
            for i in range(len(ground_links)):
                node_plus = (ground_links.iloc[i][0], ground_links.iloc[i][1])
                node_minus = (ground_links.iloc[i][0], ground_links.iloc[i][2])
                self.n_plus[k, node_plus] = list(ground_links.index)[i]
                self.n_minus[k, node_minus] = list(ground_links.index)[i]

        # Define complete set of time cuts
        self.TC = dict()

        # Determine set of time cuts for each AC, based on median between each node + 0 for overnight links
        for k in self.K:
            time_list = []
            for n in self.N[k]:
                time_list.append(n[1])
            unique_time_list = np.unique(np.sort(time_list))
            TC_list = [0]
            for i in range(len(unique_time_list)-1):
                TC_time = np.median([unique_time_list[i], unique_time_list[i+1]])
                TC_list.append(TC_time)
            self.TC[k] = TC_list

        # Define complete set of arcs per time cut per AC
        self.NG = dict()

        # Determine set of ground links and flight arcs per time cut and per AC type
        for k in self.K:
            ground_links = self.Ground_Links[self.Ground_Links['Type'] == k]
            flight_links = self.flight_data[self.flight_data[k] < 1000000]
            for tc in self.TC[k]:
                arcs = []
                # Ground Links:
                for i in range(len(ground_links)):
                    t1 = ground_links.iloc[i][1]
                    t2 = ground_links.iloc[i][2]
                    if t1 < tc < t2:
                        arcs.append(list(ground_links.index)[i])
                    elif t1 > t2 and (tc > t1 or tc < t2):
                        arcs.append(list(ground_links.index)[i])
                # Flight Arcs:
                for i in range(len(flight_links)):
                    t1 = flight_links.iloc[i][3]
                    t2 = flight_links.iloc[i][4]
                    if t1 < tc < t2:
                        arcs.append(flight_links.iloc[i][0])
                    elif t1 > t2 and (tc > t1 or tc < t2):
                        arcs.append(flight_links.iloc[i][0])
                self.NG[k, tc] = arcs

        print(self.NG)

if __name__ == "__main__":
    parameters = Parameters()

