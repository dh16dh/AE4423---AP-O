# Import relevant modules
import pandas as pd
import plotly.graph_objects as go


flight_data = pd.read_csv('Assignment 2/Group_data_17/Flight_data.csv', sep=';')
database = pd.read_csv('Assignment 2/AirportDB.csv', index_col=0)
ICAO = flight_data['ORG'].unique()
ICAO_long = []
ICAO_lat = []



for i in ICAO:
    ICAO_long.append(database['longitude'].loc[i])
    ICAO_lat.append(database['latitude'].loc[i])


def plot_routes(flight_data, database, ICAO_long, ICAO_lat):
    # airport_data = self.parameter_set.airport_data
    fig = go.Figure(data=go.Scattergeo())

    # Europe
    fig.update_geos(showcountries=True,
                    showsubunits=True,
                    lataxis_range=[35, 60],
                    lonaxis_range=[-11, 30],
                    resolution=50)
    
    # World
    # fig.update_geos(showcountries=True,
    #                 showsubunits=True,
    #                 lataxis_range=[-40, 75],
    #                 lonaxis_range=[-130, 150],
    #                 resolution=50)

    # Add Routes
    for i in range(len(flight_data.index)):
        fig.add_trace(go.Scattergeo(
            lon=[database['longitude'].loc[flight_data['ORG'].loc[i]],
                    database['longitude'].loc[flight_data['DEST'].loc[i]]],
            lat=[database['latitude'].loc[flight_data['ORG'].loc[i]],
                    database['latitude'].loc[flight_data['DEST'].loc[i]]],
            mode='lines',
            line=dict(
                width=1,
                color='red'
            )
        ))

    #Add Airports:
    fig.add_trace(go.Scattergeo(
        lon=ICAO_long,
        lat=ICAO_lat,
        #text=airport_data['City Name'],
        mode='markers',
        marker=dict(
            size=5,
            color='red'
        )
    ))
    fig.add_trace(go.Scattergeo(
        lon=[database['longitude'].loc['LIMC'], database['longitude'].loc['LIME']],
        lat=[database['latitude'].loc['LIMC'], database['latitude'].loc['LIME']],
        #text=airport_data['City Name']['LIRA'],
        mode='markers',
        marker=dict(
            size=10,
            color='blue'
        )
    ))
    fig.update_layout(showlegend=False)
    #fig.write_image('Network Map.svg')
    fig.show()

plot_routes(flight_data, database, ICAO_long, ICAO_lat)


