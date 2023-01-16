# Import relevant modules
import pandas as pd
import plotly.graph_objects as go


flight_data = pd.read_csv('Assignment 2/Group_data_17/Flight_data.csv', sep=';')
database = pd.read_csv('Assignment 2/AirportDB.csv', index_col=0)


def plot_routes(flight_data, database):
    # airport_data = self.parameter_set.airport_data
    fig = go.Figure(data=go.Scattergeo())

    fig.update_geos(showcountries=True,
                    showsubunits=True,
                    lataxis_range=[36, 48],
                    lonaxis_range=[6, 20],
                    resolution=50)

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

    # Add airports:
    # fig.add_trace(go.Scattergeo(
    #     lon=airport_data['Longitude (deg)'],
    #     lat=airport_data['Latitude (deg)'],
    #     text=airport_data['City Name'],
    #     mode='markers',
    #     marker=dict(
    #         size=5,
    #         color='red'
    #     )
    # ))
    # fig.add_trace(go.Scattergeo(
    #     lon=[airport_data.loc['LIRA']['Longitude (deg)']],
    #     lat=[airport_data.loc['LIRA']['Latitude (deg)']],
    #     text=airport_data['City Name']['LIRA'],
    #     mode='markers',
    #     marker=dict(
    #         size=10,
    #         color='blue'
    #     )
    # ))
    fig.update_layout(showlegend=False)
    #fig.write_image('Network Map.svg')
    fig.show()

plot_routes(flight_data, database)


