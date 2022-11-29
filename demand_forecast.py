import pandas as pd

airport_info = pd.read_csv("Groups_data/Group_17_Airport_info.csv", index_col=1)
demand_2020 = pd.read_csv("Groups_data/Group_17_Demand.csv", index_col=0)

f = 1.42  # USD/gallon

pop_list_2020 = airport_info["Population"]
gdp_list_2020 = airport_info["GDP"]

log(D) = log(k) + b1*log(pipj) + b2*log(gigj) - b3*log(fd)