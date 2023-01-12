# AE4423 - Airline Planning & Optimization

## About

This repository contains code and other relevant files for the assignments in the 
TU Delft course AE4423 - Airline Planning & Optimisation.

All code thus far is for Assignment 1.

### Data Files
The data provided is in the [Groups_data](Groups_data) folder and contains the following files with the relevant data.

- [Aircraft Data](Groups_data/Aircraft_info.csv)
- [Airport Information](Groups_data/Group_17_Airport_info.csv)
- [Annual Population Growth](Groups_data/Group_17_Annual_growth.csv)
- [Demand Data (2020)](Groups_data/Group_17_Demand.csv)
- [Distance Data](Groups_data/Group_17_Distances.csv)

### Code
There are four Python files that are used in this assignment.
1. [Demand Forecast](demand_forecast.py)
2. [Leg Based Model & Parameter Class](leg_based_model.py)
3. [Route Preprocessing](routes.py)
4. [Route Based Model](route_based_model.py)

### Dependencies
To ensure that all code works, ensure the following packages (and their dependencies) are installed

* ``numpy``
* ``pandas``
* ``gurobipy``
* ``plotly``
* ``kaleido``
* ``statsmodels``
* ``itertools``