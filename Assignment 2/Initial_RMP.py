import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gurobipy import Model, GRB, LinExpr, quicksum
from Parameters import Parameters


class initial_RMP:
    def __init__(self):
        self.parameter_set = Parameters()

        # Define Sets
        self.N =
        self.K = 0
        self.F = 0
        self.Gk = 0
        self.P = 0
        self.NG = 0
        self.O = 0
        self.I = 0
        self.npos = 0
        self.nneg = 0
