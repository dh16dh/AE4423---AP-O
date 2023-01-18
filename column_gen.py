# Import relevant modules
import numpy as np
import os
import pandas as pd
import time
from gurobipy import Model,GRB,LinExpr,Column

model.update()

# Solve
model.setParam('MIPGap',0.01)
model.setParam('TimeLimit',6*3600)
model.setParam('MIPFocus',3)
model.setParam('Presolve',1)
model.setParam('Heuristics',0.05)
model.setParam('Cuts',2)
model.setParam('FlowCoverCuts',2) 

exit_condition = False
cont           = 0

while exit_condition is False and cont<100:

   
    linear_relaxation = model.relax()
    linear_relaxation.optimize()
    linear_relaxation.setParam('OutputFlag', 0)