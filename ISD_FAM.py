# Import relevant modules
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from gurobipy import Model, GRB, Column, LinExpr, quicksum
from Parameters import Parameters
import time

pd.set_option('display.max_columns', None)


class ISD_FAM:
    def __init__(self):
        self.parameter_set = Parameters()
        self.parameter_set.ISD_FAM()

        # Define Sets
        self.N = self.parameter_set.N  # set of nodes  [k]
        self.K = self.parameter_set.K  # set of aircraft types
        self.L = self.parameter_set.L  # set of flights
        self.L_O = self.parameter_set.L_O  # set of optional flights
        self.L_F = self.parameter_set.L_F  # set of mandatory flights
        self.Lq = self.parameter_set.Lq  # set of flights in itinerary q  [q]
        self.P = self.parameter_set.P  # set of all passenger itineraries (paths)
        self.P_O = self.parameter_set.P_O  # set of itineraries containing optional flights
        self.G = self.parameter_set.G  # set of ground arcs  [k]
        self.TC = self.parameter_set.TC  # set of unique time cuts   [k]
        self.NG = self.parameter_set.NG  # set of flight and ground arcs intercepted by the time cut  [k, tc]
        self.O = self.parameter_set.O  # flight arcs originating at node n in fleet k   [k, n]
        self.I = self.parameter_set.I  # flight arcs terminating at node n in fleet k   [k, n]
        self.n_plus = self.parameter_set.n_plus  # ground arcs originating at any node n    n+[k, n]
        self.n_min = self.parameter_set.n_minus  # ground arcs terminating at any node n   n-[k, n]

        self.ac = self.parameter_set.AC  # number of aircraft in fleet of type k   [k]
        self.cost = self.parameter_set.c  # operating cost of AC type k for flight i   .loc[i, k]
        self.s = self.parameter_set.s  # number of seats for aircraft type k   [k]
        self.fare = self.parameter_set.fare  # average fare for itinerary p   [p]

        self.D = self.parameter_set.D  # daily unconstrained demand for itinerary p   [p]
        self.Q = self.parameter_set.Q  # daily unconstrained demand on flight (leg) i   .loc[i]
        self.Nq = self.parameter_set.Nq  # number of flights in itinerary q [q]

        self.b = self.parameter_set.b  # recapture rate of a pax that desired itinerary p and is allocated to r .loc[p, r]
        self.delta = self.parameter_set.delta  # if flight i is in itinerary p [i, p]

        self.f = {}
        self.y = {}
        self.t = {}
        self.Z = {}

        self.F_in_P = self.parameter_set.Lp
        self.Optional_P = self.parameter_set.optional_itinerary_df

        self.C1_F = {}
        self.C1_O = {}
        self.C2 = {}
        self.C3 = {}
        self.C4 = {}
        self.C5 = {}
        self.C6 = {}
        self.C7 = {}

    def isd_fam_model(self):
        # Initialise gurobipy model
        model = Model("ISD_FAM")

        # Define Decision Variables
        self.f = {}  # 1 if flight arc i is assigned to aircraft type k, 0 otherwise [i, k]
        self.y = {}  # number of aircraft of type k on the ground arc a [a, k]
        self.t = {}  # number of passengers that would like to travel on itinerary p and are reallocated to itin. r [p, r]
        self.Z = {}  # 1 if itinerary q is in flight schedule, 0 otherwise [q]

        # Add Variables to Objective Function
        for i in self.L:
            for k in self.K:
                self.f[i, k] = model.addVar(vtype=GRB.BINARY, name=f'f-{i}-{k}')
                self.y[i, k] = model.addVar(ub=0, vtype=GRB.INTEGER, name=f'y-{i}-{k}')

        for p in self.P:
            self.t[p, 9999] = model.addVar(vtype=GRB.INTEGER, name=f't-{p}-{9999}')

        for q in self.P_O:
            self.Z[q] = model.addVar(vtype=GRB.BINARY, name=f'Z-{q}')

        for k in self.K:
            for a in self.G[k]:
                self.y[a, k] = model.addVar(vtype=GRB.INTEGER, name=f'y-{a}-{k}')
                self.f[a, k] = model.addVar(ub=0, vtype=GRB.BINARY, name=f'f-{a}-{k}')

        model.update()

        obj = LinExpr()
        for i in self.L:
            for k in self.K:
                obj += self.cost.loc[i, k] * self.f[i, k]
        for p in self.P:
            obj += (self.fare[p] - (float(self.b.loc[p, 9999]) * self.fare[9999])) * self.t[p, 9999]
        for q in self.P_O:
            obj += self.fare[q] * self.D[q]
            obj += - self.fare[q] * self.D[q] * self.Z[q]

        model.setObjective(obj, GRB.MINIMIZE)
        model.update()
        print("Objective Function Defined")

        # Define Constraints
        for i in self.L_F:
            self.C1_F[i] = model.addConstr(quicksum(self.f[i, k] for k in self.K) == 1, name=f'C1.F-{i}')
        for i in self.L_O:
            self.C1_O[i] = model.addConstr(quicksum(self.f[i, k] for k in self.K) <= 1, name=f'C1.O-{i}')
        print("Added Constraint: C1")
        for k in self.K:
            for n in self.N[k]:
                self.C2[k, n] = model.addConstr(
                    self.y[self.n_plus[k, n], k] + quicksum(self.f[i, k] for i in self.O[k, n]) - self.y[
                        self.n_min[k, n], k] - quicksum(
                        self.f[i, k] for i in self.I[k, n]) == 0, name=f'C2-{k}-{n}')
        print("Added Constraint: C2")
        for k in self.K:
            self.C3[k] = model.addConstr(
                quicksum(self.y[a, k] + self.f[a, k] for a in self.NG[k, self.TC[k][0]]) <= self.ac[k],
                name=f'C3-{k}')
        print("Added Constraint: C3")
        for i in self.L:
            self.C4[i] = model.addConstr(quicksum(int(self.s[k]) * self.f[i, k] for k in self.K) +
                                         quicksum(self.delta[i, p] * self.t[p, 9999] for p in self.P) >= int(
                self.Q.loc[i]),
                                         name=f'C4-{i}')
        print("Added Constraint: C4")
        for p in self.P:
            self.C5[p] = model.addConstr(self.t[p, 9999] <= self.D[p], name=f'C5-{p}')
        print("Added Constraint: C5")
        for q in self.P_O:
            for i in self.Lq[q]:
                self.C6[q, i] = model.addConstr(self.Z[q] - quicksum(self.f[i, k] for k in self.K) <= 0, name=f'C6-{q}-{i}')
        for q in self.P_O:
            self.C7[q] = model.addConstr(self.Z[q] - quicksum(quicksum(self.f[i, k] for i in self.Lq[q]) for k in self.K) <= 0, name=f'C7-{q}')

        print("Added Constraint: C6 & C7")

        model.write('model_isd_fam.lp')

        model.update()

        return model

    def column_generation(self, cols_to_add=None, iterations=20):
        model = self.isd_fam_model()

        exit_condition = False
        it_num = 0

        results_col_gen = []

        while exit_condition is False and it_num < iterations:

            model_start_time = time.time()

            model_relax = model.relax()
            model_relax.optimize()

            model_runtime = time.time() - model_start_time
            print('\nModel Runtime: \t %g' % model_runtime)

            column_gen_start_time = time.time()

            pi_list = dict()
            sigma_list = dict()
            for i in self.L:
                pi = model_relax.getConstrByName(f'C4-{i}').Pi
                pi_list[i] = pi
            for p in self.P:
                sigma = model_relax.getConstrByName(f'C5-{p}').Pi
                sigma_list[p] = sigma

            blacklist = []

            cpr = dict()
            for p in self.P:
                blacklist.append((p, 9999))
            for p in self.P:
                for r in self.P:
                    if p == r or (p, r) in blacklist:
                        continue
                    value = (self.fare[p] - sum([pi_list[i] for i in self.F_in_P[p]])) - float(self.b.loc[p, r]) * (
                            self.fare[r] - sum([pi_list[j] for j in self.F_in_P[r]])) - sigma_list[p]
                    if value < -0.0001:
                        cpr[p, r] = value

            cpr = sorted(cpr.items(), key=lambda x: x[1])
            if cols_to_add is not None:
                cols_to_add = min(cols_to_add, len(cpr))
            else:
                cols_to_add = len(cpr)

            print('Iteration Number', it_num)
            print('Column(s) Added')
            columns_added = []
            for c in range(cols_to_add):
                p, r = cpr[c][0]
                col_pr = Column()
                col_rp = Column()
                # Adding to C4
                for i in self.L:
                    coef_pr = self.delta[i, p]
                    constr_pr = self.C4[i]
                    col_pr.addTerms(coef_pr, constr_pr)
                    coef_rp = -self.delta[i, p] * float(self.b.loc[r, p])
                    constr_rp = self.C4[i]
                    col_rp.addTerms(coef_rp, constr_rp)
                # Adding to C5
                coef_pr = float(1)
                constr_pr = self.C5[p]
                col_pr.addTerms(coef_pr, constr_pr)
                coef_rp = float(1)
                constr_rp = self.C5[r]
                col_rp.addTerms(coef_rp, constr_rp)
                obj_coef_pr = self.fare[p] - float(self.b.loc[p, r]) * self.fare[r]
                obj_coef_rp = self.fare[r] - float(self.b.loc[r, p]) * self.fare[p]
                self.t[p, r] = model.addVar(obj=obj_coef_pr, vtype=GRB.INTEGER, name=f't-{p}-{r}', column=col_pr)
                self.t[r, p] = model.addVar(obj=obj_coef_rp, vtype=GRB.INTEGER, name=f't-{r}-{p}', column=col_rp)
                blacklist.append((p, r))
                blacklist.append((r, p))
                model.update()
                print('Column', c, 't[', p, ',', r, ']')
                columns_added.append((p, r))

            column_gen_runtime = time.time() - column_gen_start_time

            new_iteration = {'Iteration Number': it_num,
                             'Model Runtime': model_runtime,
                             'ColGen Runtime': column_gen_runtime,
                             'Columns Added': columns_added}
            results_col_gen.append(new_iteration)
            if len(cpr) == 0:
                exit_condition = True
            it_num += 1

        model.optimize()

        status = model.status

        if status == GRB.Status.UNBOUNDED:
            print('The model cannot be solved because it is unbounded')

        elif status == GRB.Status.OPTIMAL or True:
            f_objective = model.objVal
            print('***** RESULTS ******')
            print('\nObjective Function Value: \t %g' % f_objective)

        elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
            print('Optimization was stopped with status %d' % status)

        spilled_pax = []
        for p in self.P:
            spilled_pax.append(self.t[p, 9999].X)

        total_spilled = sum(spilled_pax)

        t_list = []
        i = 0
        j = 0
        while len(t_list) < 5:
            p = self.P[i]
            r = self.P[j]
            if r == 9999:
                i += 1
                j = 0
                continue
            try:
                var = self.t[p, r].X
                if var > 0:
                    t_list.append({'From Itinerary': p,
                                   'To Itinerary': r,
                                   'Value': var})
            except KeyError as ke:
                pass
            j += 1

        t_df = pd.DataFrame(t_list)

        f_list = []
        h = 0
        j = 0
        while len(f_list) < 5:
            i = self.L[h]
            k = self.K[j]
            var = self.f[i, k].X
            if var > 0:
                f_list.append({'Flight Number': i,
                               'Aircraft Type': k})
            j += 1
            if k == 'BUS':
                h += 1
                j = 0

        f_df = pd.DataFrame(f_list)

        print('Total Spilled Passengers', total_spilled)
        print(t_df)
        print(f_df)

        itineraries_added = []
        itineraries_excluded = []
        for q in self.P_O:
            flight = self.Optional_P.loc[q]
            new_info = {'Itinerary': q,
                        'Origin': flight['Origin'],
                        'Destination': flight['Destination'],
                        'Leg 1': flight['Leg 1'],
                        'Leg 2': flight['Leg 2']}
            if self.Z[q].X > 0:
                itineraries_added.append(new_info)
            else:
                itineraries_excluded.append(new_info)

        included_df = pd.DataFrame(itineraries_added)
        excluded_df = pd.DataFrame(itineraries_excluded)

        print('Itineraries Included')
        print(included_df)
        print('Itineraries Excluded')
        print(excluded_df)

        return pd.DataFrame(results_col_gen).set_index('Iteration Number')


if __name__ == '__main__':
    # RMP().get_rmp_results()
    column_gen_results = ISD_FAM().column_generation()
    # column_gen_results.to_csv('ColumnGenerationKPIs.csv')
