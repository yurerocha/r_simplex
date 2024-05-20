#!/bin/python3

import pulp

prob = pulp.LpProblem("input1", pulp.LpMaximize)
x1 = pulp.LpVariable("x_1", 0, pulp.inf)
x2 = pulp.LpVariable("x_2", 0, pulp.inf)
x3 = pulp.LpVariable("x_3", 0, pulp.inf)
x4 = pulp.LpVariable("x_4", 0, pulp.inf)
prob += 19*x1 + 13*x2 + 12*x3 + 17*x4, "obj"
prob += 3*x1 + 2*x2 + 1*x3 + 2*x4 <= 225, "c1"
prob += 1*x1 + 1*x2 + 1*x3 + 1*x4 <= 117, "c2"
prob += 4*x1 + 3*x2 + 3*x3 + 4*x4 <= 420, "c3"

prob.writeMPS("input1.mps")