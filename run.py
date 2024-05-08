#!/bin/python3

from model import Model

m = Model()

m.read('input1.lp')

print(str(m))

m.solve()
