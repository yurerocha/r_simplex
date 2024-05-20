#!/bin/python3

from model import Model
import sys

if len(sys.argv) < 2:
    print('Not enough arguments. Expected:\n./run filename')
    exit()

m = Model()

# m.read('input1.lp')
m.read(sys.argv[1])

# m.to_dual()

m.solve()
