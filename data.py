import numpy as np
import utils as u
import pulp

def populate_vars(lp_vars, lp_constrs):
    '''Returns the keys of lp_vars as a map between key->index.
    '''
    i = 0
    vars = dict()
    for x in lp_vars:
        vars[x] = i
        i += 1
    return vars, lp_constrs

def populate_c(obj, vars):
    '''Populates with the values in the objective function.
    '''
    c = np.full(shape=len(vars), fill_value=0.0)
    for k in obj:
        c[vars[str(k)]] = obj[k]
    return c

def populateAb(m, n, vars, constrs):
    '''Populates matrix A and vector b.
    '''
    A = np.full(shape=(m, n), fill_value=0.0)
    b = np.array([])
    i = 0
    for k in constrs:
        b = np.append(b, -constrs[k].constant)
        for x in constrs[k]:
            v = constrs[k][x]
            j = vars[str(x)]
            A[i][j] = v
        i += 1
    return A, b
            
