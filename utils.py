import numpy as np
import re
import pulp

# ---------------------------------------------------------------------------- #
# Constants:
__max__ = 'max'
__min__ = 'min'
__st__ = 'subject to'
__bounds__ = 'variables'
__end__ = 'end'
__nb_header__= 3
__nb_footer__ = 3
__inf__ = 10**7
__eps__ = 1e-9

# ---------------------------------------------------------------------------- #
# Helper functions:
def iseq(a, b):
    return abs(a - b) < __eps__

def isl(a, b):
    return a < b - __eps__

def isg(a, b):
    return a > b + __eps__

def obj(CB, XB):
    """Returns the objective value based on the cost coefficients and the x 
    values in the base.
    """
    return np.dot(CB.transpose(), XB)

def get_coef_and_x(cxi:str):
    """Returns coefficient (c) and x variable index (i - 1).
    """
    ci = cxi.split('X')
    return float(ci[0]), int(ci[1]) - 1

def col(V, i):
    """Returns ith column of vector V.
    """
    return V[:, [i]]

def quocient_test(XB, y):
    """Returns max t such that xB − t.d >= 0.
    """
    t = np.array([])
    is_init = False
    min_q = 0.0
    min_i = -1
    # t_i = np.array([], dtype=int)
    for i, x in enumerate(XB):
        if isg(y[i][0], 0.0):
            q = x / y[i][0]
            if not is_init or isl(q, min_q):
                min_q = q
                min_i = i
                is_init = True
    return min_i

def pricing(Nc, A, pi, c):
    """ Computes reduced costs of non-basic variables xj: cj_bar = cj − piAj """
    s = np.array([])
    s_k = c[0] - np.matmul(pi.transpose(), np.array(col(A, Nc[0])))
    min_i = 0
    for i, j in enumerate(Nc):
        Aj = np.array(col(A, j))
        s = c[j] - np.matmul(pi.transpose(), Aj)
        if isl(s, s_k):
            s_k = s
            min_i = i
    return s_k, min_i

def getBNCB(A, c, Bc):
    """Returns B matrix and CB row vector, based on the selected basic columns 
    Bc.
    """
    B = np.array(col(A, Bc[0]))
    CB = np.array([c[Bc[0]]])
    for i in range(1, len(Bc)):
        B = np.hstack([B, col(A, Bc[i])])
        CB = np.append(CB, c[Bc[i]])
    return B, CB

def compXB(B, b):
    """Solves B.XB = b >= 0.
    """
    return np.matmul(np.linalg.inv(B), b)

def getNc(n, Bc):
    """Returns row vector of nonbasic columns Nc.
    """
    Nc = np.array([], dtype=int)
    for i in range(n):
        if i not in Bc:
            Nc = np.append(Nc, i)
    return Nc

def print_model(obj, A, C, b, constrs, bounds, var):
    """Prints the model.
    """
    print(obj, end='\n\t')
    for i, c in enumerate(C):
        sign = ' + '
        if i == len(C) - 1:
            sign = ' '
        print(str(c) + var + str(i+1) + sign, end='')
    print('\nsubject to')
    for i, ai in enumerate(A):
        print(end='\t')
        for j, aij in enumerate(ai):
            sign = ' + '
            if j == len(ai) - 1:
                sign = ' '
            print(str(aij) + var + str(j+1) + sign, end='')
        print(constrs[i] + ' ' + str(b[i]))
    print('bounds')
    for j, bd in enumerate(bounds):
        print('\t' + var  + str(j+1) + ' ' + bd + ' 0' )
    print('end')

def change_signs(items):
    """Changes '>' for '<' and '<' for '>'.
    """
    new_items = np.array([])
    for it in items:
        if '<' in it:
            it = re.sub(r'<', '>', it)
        elif '>' in it:
            it = re.sub(r'>', '<', it)
        new_items = np.append(new_items, it)
    return new_items

def print_sol(obj, Bc, XB, pi):
    """Prints primal and dual solutions.
    """
    print('Obj = ', obj)
    # for j, x in enumerate(Bc):
    #     print('x' + str(x + 1) + ' = ' + str(XB[j]) + ' ')
    # print('Dual')
    # for j in range(len(pi)):
    #     print('y' + str(j + 1) + ' = ' + str(pi[j]) + ' ')

def perform_sa(XB, B, Bc, m):
    """Performs sensitivity analysis on the values of the variables in the
    optimal solution.
    """
    Binv = np.linalg.inv(B)
    for j in range(m):
        c = col(Binv, j)
        mi = np.array([])
        ma = np.array([])
        for i in range(m):
            if not iseq(c[i], 0.0):
                # print(XB[i], '+ delta *', c[i], '>= 0')
                v = -XB[i] / c[i][0]
                if isl(v, 0):
                    mi = np.append(mi, v)
                else:
                    ma = np.append(ma, v)
        x = 'b' + str(Bc[j]+1)
        if mi.any():
            print(x, '>=', max(mi))
        if ma.any():
            print(x, '<=', min(ma))

def update_constraints(constrs):
    """The two-phase dual simplex method: 
        2. Change the prob­lem so that b >= O.
    """   
    for k in constrs:
        # Pulp considers the constraint is in the lhs which means that negative 
        # constants are positive in the lhs.
        if isg(constrs[k].constant, 0.0):
            constrs[k] = -1*constrs[k]
            constrs[k].sense = -1*constrs[k].sense

    return constrs

def vars_bounds_as_constrs(vars, constrs):
    ilb = 0
    iub = 0
    for k in vars:
        lb = vars[k].getLb()
        ub = vars[k].getUb()
        print(lb, ub)
        if isg(lb, 0.0):
            constrs['LB' + str(ilb)] = pulp.LpConstraint(vars[k] >= lb)
            # print(vars[k].getLb(), vars[k].getUb())
            ilb += 1
        if ub != None:
            constrs['UB' + str(iub)] = pulp.LpConstraint(vars[k] <= ub)
            # print(vars[k].getLb(), vars[k].getUb())
            iub += 1
    return constrs