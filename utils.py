import numpy as np
import re

# ---------------------------------------------------------------------------- #
# Constants:
__max__ = 'max'
__min__ = 'min'
__st__ = 'subject to'
__bounds__ = 'bounds'
__end__ = 'end'
__nb_header__= 3
__nb_footer__ = 3
__inf__ = 10**3

# ---------------------------------------------------------------------------- #
# Helper functions:
def obj(CB, XB):
    """Returns the objective value based on the cost coefficients and the x 
    values in the base.
    """
    return np.dot(CB.transpose(), XB)
def get_coef_and_x(cxi:str):
    """Returns coefficient (c) and x variable index (i - 1).
    """
    ci = cxi.split('x')
    return float(ci[0]), int(ci[1]) - 1

def col(V, i):
    """Returns ith column of vector V.
    """
    return V[:, [i]]

def comp_max_t(XB, d):
    """Returns max t such that xB − t.d >= 0.
    """
    t = np.array([])
    # t_i = np.array([], dtype=int)
    for i, x in enumerate(XB):
        div = x / d[i][0]
        if div >= 0:
            t = np.append(t, {'i': i, 't': div})
            # t_i = np.append(t_i, i)
    if len(t) == 0:
        # Unbounded
        return -1
    # Pegar o valor maior dos que sobraram.
    v = min(t, key=lambda e: e['t'])
    return v['i']

def pricing(Nc, A, pi, c):
    """ Computes reduced costs of non-basic variables xj: cj_bar = cj − piAj """
    cbar = np.array([])
    jmax = 0
    for i, j in enumerate(Nc):
        Aj = np.array(col(A, j))
        cbar = np.append(cbar, c[j] - np.matmul(pi, Aj))
        if cbar[i] > cbar[jmax]:
            jmax = i
    return cbar[jmax], jmax

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
    return np.linalg.solve(B, b)

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