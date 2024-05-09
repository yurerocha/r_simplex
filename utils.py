import numpy as np

# ---------------------------------------------------------------------------- #
# Constants:
__max__ = 'maximize'
__st__ = 'subject to'
__bounds__ = 'bounds'
__end__ = 'end'
__nb_header__= 3
__nb_footer__ = 3

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
    return int(ci[0]), int(ci[1]) - 1

def col(V, i):
    """Returns ith column of vector V.
    """
    return V[:, [i]]

def comp_max_t(XB, d):
    """Returns max t such that xB − t.d >= 0.
    """
    # Calcular os valores
    t = np.array([])
    t_i = np.array([], dtype=int)
    for i, x in enumerate(XB):
        div = x / d[i][0]
        if div >= 0:
            t = np.append(t, div)
            t_i = np.append(t_i, i)

    if len(t) == 0:
        # Unbounded
        return -1

    # Pegar o valor maior dos que sobraram
    v = min(t_i, key=lambda i: t[i])
    return v

def pricing(Nv, A, pi, c):
    """ Computes reduced costs of non-basic variables xj: cj_bar = cj − piAj """
    cbar = np.array([])
    jmax = 0
    for i, j in enumerate(Nv):
        Aj = np.array(col(A, j))
        cbar = np.append(cbar, c[j] - np.matmul(pi, Aj))
        if cbar[i] > cbar[jmax]:
            jmax = i
    print(cbar)
    return cbar[jmax], jmax

# def comp_max_t(self, XB, d):
#     # Calcular os valores
#     t = np.array([])
#     for i, x in enumerate(XB):
#         div = x / d[i][0]
#         if div >= 0:
#             t = np.append(t, {'i': i, 't': div})
    
#     print('t = ', t)

#     # Testar nas outras restrições pra ver quais dão problema e descartá-los
#     f = np.array([])
#     for i, v in enumerate(t):
#         is_ok = True
#         for j, x in enumerate(XB):
#             if x-v['t']*d[j][0] < 0:
#                 is_ok = False
#                 break
#         if is_ok:
#             f = np.append(f, v)

#     if len(f) == 0:
#         print('Unbounded')

#     # Pegar o valor maior dos que sobraram
#     print('f = ', f)
#     max_t_i = f[0]['i']
#     for i in range(1, len(f)):
#         v = f[i]
#         if v['t'] > f[max_t_i]['t']:
#             max_t_i = v['i']

#     print('max_t_i = ', max_t_i)
    
#     return max_t_i