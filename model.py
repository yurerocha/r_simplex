import numpy as np
import utils as u
import re
import pulp
import data as dt

# ---------------------------------------------------------------------------- #
# Class:
class Model:
    def __init__(self):
        self.obj = 0
        self.variables = dict()
        self.c = np.array([], dtype=np.double)
        self.A = np.array([], dtype=np.double)
        self.b = np.array([], dtype=np.double)
        self.m = 0 # Number of constraints.
        self.n = 0 # Number of variables.
        self.constrs = np.array([], dtype=str)
        self.bounds = np.array([], dtype=str)
        self.missing = set()

    def __str__(self):
        a = f'Model \nobj={self.obj}\nx = [1..{self.n}]\nc = {self.c}\n'
        b = f'A = \n{self.A}\nb = {self.b}\nm = {self.m}\nn = {self.n}\n'
        c = f'symbols = {self.constrs}\nEnd Model\n'
        return a + b + c

    def read(self, filename):
        lp_vars, self.lp = pulp.LpProblem.fromMPS(filename)
        self.obj = self.lp.sense

        self.variables, self.lp.constraints = dt.populate_vars(lp_vars, self.lp.constraints)
        self.m = len(self.lp.constraints)
        self.n = len(self.variables)
        self.c = dt.populate_c(self.lp.objective, self.variables)
        self.A, self.b = dt.populateAb(self.m, self.n, self.variables, self.lp.constraints)
    
    def to_standard(self):
        """Puts the problem in the stantard form, while populating the base, if
        possible.
        """
        constrs = self.lp.constraints
        # First, substitute unbounded x for xa - xb (x', x'' >= 0).
        # newc = np.array([])
        # for k in self.c:
        #     newc = np.append(newc, [k, -k])
        
        # A = np.array(u.col(self.A, 0))
        # A = np.hstack((A, -u.col(self.A, 0)))
        # for j in range(1, self.n):
        #     A = np.hstack((A, u.col(self.A, j)))
        #     A = np.hstack((A, -u.col(self.A, j)))

        # Second, add slack or surplus variables, when required.
        Bc = np.array([], dtype=int)
        missing = set([j for j in range(self.m)]) # Keep track of missing columns in
                                            # initial base.
        # self.m, self.n = self.A.shape
        i = 0
        for k in constrs:
            c = 0.0
            if constrs[k].sense == pulp.LpConstraintLE:
                c = 1.0
                # Create initial base.
                Bc = np.append(Bc, self.n)
                missing.remove(i)
            elif constrs[k].sense == pulp.LpConstraintGE:
                c = -1.0
            if c:
                v = np.full(shape=(self.m, 1), fill_value=0.0)
                v[i] = c
                self.A = np.hstack([self.A, v])
                self.c = np.append(self.c, 0.0)
                self.n += 1
            i += 1
        self.missing = missing
        return Bc
    
    def add_y(self, Bc):
        m = 1.0
        if self.obj == pulp.LpMaximize:
            m = -1.0
        for i in self.missing:
            self.c = np.append(self.c, m * u.__M__)
            Aj = np.full(shape=(self.m, 1), fill_value=0)
            Aj[i] = 1
            self.A = np.hstack([self.A, Aj])
            Bc = np.append(Bc, self.n)
            self.n += 1
        return Bc
    
    def solve(self):
        # For min problems.
        Bc = self.to_standard()
        # Not enough columns in the base.
        if len(Bc) != self.m:
            Bc = self.add_y(Bc)
            
        # if self.obj == pulp.LpMinimize:
        #     self.c = [-1 * c for c in self.c]
        B, CB = u.getBNCB(self.A, self.c, Bc)
        XB = u.compXB(B, self.b)
        Nc = u.getNc(self.n, Bc)
        
        it = 0
        while True:
            # print(f'-------------------- It %d --------------------' %(self.it))
            it += 1

            Binv = np.linalg.inv(B)

            # print('Bc =', Bc)
            # print('Nc =', Nc)
            # print('B =', B)
            # print("B' =", Binv)
            # print("b =", self.b)

            # Step 2: Calcular variáveis duais pi resolvendo o sistema piB = CB.
            pi = np.matmul(CB.transpose(), Binv)
            # print('pi =', pi)

            # Step 3: Calcular os custos reduzidos das variáveis não básicas xj 
            # (pricing): cj_bar = cj − piAj
            sk, jin = u.pricing(Nc, self.A, pi, self.c)

            obj = u.obj(CB, XB)        
            if u.isg(sk, 0.0) or u.iseq(sk, 0.0):
                # if self.obj == pulp.LpMinimize:
                #     obj = -obj
                u.print_sol(obj, Bc, XB, pi)
                print("OPTIMAL")
                # u.perform_sa(XB, B, Bc, self.m)
                return
            
            Aj = self.A[:,[Nc[jin]]]
        
            # Step 4: Resolva o sistema B.d = A j onde A j é a coluna da variável j 
            # escolhida para entrar na base.
            y = np.matmul(Binv, Aj)
            # print('y =', y)

            # Step 5: Determine o maior t tal que x B − t.d ≥ 0.
            # Uma das variáveis que limitaram o crescimento de t deve ser escolhida 
            # para sair da base.
            iout = u.quocient_test(XB, y)
            if iout < 0:
                print('UNBOUNDED')
                return
            
            # Step 6: Atualizar B e xB. Depois voltar ao passo 2.
            # print('<-:', Nc[jin])
            # print('->:', Bc[iout])
            Bc[iout], Nc[jin] = Nc[jin], Bc[iout]

            B, CB = u.getBNCB(self.A, self.c, Bc)
            XB = u.compXB(B, self.b)
            # print('Obj =', obj)
            # input()

# Tratar variáveis livres.
# Printar dual do dual e verificar se é igual ao primal.