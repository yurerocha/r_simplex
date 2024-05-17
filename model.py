import numpy as np
import utils as u
import re

# ---------------------------------------------------------------------------- #
# Class:
class Model:
    def __init__(self):
        self.obj = ''
        self.c = np.array([], dtype=float)
        self.A = np.array([], dtype=float)
        self.b = np.array([], dtype=float)
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
    
    def populate_c(self, obj:str):
        obj = obj.replace('\n', '')
        obj = obj.replace(' ', '')
        obj = obj.split('+')
        # Extract all cost coefficients.
        for c in obj:
            x, _ = u.get_coef_and_x(c)
            self.c = np.append(self.c, float(x))
        self.n = len(obj)
    
    def populateAb(self, lines):
        # self.A = np.array([], dtype=int)
        for i, l in enumerate(lines):
            if u.__bounds__ in l.lower():
                break
            self.m += 1
            if len(self.A):
                self.A = np.vstack(
                             [self.A, np.empty(shape=(1, self.n), dtype=float)])
            else:
                self.A = np.empty(shape=(1, self.n), dtype=float)
            # Rm unnecessary blank spaces.
            l = l.replace(' ', '')
            l = l.replace('\n', '')
            # Get the first and only element returned in the list.
            [symb] = re.findall(r'<=|>=|=', l)
            self.constrs = np.append(self.constrs, symb)
            cx, b = l.split(symb)
            self.b = np.append(self.b, float(b))
            cx = cx.split('+')
            # Populate A.
            for cxj in cx:
                x, j = u.get_coef_and_x(cxj)
                self.A[i, j] = x
    
    def populate_bounds(self, lines):
        for l in lines:
            [symb] = re.findall(r'<=|>=|=', l)
            self.bounds = np.append(self.bounds, symb)

    def read(self, filename):
        file = open(filename, 'r')
        lines = file.readlines()

        self.obj = lines[0].replace('\n', '').lower()

        self.populate_c(lines[1])
        self.populateAb(lines[u.__nb_header__:])
        self.populate_bounds(lines[u.__nb_header__ + self.m + 1:-1])

        print('#---------- Model ----------#')
        u.print_model(self.obj, self.A, self.c, self.b, self.constrs, 
                      self.bounds, 'x')

    def to_standard(self):
        """Puts the problem in the stantard form, while populating the base, if
        possible.
        """
        Bc = np.array([], dtype=int)
        [self.missing.add(j) for j in range(self.m)]
        for i, s in enumerate(self.constrs):
            c = 0
            if '<=' in s:
                c = 1
                # Create initial base.
                Bc = np.append(Bc, self.n)
                self.missing.remove(i)
            elif '>=' in s:
                c = -1
            if c:
                v = np.full(shape=(self.m, 1), fill_value=0)
                v[i] = c
                self.A = np.hstack((self.A, v))
                self.c = np.append(self.c, 0)
                self.n += 1
                self.bounds = np.append(self.bounds, '>=')
        return Bc
    
    def add_missing_elements_inB(self, Bc):
        m = 1
        if self.obj in u.__max__:
            m = -1
        for i in self.missing:
            self.c = np.append(self.c, m * u.__inf__)
            Aj = np.full(shape=(self.m, 1), fill_value=0)
            Aj[i] = 1
            self.A = np.hstack([self.A, Aj])
            Bc = np.append(Bc, self.n)
            self.n += 1
        return Bc

    def to_dual(self):
        """Computes the dual problem.
        """
        C = self.b
        A = self.A.transpose()
        b = self.c
        constrs = np.array([])
        bounds = np.array([])
        if u.__max__ in self.obj:
            obj = u.__min__
            constrs = self.bounds
            # Troca os sinais das restrições nos bounds.
            bounds = u.change_signs(self.constrs)
        else:
            obj = u.__max__
            # Troca os sinais dos bounds nas restrições.
            bounds = self.constrs
            constrs = u.change_signs(self.bounds)
        
        print('#---------- Dual  ----------#')
        u.print_model(obj, A, C, b, constrs, bounds, 'y')
    
    def steps2to5(self, B, Bc, Nc, CB, XB):
        # Step 2: Calcular variáveis duais pi resolvendo o sistema piB = CB.
        pi = np.linalg.solve(B.transpose(), CB)

        # Step 3: Calcular os custos reduzidos das variáveis não básicas xj 
        # (pricing): cj_bar = cj − piAj
        cjbar, jin = u.pricing(Nc, self.A, pi, self.c)
        
        obj = u.obj(CB, XB)
        if u.__min__ in self.obj:
            obj = -obj
        u.print_sol(obj, Bc, XB, pi)
        # print('cjbar = ', cjbar)
        if u.isl(cjbar, 0.0):
            print("OPTIMAL")
            u.perform_sa(XB, B, Bc, self.m)
            return
        Aj = self.A[:,[Nc[jin]]]
    
        # Step 4: Resolva o sistema B.d = A j onde A j é a coluna da variável j 
        # escolhida para entrar na base.
        d = np.linalg.solve(B, Aj)

        # Step 5: Determine o maior t tal que x B − t.d ≥ 0. 
        # Se t for ilimitado, o PL também é ilimitado (VOLTAR AQUI DEPOIS).
        # Uma das variáveis que limitaram o crescimento de t deve ser escolhida 
        # para sair da base.
        iout = u.comp_max_t(XB, d)
        if iout < 0:
            print('UNBOUNDED')
            return
        
        # Step 6: Atualizar B e xB. Depois voltar ao passo 2.
        print('<-:', Nc[jin])
        print('->:', Bc[iout])
        Bc[iout], Nc[jin] = Nc[jin], Bc[iout]

        B, CB = u.getBNCB(self.A, self.c, Bc)
        XB = u.compXB(B, self.b)
        # print(B)
        self.steps2to5(B, Bc, Nc, CB, XB)

    def solve(self):
        # For min problems.
        Bc = self.to_standard()
        print('#---------- Stand ----------#')
        u.print_model(self.obj, self.A, self.c, self.b, 
                      ['=' for i in range(self.m)], self.bounds, 'x')
        # Not enough columns in the base.
        if len(Bc) != self.m:
            Bc = self.add_missing_elements_inB(Bc)
            
        if u.__min__ in self.obj:
            self.c = [-1 * c for c in self.c]
        B, CB = u.getBNCB(self.A, self.c, Bc)
        XB = u.compXB(B, self.b)
        Nc = u.getNc(self.n, Bc)
        self.steps2to5(B, Bc, Nc, CB, XB)

# Tratar variáveis livres.
# Printar dual do dual e verificar se é igual ao primal.