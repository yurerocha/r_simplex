import numpy as np
import utils as u
import re

# ---------------------------------------------------------------------------- #
# Class:
class Model:
    def __init__(self):
        self.c = np.array([], dtype=int)
        self.A = np.array([], dtype=int)
        self.b = np.array([], dtype=int)
        self.m = 0 # Number of constraints.
        self.n = 0 # Number of variables.
        self.symbols = np.array([], dtype=str)
        self.Bv = np.array([], dtype=int)

    def __str__(self):
        a = f'Model \nx = [1..{self.n}]\nc = {self.c}\nA = \n{self.A}\n'
        b = f'b = {self.b}\nm = {self.m}\nn = {self.n}\n'
        c = f'symbols = {self.symbols}\nBv = \n{self.Bv}\nEnd Model\n'
        return a + b + c
    
    def populate_c(self, obj:str):
        # Substitute 'cxj' for ''.
        obj = re.sub(r'x\d+', '', obj)
        # Extract all cost coefficients.
        obj = re.findall(r'\d+', obj)
        for c in obj:
            self.c = np.append(self.c, int(c))

        self.n = len(obj)
    
    def populateAb(self, constrs):
        self.m = len(constrs)
        self.A = np.empty(shape=(self.m, self.n), dtype=int)
        for i, c in enumerate(constrs):
            # Rm unnecessary blank spaces.
            c = c.replace(' ', '')
            c = c.replace('\n', '')
            # Get the first and only element returned in the list.
            [symb] = re.findall(r'<=|>=|=', c)
            self.symbols = np.append(self.symbols, symb)

            cx, b = c.split(symb)
            self.b = np.append(self.b, int(b))

            cx = cx.split('+')
            # Populate A.
            for cxj in cx:
                x, j = u.get_coef_and_x(cxj)
                self.A[i, j] = x
    
    def read2(self, filename):
        file = open(filename, 'r')
        lines = file.readlines()

        self.populate_c(lines[1])
        self.populateAb(lines[u.__nb_header__:len(lines)-u.__nb_footer__])
        self.to_standard()

    def to_standard(self):
        """Puts the problem in the stantard form, while populating the base, if
        possible.
        """
        for i, s in enumerate(self.symbols):
            c = 0
            if '<=' in s:
                c = 1
                # Create initial base.
                self.Bv = np.append(self.Bv, self.n)
            elif '>=' in s:
                c = -1
            if c:
                v = np.full(shape=(self.m, 1), fill_value=0)
                v[i] = c
                self.A = np.hstack((self.A, v))
                self.n += 1

    def read(self, filename):
        file = open(filename, 'r')
        lines = file.readlines()
        self.n = self.comp_nb_x(lines)
        self.make_standard(lines)

        # Parse obj function.
        fo = lines[1].split()
        self.c = self.get_coefs(fo)


        # Parse constraints.
        for constr in self.constraints:
            # Add slack or surplus variable.
            l = len(constr)
       
            bj = int(constr[l-1])
            constr = constr[0:l-2] # Discard symbol and rhs vector
            coefs = self.get_coefs(constr)

            if len(self.A):
                self.A = np.vstack([self.A, coefs])
            else:
                self.A = np.append(self.A, coefs)

            self.b = np.append(self.b, bj)
            
    def get_coefs(self, s):
        coefs = np.full(shape=self.n, fill_value=0, dtype=int)
        for v in s:
            if v != '+' and v != '-':
                c, x = u.get_coef_and_x(v)
                coefs[x] = c
        return coefs
    
    def comp_nb_x(self, lines:list[str]):
        i = len(lines)-2
        x = lines[i].split()
        x = x[0].split('x')[1]
        return int(x)
    
    def make_standard(self, lines):
        i = 3
        self.constraints = np.array([], dtype=str)
        while True:
            if u.__bounds__ in lines[i].lower():
                break
            constr = lines[i].split()

            # Add slack or surplus variable.
            l = len(constr)
            symb = ''
            if '<=' in constr[l-2]:
                symb = '+'
            elif '>=' in constr[l-2]:
                symb = '-'
            
            nc = np.array(constr[0:l-2], dtype=str)
            if len(symb):
                self.n += 1
                nc = np.append(nc, [symb, f'1x{self.n}', '=', constr[-1]])
            else:
                nc = np.append(nc, constr[l-2:l])
            
            if len(self.constraints):
                self.constraints = np.vstack([self.constraints, nc])
            else:
                self.constraints = np.append(self.constraints, nc)

            i += 1
        return lines
    
    def getBNCB(self, cols):
        B = np.array(u.col(self.A, cols[0]))
        CB = np.array([self.c[cols[0]]])
        for i in range(1, len(cols)):
            B = np.hstack([B, u.col(self.A, cols[i])])
            CB = np.append(CB, self.c[cols[i]])
        return B, CB
    
    def compXB(self, B):
        return np.linalg.solve(B, self.b)
    
    def find_basic_feas_sol(self):
        # Step 1: Achar base inicial viável, ou seja, uma submatriz B de A tal 
        # que a solução do sistema B.XB = b seja >= 0.
        Bv = np.array([0, 2, 3])
        Nv = np.array([1, 4, 5, 6])
        # Bv = np.array([4, 5, 6])
        # Nv = np.array([1, 2, 3, 4])
        B, CB = self.getBNCB(Bv)
        XB = self.compXB(B)

        return B, Bv, Nv, CB, XB
    
    def steps2to5(self, B, Bv, Nv, CB, XB):
        # Step 2: Calcular variáveis duais pi resolvendo o sistema piB = CB.
        print(B.shape, CB.shape)
        pi = np.linalg.solve(B.transpose(), CB)
        print('pi = ', pi)

        # Step 3: Calcular os custos reduzidos das variáveis não básicas xj 
        # (pricing): cj_bar = cj − piAj
        cjbar, jin = u.pricing(Nv, self.A, pi, self.c)

        print('Obj = ', u.obj(CB, XB))
        print('cjbar = ', cjbar)
        
        if cjbar < 0.0:
            print(XB)
            print("OPTIMAL")
            return
        
        Aj = self.A[:,[Nv[jin]]]
    
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
        print('<-:', Nv[jin])
        print('->:', Bv[iout])
        Bv[iout], Nv[jin] = Nv[jin], Bv[iout]

        B, CB = self.getBNCB(Bv)
        XB = self.compXB(B)

        self.steps2to5(B, Bv, Nv, CB, XB)

    def sum(self, Nv, pi):
        s = 0
        for j in Nv:
            s += self.c[j] - np.linalg.solve(pi, u.col(self.A, j))

    def solve(self):
        B, Bv, Nv, CB, XB = self.find_basic_feas_sol()
        self.steps2to5(B, Bv, Nv, CB, XB)

# Enumerar todas as possibilidades
# Fornecer 
