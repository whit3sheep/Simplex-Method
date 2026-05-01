import numpy as np
import math


class SimplexSolver:
    def __init__(self, A, b, c):
    
        self.num_constraints = len(b)
        self.num_variables = len(c)
        self.tableau = self._create_initial_tableau(A, b, c)
        self.basic_variables = self._initialize_basis()
    
    def _initialize_basis(self):
        basis =  np.array(["s" + str(i + 1) for i in range(self.num_variables, self.num_variables + self.num_constraints)])
        return basis.reshape(self.num_constraints,1)
    
    def _create_initial_tableau(self,A,b,c):

        m,_ = np.shape(A)

        left = np.zeros((m, 1))
        right = np.array([b]).reshape(m,1)
        center = np.hstack([A, np.eye(m)])

        cj_row = z_row = np.hstack((
            np.zeros(1),
            c,
            np.zeros(m + 1)
        ))

        self.tableau = np.hstack((left,center,right))
        self.tableau = np.vstack((self.tableau,z_row))
        self.tableau = np.vstack((cj_row,self.tableau))

        return self.tableau
        
    def optimal(self): 
        # if True then Stop
        obj_row = self.tableau[-1,:-1]
        return  all(np.where(obj_row != 0, obj_row, -np.inf)<0)

    def pivot_column(self):
        return np.argmax(self.tableau[-1, :-1])

    def pivot_row(self):
        idx = self.pivot_column()
        with np.errstate(divide='ignore', invalid='ignore'):
            col = np.divide(self.tableau[1:-1,-1] , self.tableau[1:-1,idx])
        return np.argmin(np.where(col > 0, col, np.inf))

    def update_tableau(self):
        m,n = np.shape(self.tableau)

        if not(self.optimal()):
            col_idx = self.pivot_column() 
            row_idx = self.pivot_row() + 1

            self.tableau[row_idx] = np.divide(self.tableau[row_idx],  self.tableau[row_idx, col_idx])
            self.tableau[row_idx,0] = self.tableau[0,col_idx] 
            self.update_basis(col_idx,row_idx-1)

            guides = self.tableau[1:,col_idx]
            for i in range(1,len(guides) +1):

                if i != row_idx:
                    self.tableau[i,1:] = self.tableau[i,1:] - guides[i-1] * self.tableau[row_idx,1:]
                else:
                    None
        else:

            print("Final Solution Reached")
        return np.round(self.tableau,2)
    

    def update_basis(self, col_idx, row_idx):
        if col_idx > self.num_variables:
            self.basic_variables[row_idx] = [f"s{col_idx - self.num_variables}"]
        else:
            self.basic_variables[row_idx] = [f"x{col_idx}"]


    def get_solution(self):
            z_val = self.tableau[-1, -1]
            
            solution = {f"x{i+1}": 0.0 for i in range(self.num_variables)}
            
            for i in range(self.num_constraints):
                var_name = self.basic_variables[i][0]
                if var_name.startswith('x'):
                    solution[var_name] = self.tableau[i+1, -1]
                    
            return z_val, solution
    
    
    def print_tableau(self):
        m, n = np.shape(self.tableau)
        
        x_names = [f"x{i+1}" for i in range(self.num_variables)]
        s_names = [f"s{i+1}" for i in range(self.num_constraints)]
        all_vars = x_names + s_names
        z_total, sol = self.get_solution()
        w = 8 # Set a fixed width for all number columns
        
        print("\n" + "=" * 9 * n)
        
        var_headers = "".join([f"{v:^{w}}" for v in all_vars])
        print(f"{'Basis':<5} | {'Cb':^5} | {var_headers} | {'RHS':^{w}}")
        print("-" * 9 * n)
        
        cj_values = "".join([f"{val:^{w}.2f}" for val in self.tableau[0, 1:-1]])
        print(f"{'Cj':<5} | {'':^5} | {cj_values} | {'':^{w}}")
        print("-" * 9 * n)
        
        for i in range(1, m - 1):
            basis_var = self.basic_variables[i-1][0] # Get 'x1', 's2', etc.
            cb_val = self.tableau[i, 0]              # Leftmost column
            rhs_val = self.tableau[i, -1]            # Rightmost column
            
            # Middle columns
            row_vals = "".join([f"{val:^{w}.2f}" for val in self.tableau[i, 1:-1]])
            
            print(f"{basis_var:<5} | {cb_val:^5.2f} | {row_vals} | {rhs_val:^{w}.2f}")
            
        print("-" * 9 * n)
        
        z_vals = "".join([f"{val:^{w}.2f}" for val in self.tableau[-1, 1:-1]])

        print(f"{'Z-Row':<5} | {'':^5} | {z_vals} | {z_total:^{w}.2f}")
        print("=" * 9 * n + "\n")
        print(f"Maximum Z Value:  {-z_total:.2f}")
        print("Optimal Solution:")

        for var, val in sol.items():
            print(f"  {var} = {val:.2f}")


