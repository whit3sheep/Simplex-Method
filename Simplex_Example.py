import numpy as np
from Simplex import SimplexSolver

A = np.array([
    [2, 1, 1, 3],  # Constraint 1
    [1, 2, 0, 1],  # Constraint 2
    [0, 1, 2, 2],  # Constraint 3
    [3, 0, 1, 1],  # Constraint 4
    [1, 1, 1, 1],  # Constraint 5
    [2, 2, 2, 0]   # Constraint 6
])
b = np.array([40, 30, 35, 45, 25, 50])
c = np.array([20, 30, 15, 25])


solver = SimplexSolver(A, b, c)

while not solver.optimal():
    solver.update_tableau()
solver.print_tableau()
solver.print_solution()


# or for just the solution 
# solver.solve()
