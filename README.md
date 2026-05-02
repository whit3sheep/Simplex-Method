# Python Simplex Algorithm Solver

A lightweight, NumPy-based Python implementation of the **Simplex Algorithm** for solving Linear Programming (LP) maximization problems. This solver uses the tabular method to iteratively find the optimal solution while tracking basic and non-basic variables.

## Overview

This project provides a `SimplexSolver` class that automatically constructs the initial Simplex tableau, identifies pivot columns and rows, and performs the necessary row operations to reach an optimal solution. It also includes a built-in formatting tool to print the tableau and the final optimal values in a clean, readable format.

### Mathematical Formulation

This solver is designed to solve standard maximization problems of the following form:

**Maximize:**
$$Z = \mathbf{c}^T \mathbf{x}$$

**Subject to:**
$$A\mathbf{x} \le \mathbf{b}$$
$$\mathbf{x} \ge 0$$

Where:
* $\mathbf{x}$ is the vector of decision variables.
* $\mathbf{c}$ is the vector of objective function coefficients.
* $A$ is the matrix of constraint coefficients.
* $\mathbf{b}$ is the vector of constraint bounds (Right-Hand Side).

The solver automatically introduces slack variables ($s_1, s_2, \dots, s_m$) to convert the inequality constraints into equalities.

---

## Features

* **NumPy Integration:** Utilizes `numpy` arrays for efficient matrix operations and row reductions.
* **Automated Tableau Construction:** Automatically handles the addition of slack variables and the objective function row ($Z$-row).
* **Step-by-Step Solving:** Allows you to iterate through the tableau updates manually or loop until an optimal solution is reached.
* **Clean CLI Output:** Features a `print_tableau()` method that renders a formatted text-based table showing the basis, $C_j$ values, row values, and the Right-Hand Side (RHS), alongside the final optimal variable values.

---



## Example Implementation

### Example Problem Formulation

This example asks the solver to **Maximize**:
$$Z = 20x_1 + 30x_2 + 15x_3 + 25x_4$$

**Subject to:**
* $2x_1 + 1x_2 + 1x_3 + 3x_4 \le 40$
* $1x_1 + 2x_2 + 0x_3 + 1x_4 \le 30$
* $0x_1 + 1x_2 + 2x_3 + 2x_4 \le 35$
* $3x_1 + 0x_2 + 1x_3 + 1x_4 \le 45$
* $1x_1 + 1x_2 + 1x_3 + 1x_4 \le 25$
* $2x_1 + 2x_2 + 2x_3 + 0x_4 \le 50$
* $x_1, x_2, x_3, x_4 \ge 0$

Here is how to initialize and run the solver using a problem with 4 variables and 6 constraints:

```python
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
    solver.print_tableau()
    solver.update_tableau()
solver.print_solution()
```

### Expected Output

When the optimal solution is reached, the solver outputs a clean, formatted tableau detailing the final basis, reduced costs (Z-row), and the maximum objective value.

```text
============================================================================================================
Basis |  Cb   |    x1      x2      x3      x4      s1      s2      s3      s4      s5      s6    |   RHS   
------------------------------------------------------------------------------------------------------------
Cj    |       |  20.00   30.00   15.00   25.00    0.00    0.00    0.00    0.00    0.00    0.00   |         
------------------------------------------------------------------------------------------------------------
x1    | 20.00 |   1.00    0.00    0.00    1.00    0.57   -0.14   -0.29    0.00    0.00    0.00   |   8.57  
x2    | 30.00 |   0.00    1.00    0.00    0.00   -0.29    0.57    0.14    0.00    0.00    0.00   |  10.71  
x3    | 15.00 |   0.00    0.00    1.00    1.00    0.14   -0.29    0.43    0.00    0.00    0.00   |  12.14  
s8    | 0.00  |   0.00    0.00    0.00   -3.00   -1.86    0.71    0.43    1.00    0.00    0.00   |   7.14  
s9    | 0.00  |   0.00    0.00    0.00   -1.00   -0.43   -0.14   -0.29    0.00    1.00    0.00   |  -6.43  
s10   | 0.00  |   0.00    0.00    0.00   -4.00   -0.86   -0.29   -0.57    0.00    0.00    1.00   | -12.86 
------------------------------------------------------------------------------------------------------------
Z-Row |       |   0.00    0.00    0.00  -10.00   -5.00  -10.00   -5.00    0.00    0.00    0.00   | -675.00 
============================================================================================================

Maximum Z Value:  675.00
Optimal Solution:
  x1 = 8.57
  x2 = 10.71
  x3 = 12.14
  x4 = 0.00
```

---

## API Reference

### `SimplexSolver(A, b, c)`
The main class to instantiate the solver.
* `A` *(numpy.ndarray)*: An $m \times n$ matrix representing the constraint coefficients.
* `b` *(numpy.ndarray)*: A 1D array of length $m$ representing the RHS values.
* `c` *(numpy.ndarray)*: A 1D array of length $n$ representing the objective function coefficients.

### Core Methods
* `optimal()`: Returns `True` if the current tableau represents the optimal solution (i.e., all values in the objective row are $\ge 0$).
* `update_tableau()`: Performs a single iteration of the Simplex method (finds the pivot column, pivot row, and performs Gaussian elimination).
* `print_tableau()`: Prints the current state of the tableau.
* `print_solution()`: Prints the maximum $Z$ value, and the values of the decision variables to the console.
* `get_solution()`: Returns a tuple containing the current objective value ($Z$) and a dictionary of the decision variables ($x_1, \dots, x_n$) and their current values.
