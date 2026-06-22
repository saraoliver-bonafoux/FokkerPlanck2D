import numpy as np
import time
import matplotlib.pyplot as plt
from numba import njit

"""
Program to numerically integrate the Fokker-Planck equation in 2D:
dP(x,y;t)/dt = -dx[P(x,y;t) * Fx(x,y)] -dy[P(x,y;t) * Fy(x,y)] + D/2 * dxx[P(x,y;t) * Bx(x, y)] + D/2 * dyy[P(x,y;t) * By(x, y)],
where D > 0 is the diffusion coefficient.

Spatial derivatives are discretized using a conservative finite-difference scheme, while time integration is performed with an alternating-direction implicit (ADI) method.

INPUTS:
- Fx, Fy: drift functions.
- Bx, By: diffusion functions.
- D: diffusion coefficient.
- [xmin, xmax] x [ymin, ymax]: spatial domain.
- dx, dy: spatial resolution. 
- dt: time step.
- tmax: final time. 
- t_save_list: times at which the solution is stored.
- t_print_list: times at which the solution is monitored.
- filename: base name for output files ("filename_t{}.txt").
- params: model parameters. params = (params1, params2,...).
- P: initial probability density function. 

OUTPUTS:
- P: final probability density function. 

In this code, the drift and diffusion functions correspond to a genetic toggle switch model with demographic noise:
Fx = a * x**n / (Sn + x**n) + b * Sn / (Sn + y**n) - k * x
Fy = a * y**n / (Sn + y**n) + b * Sn / (Sn + x**n) - k * y
Bx = a * x**n / (Sn + x**n) + b * Sn / (Sn + y**n) + k * x
By = a * y**n / (Sn + y**n) + b * Sn / (Sn + x**n) + k * y,
where params = (a, b, k, n, Sn). 
We use a Dirac delta distribution centered at (x0,y0) as initial condition. 
"""

# --------------------------------------------------------------------------------
# DRIFT AND DIFFUSION FUNCTIONS (Genetic toggle switch model) 
# --------------------------------------------------------------------------------

@njit
def Fx(x, y, params): 
    a, b, k, n, Sn = params
    return a * x**n / (Sn + x**n) + b * Sn / (Sn + y**n) - k * x

@njit
def Fy(x, y, params): 
    a, b, k, n, Sn = params
    return a * y**n / (Sn + y**n) + b * Sn / (Sn + x**n) - k * y

@njit
def Bx(x, y, params): 
    a, b, k, n, Sn = params
    return a * x**n / (Sn + x**n) + b * Sn / (Sn + y**n) + k * x

@njit
def By(x, y, params): 
    a, b, k, n, Sn = params
    return a * y**n / (Sn + y**n) + b * Sn / (Sn + x**n) + k * y

# --------------------------------------------------------------------------------
# PARAMETERS
# --------------------------------------------------------------------------------

## GENERAL PARAMETERS
D = 5 * 1e-3
xmin, xmax = 0, 3
ymin, ymax = 0, 3
dx, dy = 5*1e-2, 5*1e-2
dt = 1e-3
tmax = 20.1
t_save_list = [10, 15, 20]
t_print_list = [0.1, 5, 10, 12, 15, 20]
filename = "FokkerPlanckSolution"

## SPECIFIC PARAMETERS (Genetic toggle switch model)
a, b, k = 1, 1, 1
n, S = 4, 0.5
Sn = S**n
params = (a, b, k, n, Sn)


tmax = 0.003
t_save_list = [0.002]
t_print_list = [0.001, 0.002, 0.003]


# --------------------------------------------------------------------------------
# INITIAL CONDITION: Dirac delta distribution centered at (x0, y0)
# --------------------------------------------------------------------------------
x0, y0 = 1, 1 
Nx = round((xmax - xmin) / dx) + 1 # Number of mesh points in the x-direction
Ny = round((ymax - ymin) / dy) + 1 # Number of mesh points in the y-direction
P = np.zeros((Ny, Nx))
i0 = round((x0 - xmin) / dx)
j0 = round((y0 - ymin) / dy)
P[j0, i0] = 1.0 
P /= np.sum(P) * dx * dy

# --------------------------------------------------------------------------------
# VISUALIZATION OF THE PROBABILITY DISTRIBUTION
# --------------------------------------------------------------------------------

def plot_FokkerPlanck(P, xmin, xmax, ymin, ymax):
    
    fs = 23
    plt.rcParams.update({'font.size': fs})
    fig = plt.figure(figsize=(9,7))
    ax = plt.gca()

    Nx = round((xmax - xmin) / dx) + 1 # Number of mesh points in the x-direction
    Ny = round((ymax - ymin) / dy) + 1 # Number of mesh points in the y-direction
    x = np.linspace(xmin, xmax, int(Nx))
    y = np.linspace(ymin, ymax, int(Ny))

    #### Linear scale
    cont = plt.contourf(x, y, P, levels = 20)
    cbar = plt.colorbar(cont, ax = ax)
    cbar.ax.set_title(r"$P(x,y)$", fontsize = fs)

    #### Log scale
    # from matplotlib.colors import LogNorm  
    # levels = np.logspace(np.log10(P.min()), np.log10(P.max()), 20)
    # cont = plt.contourf(x, y, P, levels = levels, norm = LogNorm(vmin = P.min(), vmax = P.max()))
    # inter_ticks = np.logspace(np.floor(np.log10(P.min())), np.ceil(np.log10(P.max())), len(levels))
    # ticks = np.unique(np.concatenate(([P.min()], inter_ticks, [P.max()])))
    # cbar = plt.colorbar(cont, ticks = ticks, format = '%.1e', ax = ax)
    # cbar.ax.set_title(r"$P(x,y)$", fontsize = fs)

    plt.xlabel(r"$x$", loc = "right")
    plt.ylabel(r"$y$", loc = "top", rotation = 0)
    plt.show()


# --------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS TO PERFORM THE NUMERICAL INTEGRATION 
# --------------------------------------------------------------------------------

@njit
def deltax(x, y, params):
    if Fx(x, y, params) > 0: return 1
    else: return 0

@njit
def deltay(x, y, params):
    if Fy(x, y, params) > 0: return 1
    else: return 0

@njit
def calculate_coefficients_QRS_ij(i, j, D, dx, dy, Nx, Ny, xmin, ymin, params): 
    # Computes the local coefficients Qx, Rx, Sx, Qy, Ry, and Sy corresponding to grid point (i, j).    
    x = xmin + i * dx
    y = ymin + j * dy
    xmidP, xmidM = x + 0.5 * dx, x - 0.5 * dx
    ymidP, ymidM = y + 0.5 * dy, y - 0.5 * dy
    inv_dx = 1 / dx
    inv_dy = 1 / dy
    ##### Qx, Qy
    if i > 0: Qx = Fx(xmidM, y, params) * deltax(xmidM, y, params) + 0.5 * D * inv_dx * Bx(x - dx, y, params)
    else: Qx = 0
    if j > 0: Qy = Fy(x, ymidM, params) * deltay(x, ymidM, params) + 0.5 * D * inv_dy * By(x, y - dy, params)
    else: Qy = 0
    ##### Sx, Sy
    if i < Nx - 1: Sx = 0.5 * D * inv_dx * Bx(x + dx, y, params) - Fx(xmidP, y, params) * (1 - deltax(xmidP, y, params))
    else: Sx = 0 
    if j < Ny - 1: Sy = 0.5 * D * inv_dy * By(x, y + dy, params) - Fy(x, ymidP, params) * (1 - deltay(x, ymidP, params))
    else: Sy = 0
    ##### Rx, Ry
    sumRx = 0
    if i != 0: sumRx += Fx(xmidM, y, params) * (1 - deltax(xmidM, y, params)) - 0.5 * D * inv_dx * Bx(x, y, params)
    if i != Nx-1: sumRx += - Fx(xmidP, y, params) * deltax(xmidP, y, params) - 0.5 * D * inv_dx * Bx(x, y, params)
    Rx = sumRx
    sumRy = 0
    if j != 0: sumRy += Fy(x, ymidM, params) * (1 - deltay(x, ymidM, params)) - 0.5 * D * inv_dy * By(x, y, params)
    if j != Ny-1: sumRy += - Fy(x, ymidP, params) * deltay(x, ymidP, params) - 0.5 * D * inv_dy * By(x, y, params)
    Ry = sumRy
    return Qx, Rx, Sx, Qy, Ry, Sy

@njit
def calculate_matrices_QRS(Qx_arr, Rx_arr, Sx_arr, Qy_arr, Ry_arr, Sy_arr, D, dx, dy, Nx, Ny, xmin, ymin, params): 
    # Computes the coefficient matrices Qx, Rx, Sx, Qy, Ry, and Sy on the entire spatial grid.
    Ny, Nx = Qx_arr.shape
    for j in range(Ny):
        for i in range(Nx):
            Qx, Rx, Sx, Qy, Ry, Sy = calculate_coefficients_QRS_ij(i, j, D, dx, dy, Nx, Ny, xmin, ymin, params)
            Qx_arr[j, i] = Qx
            Rx_arr[j, i] = Rx
            Sx_arr[j, i] = Sx
            Qy_arr[j, i] = Qy
            Ry_arr[j, i] = Ry
            Sy_arr[j, i] = Sy

@njit
def solve_tridiagonal_Thomas(A, B, C, D, X): 
    # Solves the tridiagonal system A[i]*X[i-1] + B[i]*X[i] + C[i]*X[i+1] = D[i] using the Thomas algorithm.
    # A, B, C: lower, main, and upper diagonals.
    # D: right-hand side.
    # X: solution vector.
    N = len(A)

    # Auxiliary vectors
    c_prime = np.zeros(N)
    d_prime = np.zeros(N)

    # New variables
    c_prime[0] = C[0] / B[0]
    d_prime[0] = D[0] / B[0]
    for i in range(1, N):
        denom = B[i] - A[i] * c_prime[i-1]
        if i < N - 1: c_prime[i] = C[i] / denom
        d_prime[i] = (D[i] - A[i] * d_prime[i-1]) / denom

    # We obtain the solution X by backwards substitution
    X[N-1] = d_prime[N-1]
    for i in range(N-2, -1, -1):
        X[i] = d_prime[i] - c_prime[i] * X[i+1]

@njit
def operator_I_plus_Ly(P, RHS, Qy_arr, Ry_arr, Sy_arr, nuy_2): 
    # Computes (I + nu_y/2 * Ly) P, i.e., the right-hand side of the first ADI substep.
    Ny, Nx = P.shape
    for j in range(Ny):
        for i in range(Nx):
            Qy = Qy_arr[j, i]
            Ry = Ry_arr[j, i]
            Sy = Sy_arr[j, i]
            if j == 0: RHS[j, i] = P[j,i] + nuy_2 * (Ry * P[j,i] + Sy * P[j+1,i])
            elif j == Ny-1: RHS[j, i] = P[j,i] + nuy_2 * (Qy * P[j-1,i] + Ry * P[j,i])
            else: RHS[j, i] = P[j,i] + nuy_2 * (Qy * P[j-1,i] + Ry * P[j,i] + Sy * P[j+1,i])

@njit
def operator_I_plus_Lx(P, RHS, Qx_arr, Rx_arr, Sx_arr, nux_2):
    # Computes (I + nu_x/2 * Lx) P, i.e., the right-hand side of the second ADI substep.
    Ny, Nx = P.shape
    for j in range(Ny):
        for i in range(Nx):
            Qx = Qx_arr[j, i]
            Rx = Rx_arr[j, i]
            Sx = Sx_arr[j, i]
            if i == 0: RHS[j, i] = P[j, i] + nux_2 * (Rx * P[j,i] + Sx * P[j,i+1])
            elif i == Nx-1: RHS[j, i] = P[j, i] + nux_2 * (Qx * P[j,i-1] + Rx * P[j,i])
            else: RHS[j, i] = P[j, i] + nux_2 * (Qx * P[j,i-1] + Rx * P[j,i] + Sx * P[j,i+1])


@njit
def adi_onestep(P, P_mid, Nx, Ny, Qx_arr, Rx_arr, Sx_arr, Qy_arr, Ry_arr, Sy_arr, nux_2, nuy_2, RHS_stepx, RHS_stepy, a_x, b_x, c_x, a_y, b_y, c_y): 
    # Advances the probability distribution by one time step using the ADI integration scheme.

    # -----------------------------------------------------------------
    # 1st ADI substep: Implicit in X, explicit in Y. Advances the solution from t_n to t_(n+1/2). 
    # -----------------------------------------------------------------
    operator_I_plus_Ly(P, RHS_stepx, Qy_arr, Ry_arr, Sy_arr, nuy_2)
    for j in range(Ny):
        for i in range(Nx):
            b_x[i] = 1 - nux_2*Rx_arr[j,i]
            if i > 0: a_x[i] = -nux_2*Qx_arr[j,i]
            if i < Nx-1: c_x[i] = -nux_2*Sx_arr[j,i]
        solve_tridiagonal_Thomas(a_x, b_x, c_x, RHS_stepx[j, :], P_mid[j,:])

    # -----------------------------------------------------------------
    # 2nd ADI substep: Implicit in Y, explicit in X. Advances the solution from t_(n+1/2) to t_(n+1).
    # -----------------------------------------------------------------
    operator_I_plus_Lx(P_mid, RHS_stepy, Qx_arr, Rx_arr, Sx_arr, nux_2)
    for i in range(Nx):
        for j in range(Ny):
            b_y[j] = 1 - nuy_2 * Ry_arr[j,i]
            if j > 0: a_y[j] = -nuy_2 * Qy_arr[j,i]
            if j < Ny-1: c_y[j] = -nuy_2 * Sy_arr[j,i]
        solve_tridiagonal_Thomas(a_y, b_y, c_y, RHS_stepy[:,i], P[:,i])

    return P

# --------------------------------------------------------------------------------
# MAIN FUNCTION TO PERFORM THE NUMERICAL INTEGRATION 
# --------------------------------------------------------------------------------

def integrate_FokkerPlanck(P, D, xmin, ymin, xmax, ymax, dx, dy, dt, tmax, t_save_list, t_print_list, filename, params):

    print("D = ", D)
    print("dx, dy, dt", dx, dy, dt)
    print("Parameters", params)

    #### Auxiliary quantities
    K = int(tmax / dt) # Number of temporal steps to perform
    idx_save_dict = {round(t/dt)-1: t for t in t_save_list} # List of temporal indices at which we want to store the solution 
    idx_print_dict = {round(t/dt)-1: t for t in t_print_list} # List of temporal indices at which we want to monitor the solution 
    nux_2 = 0.5 * dt / dx
    nuy_2 = 0.5 * dt / dy

    #### Arrays used during ADI integration 
    Ny, Nx = P.shape
    P_mid = np.zeros((Ny, Nx)) 
    RHS_stepx = np.zeros((Ny, Nx))
    RHS_stepy = np.zeros((Ny, Nx))
    ## Since the drift and diffusion functions are time-independent, the matrices Qx, Qy, Rx, Ry, Sx and Sy remain constant throughout 
    # the simulation and are therefore precomputed only once.
    Qx_arr, Rx_arr, Sx_arr = np.zeros((Ny,Nx)), np.zeros((Ny,Nx)), np.zeros((Ny,Nx))
    Qy_arr, Ry_arr, Sy_arr = np.zeros((Ny,Nx)), np.zeros((Ny,Nx)), np.zeros((Ny,Nx))
    calculate_matrices_QRS(Qx_arr, Rx_arr, Sx_arr, Qy_arr, Ry_arr, Sy_arr, D, dx, dy, Nx, Ny, xmin, ymin, params) 
    
    #### Auxiliary arrays used by the Thomas solver
    a_x, b_x, c_x = np.zeros(Nx), np.zeros(Nx), np.zeros(Nx) 
    a_y, b_y, c_y = np.zeros(Ny), np.zeros(Ny), np.zeros(Ny)  
    
    print("Initial matrices constructed")

    for k in range(K):

        adi_onestep(P, P_mid, Nx, Ny, Qx_arr, Rx_arr, Sx_arr, Qy_arr, Ry_arr, Sy_arr, nux_2, nuy_2, RHS_stepx, RHS_stepy, a_x, b_x, c_x, a_y, b_y, c_y)

        if k in idx_print_dict:
            t = idx_print_dict[k]
            print("###########")
            print("t =", t)
            print("tCPU (s) =", time.time() - t1)
            print("Pmin, Pmax", np.min(P), np.max(P))
        if k in idx_save_dict:
            t = idx_save_dict[k]
            with open(filename + f"t{t}.txt", "w") as f:
                f.write(f"{Nx} {Ny} {xmin} {xmax} {ymin} {ymax}\n")
                np.savetxt(f, P)
    return P


t1 = time.time()
P = integrate_FokkerPlanck(P, D, xmin, ymin, xmax, ymax, dx, dy, dt, tmax, t_save_list, t_print_list, filename, params)
print("###########")
print("Total computation time (s) =", time.time() - t1)
plot_FokkerPlanck(P, xmin, xmax, ymin, ymax)

