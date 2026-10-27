"""Domain and discretization constants."""

L = 100.0   # domain length
g = 9.812   # gravitational acceleration

N_f = 2000        # fine cells
dx = L / N_f
dt = 0.1 * dx     # fine time step

k = 20            # coarsening factor
N_c = N_f // k    # coarse cells
dX = k * dx       # coarse cell width

dT0 = 0.005       # default coarse time step
