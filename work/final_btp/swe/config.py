"""Problem setup from Section 4 of the paper."""

L = 100.0
G = 9.812

N_FINE = 2000
DX = L / N_FINE          # 0.05
DT = 0.1 * DX            # 0.005

K = 20                   # coarsening degree
N_COARSE = N_FINE // K   # 100
DX_COARSE = K * DX       # 1.0
DT_COARSE = 0.005

# random training initial conditions, eq. (15)
TRAIN_H0 = 2.0
TRAIN_V0 = (1.0, 2.0)
TRAIN_AMP = (0.2, 0.6)
TRAIN_WAVENUMBER = (1, 6)

# Table 1
TEST_CASES = {
    1: dict(H0=2.0, Ah=0.20, kh=3, phih=0.00, V0=1.0, Av=0.0, kv=0, phiv=0.0),
    2: dict(H0=2.0, Ah=0.45, kh=4, phih=2.78, V0=1.1, Av=0.5, kv=3, phiv=4.5),
    3: dict(H0=2.0, Ah=0.45, kh=4, phih=2.78, V0=1.1, Av=0.5, kv=4, phiv=4.5),
    4: dict(H0=2.0, Ah=0.20, kh=3, phih=0.00, V0=1.1, Av=0.1, kv=2, phiv=0.0),
}
