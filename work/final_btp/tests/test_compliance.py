"""Checks every building block against the equations of the paper."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swe import config as cfg
from swe.data import random_ic_params, stencil, subgrid_flux
from swe.grid import cell_centers, coarsen, sine_ic, step_ic
from swe.mcl import mcl
from swe.network import NeuralClosure, Normalizer, SubgridNet, build_net
from swe.physics import bar_states, flux, interface_flux, left, llf_flux, right, wave_speed
from swe.solver import FVModel


def fine_state(case=2, steps=400):
    x = cell_centers(cfg.N_FINE)
    h, q = sine_ic(x, **cfg.TEST_CASES[case])
    model = FVModel(cfg.DX)
    for _ in range(steps):
        h, q = model.heun_step(h, q, cfg.DT)
    return h, q


def random_coarse_state(rng, n=cfg.N_COARSE):
    h = 2.0 + 0.6 * rng.standard_normal(n).clip(-2, 2)
    q = h * (1.2 + 0.8 * rng.standard_normal(n))
    return h, q


# --- setup (Section 4, Table 1, eq. 15) -------------------------------------

def test_section4_parameters():
    assert cfg.L == 100 and cfg.G == 9.812
    assert cfg.N_FINE == 2000 and cfg.DX == pytest.approx(0.05)
    assert cfg.DT == pytest.approx(0.1 * cfg.DX) == pytest.approx(0.005)
    assert cfg.K == 20 and cfg.N_COARSE == 100 and cfg.DX_COARSE == pytest.approx(1.0)
    assert cfg.DT_COARSE == pytest.approx(0.005)


def test_table1():
    assert cfg.TEST_CASES[1] == dict(H0=2.0, Ah=0.2, kh=3, phih=0.0, V0=1.0, Av=0.0, kv=0, phiv=0.0)
    assert cfg.TEST_CASES[2] == dict(H0=2.0, Ah=0.45, kh=4, phih=2.78, V0=1.1, Av=0.5, kv=3, phiv=4.5)
    assert cfg.TEST_CASES[3] == dict(H0=2.0, Ah=0.45, kh=4, phih=2.78, V0=1.1, Av=0.5, kv=4, phiv=4.5)
    assert cfg.TEST_CASES[4] == dict(H0=2.0, Ah=0.2, kh=3, phih=0.0, V0=1.1, Av=0.1, kv=2, phiv=0.0)


def test_training_ic_distributions():
    p = random_ic_params(np.random.default_rng(1), 200_000)
    assert np.all(p["H0"] == 2.0)
    assert 1 <= p["V0"].min() and p["V0"].max() <= 2
    for a in ("Ah", "Av"):
        assert 0.2 <= p[a].min() and p[a].max() <= 0.6
    for k in ("kh", "kv"):
        assert set(np.unique(p[k])) == {1, 2, 3, 4, 5, 6}
    for ph in ("phih", "phiv"):
        assert 0 <= p[ph].min() and p[ph].max() <= 2 * np.pi


def test_initial_conditions():
    x = cell_centers(cfg.N_FINE)
    h, q = sine_ic(x, **cfg.TEST_CASES[4])
    assert np.allclose(h, 2 + 0.2 * np.sin(2 * np.pi * 3 * x / 100))
    assert np.allclose(q / h, 1.1 + 0.1 * np.sin(2 * np.pi * 2 * x / 100))
    h, q = step_ic(x, 2.4, 0.5)
    assert np.all(q == 0)
    assert np.allclose(h[(x > 100 / 3) & (x < 200 / 3)], 2.9)
    assert np.allclose(h[(x < 100 / 3) | (x > 200 / 3)], 2.15)


# --- LLF discretization (eqs. 2-7) ------------------------------------------

def test_flux_and_wave_speed():
    fh, fq = flux(2.0, 3.0)
    assert fh == 3.0 and fq == pytest.approx(4.5 + 0.5 * 9.812 * 4)
    lam = wave_speed(2.0, -3.0, 1.0, 0.5)
    assert lam == pytest.approx(max(1.5 + np.sqrt(9.812 * 2), 0.5 + np.sqrt(9.812)))


def test_llf_is_consistent_and_matches_eq4():
    rng = np.random.default_rng(0)
    h, q = random_coarse_state(rng)
    fh, fq = llf_flux(h, q, h, q)
    assert np.allclose(fh, flux(h, q)[0]) and np.allclose(fq, flux(h, q)[1])

    hr, qr = right(h), right(q)
    lam = wave_speed(h, q, hr, qr)
    Fh, Fq = interface_flux(h, q)
    assert np.allclose(Fh, 0.5 * (q + qr) - 0.5 * lam * (hr - h))
    assert np.allclose(Fq, 0.5 * (flux(h, q)[1] + flux(hr, qr)[1]) - 0.5 * lam * (qr - q))


def test_fluctuation_form_eq7():
    rng = np.random.default_rng(1)
    h, q = random_coarse_state(rng)
    dh, dq = FVModel(1.0).rhs(h, q)
    hb, qb, lam = bar_states(h, q)
    assert np.allclose(dh, left(lam) * (left(hb) - h) + lam * (hb - h))
    assert np.allclose(dq, left(lam) * (left(qb) - q) + lam * (qb - q))


def test_heun_is_convex_combination_of_euler_steps():
    rng = np.random.default_rng(2)
    h, q = random_coarse_state(rng)
    m, dt = FVModel(1.0), 0.01
    h1 = h + dt * m.rhs(h, q)[0]
    q1 = q + dt * m.rhs(h, q)[1]
    h2 = h1 + dt * m.rhs(h1, q1)[0]
    assert np.allclose(m.heun_step(h, q, dt)[0], 0.5 * h + 0.5 * h2)


def test_fine_solver_conserves_and_stays_positive():
    x = cell_centers(cfg.N_FINE)
    h0, q0 = sine_ic(x, **cfg.TEST_CASES[3])
    h, q = fine_state(3, 400)
    assert h.min() > 0
    assert h.sum() == pytest.approx(h0.sum(), rel=1e-12)
    assert q.sum() == pytest.approx(q0.sum(), rel=1e-12)


def test_batched_solver_matches_single_runs():
    x = cell_centers(cfg.N_FINE)
    hs, qs = zip(*(sine_ic(x, **cfg.TEST_CASES[c]) for c in (1, 2)))
    m = FVModel(cfg.DX)
    hb, qb = m.heun_step(np.stack(hs), np.stack(qs), cfg.DT)
    for i in range(2):
        assert np.array_equal(hb[i], m.heun_step(hs[i], qs[i], cfg.DT)[0])


# --- coarse variables and subgrid flux (eqs. 8-13) --------------------------

def test_coarsening_eq9():
    u = np.arange(40.0)
    assert np.allclose(coarsen(u, 20), [9.5, 29.5])


def test_subgrid_flux_uses_fine_flux_at_macrocell_interface():
    h, q = fine_state()
    H, Q, gh, gq = subgrid_flux(h, q)
    k = cfg.K
    for i in (0, 37, cfg.N_COARSE - 1):
        a, b = k * i + k - 1, (k * i + k) % cfg.N_FINE
        fh, fq = llf_flux(h[a], q[a], h[b], q[b])
        Fh, Fq = llf_flux(H[i], Q[i], H[(i + 1) % cfg.N_COARSE], Q[(i + 1) % cfg.N_COARSE])
        assert gh[i] == pytest.approx(fh - Fh) and gq[i] == pytest.approx(fq - Fq)


def test_exact_subgrid_flux_closes_coarse_equation_eq10():
    """Averaging the fine RHS reproduces eq. (10) exactly."""
    h, q = fine_state()
    dh, dq = FVModel(cfg.DX).rhs(h, q)
    H, Q, gh, gq = subgrid_flux(h, q)
    exact = lambda H_, Q_: (gh, gq)
    dH, dQ = FVModel(cfg.DX_COARSE, closure=exact).rhs(H, Q)
    assert np.allclose(coarsen(dh, cfg.K), dH, atol=1e-12)
    assert np.allclose(coarsen(dq, cfg.K), dQ, atol=1e-12)


def test_stencil_ordering():
    H, Q = np.arange(10.0), 100 + np.arange(10.0)
    X = stencil(H, Q)
    assert X.shape == (10, 8)
    assert np.allclose(X[3], [2, 102, 3, 103, 4, 104, 5, 105])
    assert np.allclose(X[9], [8, 108, 9, 109, 0, 100, 1, 101])


# --- network (Section 3.1, Fig. 1) ------------------------------------------

@pytest.mark.parametrize("hidden", [128, 64, 32, 16])
def test_network_architecture(hidden):
    net = SubgridNet(hidden)
    linear = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    acts = [m for m in net.modules() if isinstance(m, torch.nn.LeakyReLU)]
    assert [(l.in_features, l.out_features) for l in linear] == \
        [(8, hidden), (hidden, hidden), (hidden, hidden), (hidden, 2)]
    assert len(acts) == 3 and all(a.negative_slope == 0.01 for a in acts)


# --- MCL (eqs. 17-31) -------------------------------------------------------

def limited_bar_states(h, q, gh, gq):
    hb, qb, lam = bar_states(h, q)
    gh_s, gq_s = mcl(h, q, gh, gq)
    return hb, qb, lam, gh_s, gq_s, hb - gh_s / lam, qb - gq_s / lam, hb + gh_s / lam, qb + gq_s / lam


@pytest.mark.parametrize("seed", range(5))
def test_mcl_enforces_local_bounds_eq24(seed):
    rng = np.random.default_rng(seed)
    h, q = random_coarse_state(rng)
    gh, gq = 5 * rng.standard_normal((2, h.size))
    hb, qb, lam, gh_s, gq_s, hm, qm, hp, qp = limited_bar_states(h, q, gh, gq)
    vb = qb / hb
    hmin, hmax = np.minimum(left(hb), hb), np.maximum(left(hb), hb)
    vmin, vmax = np.minimum(left(vb), vb), np.maximum(left(vb), vb)
    tol = 1e-12
    assert np.all(hmin - tol <= hm) and np.all(hm <= hmax + tol)
    assert np.all(right(hmin) - tol <= hp) and np.all(hp <= right(hmax) + tol)
    assert np.all(hm * vmin - tol <= qm) and np.all(qm <= hm * vmax + tol)
    assert np.all(hp * right(vmin) - tol <= qp) and np.all(qp <= hp * right(vmax) + tol)
    assert np.all(hm > 0) and np.all(hp > 0)


def test_mcl_keeps_admissible_fluxes_and_sign():
    rng = np.random.default_rng(7)
    h, q = random_coarse_state(rng)
    zero = np.zeros_like(h)
    assert np.all(mcl(h, q, zero, zero)[0] == 0) and np.all(mcl(h, q, zero, zero)[1] == 0)
    gh, gq = 1e-9 * rng.standard_normal((2, h.size))
    gh_s, _ = mcl(h, q, gh, gq)
    assert np.all(np.sign(gh_s) * np.sign(gh) >= 0) and np.all(np.abs(gh_s) <= np.abs(gh))


def test_mcl_is_idempotent():
    rng = np.random.default_rng(8)
    h, q = random_coarse_state(rng)
    gh, gq = 3 * rng.standard_normal((2, h.size))
    once = mcl(h, q, gh, gq)
    twice = mcl(h, q, *once)
    assert np.allclose(once, twice)


@pytest.mark.parametrize("seed", range(3))
def test_mcl_forward_euler_is_positivity_preserving_eq21(seed):
    """Under the CFL condition (21) every MCL Euler stage stays positive and
    is a convex combination of U_i and the limited bar states."""
    rng = np.random.default_rng(seed)
    h, q = random_coarse_state(rng)
    garbage = lambda H, Q: tuple(50 * rng.standard_normal((2, H.size)))
    model = FVModel(1.0, closure=garbage, limiter=True)
    hb, _, lam = bar_states(h, q)
    dt = 1.0 / np.max(lam + left(lam))
    for _ in range(200):
        dh, dq = model.rhs(h, q)
        h, q = h + dt * dh, q + dt * dq
        assert h.min() > 0
        hb, _, lam = bar_states(h, q)
        dt = 1.0 / np.max(lam + left(lam))


def test_coarse_model_with_limiter_is_conservative():
    rng = np.random.default_rng(3)
    h, q = random_coarse_state(rng)
    closure = lambda H, Q: tuple(rng.standard_normal((2, H.size)))
    model = FVModel(1.0, closure=closure, limiter=True)
    h1, q1 = model.heun_step(h, q, 0.01)
    assert h1.sum() == pytest.approx(h.sum(), rel=1e-13)
    assert q1.sum() == pytest.approx(q.sum(), rel=1e-13)


# --- mirror symmetry x -> -x ------------------------------------------------

def mirror_fine(h, q):
    return h[::-1].copy(), -q[::-1].copy()


def test_exact_subgrid_flux_is_mirror_symmetric():
    h, q = fine_state()
    _, _, gh, gq = subgrid_flux(h, q)
    _, _, ghm, gqm = subgrid_flux(*mirror_fine(h, q))
    j = (cfg.N_COARSE - 2 - np.arange(cfg.N_COARSE)) % cfg.N_COARSE
    assert np.allclose(ghm[j], -gh, atol=1e-12) and np.allclose(gqm[j], gq, atol=1e-12)


def test_mirror_network_is_exactly_equivariant():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    H, Q = random_coarse_state(rng)
    norm_x = Normalizer(rng.normal(size=8), 0.5 + rng.random(8))
    norm_y = Normalizer(rng.normal(size=2), 0.5 + rng.random(2))
    arch = dict(hidden=16, symmetric=True)
    closure = NeuralClosure(build_net(arch, norm_x, norm_y), norm_x, norm_y, arch)
    gh, gq = closure(H, Q)
    ghm, gqm = closure(H[::-1].copy(), -Q[::-1].copy())
    j = (H.size - 2 - np.arange(H.size)) % H.size
    assert np.allclose(ghm[j], -gh, atol=1e-5) and np.allclose(gqm[j], gq, atol=1e-5)


def test_checkpoint_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    norm = Normalizer(np.zeros(8), np.ones(8)), Normalizer(np.zeros(2), np.ones(2))
    arch = dict(hidden=8, symmetric=True)
    closure = NeuralClosure(build_net(arch, *norm), *norm, arch)
    closure.save(tmp_path / "m.pt")
    loaded = NeuralClosure.load(tmp_path / "m.pt")
    H, Q = random_coarse_state(rng)
    assert loaded.arch == arch
    assert np.allclose(closure(H, Q)[1], loaded(H, Q)[1])


# --- dashboard simulation ---------------------------------------------------

def test_dashboard_simulation_layout_and_metrics():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
    from simulation import simulate

    zero = lambda H, Q: (np.zeros_like(H), np.zeros_like(Q))
    out = simulate({"zero": zero}, dict(type="sine", **cfg.TEST_CASES[2], t_end=0.5, dt=0.05))
    assert set(out["curves"]) == {"llf", "zero/nn", "zero/mcl"}
    assert len(out["fine"]["h"]) == cfg.N_FINE and len(out["x"]) == cfg.N_COARSE
    # a zero closure is plain LLF, and MCL leaves a zero correction untouched
    for key in ("zero/nn", "zero/mcl"):
        assert out["curves"][key] == out["curves"]["llf"]
        assert out["metrics"][key]["q_err"] == out["metrics"]["llf"]["q_err"]
    assert out["metrics"]["fine"]["cells"] == cfg.N_FINE
