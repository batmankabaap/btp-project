"""Traditional subgrid closure used for comparison (MCL-LW, Fig. 6)."""

from .config import G
from .physics import flux, interface_flux, right


class LaxWendroffClosure:
    """G = F_LW - F_LLF with the two-step (Richtmyer) Lax-Wendroff flux.

    Combined with MCL this is the MILES-type model the paper compares against.
    """

    def __init__(self, dt, dx, g=G):
        self.dt, self.dx, self.g = dt, dx, g

    def __call__(self, h, q):
        hr, qr = right(h), right(q)
        fh, fq = flux(h, q, self.g)
        fhr, fqr = flux(hr, qr, self.g)
        c = 0.5 * self.dt / self.dx
        h_half = 0.5 * (h + hr) - c * (fhr - fh)
        q_half = 0.5 * (q + qr) - c * (fqr - fq)
        lw_h, lw_q = flux(h_half, q_half, self.g)
        llf_h, llf_q = interface_flux(h, q, self.g)
        return lw_h - llf_h, lw_q - llf_q
