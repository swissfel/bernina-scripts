"""Grazing incidence: optical constants, critical angle, penetration depth, footprint.

The refractive index n = 1 - delta + i beta comes from xraydb (tabulated
f1, f2) for a chemical formula and density when xraydb is installed; delta
and beta can also be given directly.

Penetration depth is where the intensity has dropped to 1/e. Inside the
material the wave vector's normal component is

    k_z = k sqrt(sin^2(alpha) - 2 delta + 2 i beta)

and the intensity falls as exp(-2 Im(k_z) z), so the depth is
1 / (2 Im k_z). Far above the critical angle alpha_c = sqrt(2 delta) that is
sin(alpha) / mu with mu = 4 pi beta / lambda (the attenuation length times
sin(alpha)); below alpha_c only the evanescent wave enters, a few nm deep.
A diffracted beam leaving at exit angle alpha_f is attenuated on its way
out too: the depth a reflection probes is 1 / (1/z_in + 1/z_out).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hklforge.geometry import HC_EV_ANGSTROM

try:
    import xraydb
except ImportError:          # optional: give delta, beta directly
    xraydb = None


def optical_constants(formula: str, density: float, energy: float) -> tuple[float, float]:
    """(delta, beta) of n = 1 - delta + i beta for a compound ("Bi2Se3", 7.68 g/cm3)."""
    if xraydb is None:
        raise ImportError("optical constants from a formula need xraydb (pip install xraydb); "
                          "or give delta and beta directly")
    delta, beta, _ = xraydb.xray_delta_beta(formula, float(density), float(energy))
    return float(delta), float(beta)


def critical_angle(delta: float) -> float:
    """Critical angle of total external reflection, degrees."""
    return float(np.degrees(np.sqrt(2 * delta)))


def penetration_depth(alpha_deg, delta: float, beta: float, energy: float):
    """1/e intensity depth (angstrom) at incidence (or exit) angle ``alpha_deg``
    to the surface. Accepts arrays; NaN for angles <= 0."""
    a = np.radians(np.asarray(alpha_deg, float))
    k = 2 * np.pi / (HC_EV_ANGSTROM / energy)
    kz = k * np.sqrt(np.sin(a) ** 2 - 2 * delta + 2j * beta)
    depth = 1.0 / (2 * np.abs(kz.imag))
    return np.where(a > 0, depth, np.nan)


def probed_depth(alpha_in, alpha_out, delta: float, beta: float, energy: float):
    """Depth (angstrom) probed by a reflection: incoming and outgoing
    attenuation together, 1 / (1/z_in + 1/z_out)."""
    zi = penetration_depth(alpha_in, delta, beta, energy)
    zo = penetration_depth(alpha_out, delta, beta, energy)
    return 1.0 / (1.0 / zi + 1.0 / zo)


def footprint(beam_size: float, alpha_deg: float) -> float:
    """Length of the beam's footprint on the surface (units of beam_size)
    for the beam size in the plane of incidence."""
    return float(beam_size / np.sin(np.radians(alpha_deg))) if alpha_deg > 0 else np.inf


def depth_table(delta: float, beta: float, energy: float, angles=None) -> pd.DataFrame:
    """Penetration depth at a few incidence angles around the critical angle."""
    ac = critical_angle(delta)
    if angles is None:
        angles = sorted({round(x, 3) for x in (0.25 * ac, 0.5 * ac, 0.8 * ac, ac, 1.2 * ac, 1.5 * ac,
                                               2 * ac, 0.5, 1, 2, 5, 10, 30, 90)})
    z = penetration_depth(angles, delta, beta, energy)
    return pd.DataFrame({"incidence [deg]": angles, "alpha / alpha_c": np.asarray(angles) / ac,
                         "depth [nm]": z / 10})
