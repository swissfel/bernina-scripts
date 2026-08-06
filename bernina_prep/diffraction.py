"""Diffraction-angle calculations for Bernina's diffcalc 'you' geometry.

Angle-name convention: diffcalc's internal name for the out-of-plane detector
circle is ``nu``; Bernina's beamline convention calls this circle ``gamma``.
Every public function here accepts/returns ``gam``/``gamma``, translating
to/from diffcalc's ``nu`` internally so callers never need to know about it.
"""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position, get_rotation_matrices
from diffcalc.ub.calc import UBCalculation
from xrayutilities.utilities import en2lam

# Order matches diffcalc's internal constraint names; 'nu' is exposed to
# callers/UI as 'gam' per the beamline's gamma convention.
CONSTRAINT_NAMES = [
    "delta", "nu", "qaz", "naz", "alpha", "beta", "psi", "a_eq_b",
    "mu", "eta", "chi", "phi", "mu_is_gam",
]
UI_NAMES = [n if n != "nu" else "gam" for n in CONSTRAINT_NAMES]


def parse_vector(text: str) -> tuple:
    """Safely parse a '(h,k,l)'-style string into a tuple of numbers."""
    value = ast.literal_eval(text.strip())
    return tuple(value)


def parse_orientations(specs: list[str]) -> list[tuple[tuple, tuple]]:
    """Parse ['(1,1,1) > (0,0,1)', ...] into [(hkl, xyz), ...]."""
    parsed = []
    for spec in specs:
        if ">" not in spec:
            continue
        hkl_s, xyz_s = spec.split(">", 1)
        parsed.append((parse_vector(hkl_s), parse_vector(xyz_s)))
    return parsed


def build_ubcalc(
    lat_name: str,
    uc_a: float, uc_b: float, uc_c: float,
    uc_alpha: float, uc_beta: float, uc_gamma: float,
    orientations: list[str],
    surface_normal: str,
) -> UBCalculation:
    """Build and solve a diffcalc UBCalculation in the 'you' geometry."""
    ubcalc = UBCalculation("you")
    ubcalc.set_lattice(
        lat_name, a=uc_a, b=uc_b, c=uc_c,
        alpha=uc_alpha, beta=uc_beta, gamma=uc_gamma,
    )
    for hkl, xyz in parse_orientations(orientations):
        ubcalc.add_orientation(hkl, xyz, None, None)
    ubcalc.n_hkl = parse_vector(surface_normal)
    ubcalc.calc_ub()
    return ubcalc


def make_constraints(fixed: dict) -> Constraints:
    """Build diffcalc Constraints from a {ui_name: value} mapping of the
    constraints the caller wants fixed (``gam`` is translated to ``nu``)."""
    translated = {("nu" if name == "gam" else name): value for name, value in fixed.items()}
    return Constraints(translated)


def solve_hkl_positions(ubcalc: UBCalculation, cons: Constraints, h, k, l, energy: float):
    """Solve one (h,k,l) reflection, returning diffcalc's raw
    [(Position, extra_dict), ...] solution branches (angles in degrees)."""
    lam = en2lam(energy)
    hklcalc = HklCalculation(ubcalc, cons)
    return hklcalc.get_position(h, k, l, lam)


def solve_hkl(ubcalc: UBCalculation, cons: Constraints, h, k, l, energy: float) -> pd.DataFrame:
    """Solve one (h,k,l) reflection.

    Returns a DataFrame with one column per solution branch ('sol. 0', ...)
    and angle names as rows (``nu`` renamed to ``gamma``).
    """
    result = solve_hkl_positions(ubcalc, cons, h, k, l, energy)
    df = pd.concat(
        [
            pd.DataFrame.from_dict({**pos.asdict, **extra}, orient="index", columns=[f"sol. {n}"])
            for n, (pos, extra) in enumerate(result)
        ],
        axis=1,
    )
    df.rename(index={"nu": "gamma"}, inplace=True)
    return df


def reflection_table(
    ubcalc: UBCalculation,
    cons: Constraints,
    reflections: list[tuple],
    energy: float,
    solution: int = 0,
) -> pd.DataFrame:
    """Compute one row per (h,k,l) in ``reflections`` under the same fixed
    constraints, picking solution branch ``solution`` (0 = first available)
    when several exist. Reflections diffcalc can't solve get an 'error'
    message and NaN angles instead of raising, so a whole batch never fails
    because of one bad index.
    """
    rows = []
    for h, k, l in reflections:
        try:
            df = solve_hkl(ubcalc, cons, h, k, l, energy)
            col = df.columns[solution] if solution < len(df.columns) else df.columns[-1]
            row = df[col].to_dict()
            row["error"] = ""
        except Exception as e:
            row = {"error": str(e)}
        rows.append(row)
    index = pd.MultiIndex.from_tuples(reflections, names=["h", "k", "l"])
    return pd.DataFrame(rows, index=index)


def make_position(mu=0.0, delta=0.0, gamma=0.0, eta=0.0, chi=0.0, phi=0.0) -> Position:
    """Build a diffcalc Position (degrees) from Bernina-style angle names
    (``gamma`` is translated to diffcalc's ``nu``)."""
    return Position(mu=mu, delta=delta, nu=gamma, eta=eta, chi=chi, phi=phi)


def detector_hkl_grid(
    ubcalc: UBCalculation,
    pos: Position,
    energy: float,
    tdet: float,
    width: float,
    height: float,
    n: int = 9,
):
    """hkl at an n x n grid of points across a flat area detector.

    The detector is centered on the outgoing-beam direction defined by
    ``pos``'s delta/gamma, at distance ``tdet`` from the sample, ``width``
    x ``height`` in size (same length unit as ``tdet``, e.g. mm).

    Uses diffcalc's own q_phi convention directly (see ``geometry.get_q_phi``):
    q = k_out_hat - k_in_hat (unit vectors), then hkl = (2*pi/lambda) * UB^-1 @ q_phi
    -- verified to reproduce the exact (h,k,l) at the pixel matching a
    solved reflection's own delta/gamma. Returns (H, K, L), each an
    ``n`` x ``n`` numpy array.
    """
    lam = en2lam(energy)
    MU, DELTA, NU, ETA, CHI, PHI = get_rotation_matrices(pos)
    nu_delta = NU @ DELTA
    arm_dir = (nu_delta @ np.array([[0.0], [1.0], [0.0]])).ravel()
    row_dir = (nu_delta @ np.array([[1.0], [0.0], [0.0]])).ravel()
    col_dir = (nu_delta @ np.array([[0.0], [0.0], [1.0]])).ravel()
    k_in = np.array([0.0, 1.0, 0.0])

    sample_inv = (
        np.linalg.inv(PHI) @ np.linalg.inv(CHI) @ np.linalg.inv(ETA) @ np.linalg.inv(MU)
    )
    ub_inv = np.linalg.inv(ubcalc.UB)

    us = np.linspace(-width / 2, width / 2, n)
    vs = np.linspace(-height / 2, height / 2, n)
    H = np.zeros((n, n))
    K = np.zeros((n, n))
    L = np.zeros((n, n))
    for j, v in enumerate(vs):
        for i, u in enumerate(us):
            p = tdet * arm_dir + u * row_dir + v * col_dir
            k_out_hat = p / np.linalg.norm(p)
            q_lab = k_out_hat - k_in
            q_phi = sample_inv @ q_lab
            hkl = (2 * np.pi / lam) * (ub_inv @ q_phi)
            H[j, i], K[j, i], L[j, i] = hkl
    return H, K, L


def detector_hkl_summary(
    ubcalc: UBCalculation,
    pos: Position,
    energy: float,
    tdet: float,
    width: float,
    height: float,
    n: int = 9,
) -> pd.Series:
    """min/max h, k, l covered by the detector footprint (see
    ``detector_hkl_grid``), as a convenience summary."""
    H, K, L = detector_hkl_grid(ubcalc, pos, energy, tdet, width, height, n)
    return pd.Series(
        {
            "h_min": H.min(), "h_max": H.max(),
            "k_min": K.min(), "k_max": K.max(),
            "l_min": L.min(), "l_max": L.max(),
        }
    )
