"""Crystal setup help: crystal systems, space groups, allowed reflections, CIF.

The mistakes this exists to catch, all seen in practice: a hexagonal or
trigonal cell entered with gamma = 90; aligning on a reflection the space
group forbids (e.g. (0 0 1) of R-3m Bi2Se3); a cell that does not match its
space group.

Space groups use gemmi (https://gemmi.readthedocs.io) when it is installed:
all 230 groups in their settings, systematic absences from the full
symmetry (centring, glides, screws), CIF reading and X-ray structure
factors. Without gemmi, a space group is reduced to its number (crystal
system) and its centring letter, so only centring absences are known, and
the result says so.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hklforge.geometry import HC_EV_ANGSTROM
from hklforge.lattice import b_matrix

try:
    import gemmi
except ImportError:          # optional: centring rules only
    gemmi = None

AVOGADRO = 6.02214076e23

# --- crystal systems ---------------------------------------------------------------

SYSTEMS = {
    # name: (which lengths are free, fixed angles or None if free, help)
    "cubic": ("a", {"alpha": 90, "beta": 90, "gamma": 90}, "a = b = c, all angles 90"),
    "tetragonal": ("ac", {"alpha": 90, "beta": 90, "gamma": 90}, "a = b, c; all angles 90"),
    "orthorhombic": ("abc", {"alpha": 90, "beta": 90, "gamma": 90}, "a, b, c; all angles 90"),
    "hexagonal": ("ac", {"alpha": 90, "beta": 90, "gamma": 120}, "a = b, c; alpha = beta = 90, gamma = 120"),
    "trigonal (hexagonal axes)": ("ac", {"alpha": 90, "beta": 90, "gamma": 120},
                                  "a = b, c; gamma = 120 (how R-3m, R-3c crystals are usually given)"),
    "trigonal (rhombohedral axes)": ("a", None, "a = b = c, alpha = beta = gamma (not 90)"),
    "monoclinic": ("abc", {"alpha": 90, "gamma": 90}, "a, b, c, beta; alpha = gamma = 90 (unique axis b)"),
    "triclinic": ("abc", None, "a, b, c, alpha, beta, gamma"),
}


def cell_for_system(system: str, a: float, b: float | None = None, c: float | None = None,
                    alpha: float | None = None, beta: float | None = None, gamma: float | None = None) -> dict:
    """The full cell from the parameters a crystal system leaves free, e.g.
    ``cell_for_system("hexagonal", a=4.143, c=28.636)`` -> gamma = 120."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown crystal system {system!r}; one of {', '.join(SYSTEMS)}")
    free, angles, _ = SYSTEMS[system]
    cell = {"a": float(a), "b": float(b if "b" in free and b is not None else a),
            "c": float(c if "c" in free and c is not None else a)}
    if system == "trigonal (rhombohedral axes)":
        al = float(alpha if alpha is not None else 90.0)
        cell.update(alpha=al, beta=al, gamma=al)
    elif angles is None:
        cell.update(alpha=float(alpha or 90), beta=float(beta or 90), gamma=float(gamma or 90))
    else:
        cell.update({k: float(v) for k, v in angles.items()})
        if "beta" not in angles:
            cell["beta"] = float(beta if beta is not None else 90.0)
    return cell


def system_of_number(number: int) -> str:
    """Crystal system of space group ``number`` (1-230)."""
    for top, name in ((2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"), (142, "tetragonal"),
                      (167, "trigonal"), (194, "hexagonal"), (230, "cubic")):
        if number <= top:
            return name
    raise ValueError(f"space group number {number} is not 1-230")


def _close(x, y, tol=1e-3):
    return abs(float(x) - float(y)) <= tol * max(1.0, abs(float(y)))


def check_cell(cell: dict, space_group=None) -> list[str]:
    """Warnings about a cell, alone and against a space group (a
    ``SpaceGroup``, a symbol or a number). Empty if all is consistent."""
    a, b, c = (float(cell[k]) for k in "abc")
    al, be, ga = (float(cell[k]) for k in ("alpha", "beta", "gamma"))
    out = []
    if _close(a, b) and _close(al, 90) and _close(be, 90) and _close(ga, 90) and not _close(a, c):
        if space_group is None:
            out.append("a = b with all angles 90 is a tetragonal cell. A hexagonal or trigonal crystal "
                       "given by a and c needs gamma = 120.")
    if space_group is None:
        return out
    sg = space_group if isinstance(space_group, SpaceGroup) else SpaceGroup(space_group)
    system = sg.system
    sq = _close(al, 90) and _close(be, 90) and _close(ga, 90)
    if system == "cubic" and not (_close(a, b) and _close(a, c) and sq):
        out.append(f"{sg.name} is cubic: a = b = c and all angles 90.")
    elif system == "tetragonal" and not (_close(a, b) and sq):
        out.append(f"{sg.name} is tetragonal: a = b and all angles 90.")
    elif system == "orthorhombic" and not sq:
        out.append(f"{sg.name} is orthorhombic: all angles 90.")
    elif system == "hexagonal" and not (_close(a, b) and _close(al, 90) and _close(be, 90) and _close(ga, 120)):
        out.append(f"{sg.name} is hexagonal: a = b, alpha = beta = 90, gamma = 120.")
    elif system == "trigonal":
        hex_axes = _close(a, b) and _close(al, 90) and _close(be, 90) and _close(ga, 120)
        rh_axes = _close(a, b) and _close(a, c) and _close(al, be) and _close(al, ga) and not _close(al, 90)
        if sg.centring == "R" and rh_axes and not sg.rhombohedral_axes:
            out.append(f"{sg.name} is given here in hexagonal axes, the cell is rhombohedral: "
                       f"use the space group {sg.base_symbol}:R, or the hexagonal cell (gamma = 120).")
        elif sg.centring == "R" and hex_axes and sg.rhombohedral_axes:
            out.append(f"{sg.name} is in rhombohedral axes, the cell in hexagonal ones: "
                       f"use {sg.base_symbol}:H.")
        elif not (hex_axes or rh_axes):
            out.append(f"{sg.name} is trigonal: hexagonal axes (a = b, gamma = 120) or, for R groups, "
                       "rhombohedral axes (a = b = c, alpha = beta = gamma).")
    elif system == "monoclinic" and not (_close(al, 90) and _close(ga, 90)):
        out.append(f"{sg.name} is monoclinic (unique axis b): alpha = gamma = 90.")
    return out


# --- space groups ------------------------------------------------------------------

_CENTRING_RULES = {
    "P": (lambda h, k, l: True, "no condition"),
    "A": (lambda h, k, l: (k + l) % 2 == 0, "A centring: k + l even"),
    "B": (lambda h, k, l: (h + l) % 2 == 0, "B centring: h + l even"),
    "C": (lambda h, k, l: (h + k) % 2 == 0, "C centring: h + k even"),
    "I": (lambda h, k, l: (h + k + l) % 2 == 0, "I centring: h + k + l even"),
    "F": (lambda h, k, l: len({h % 2, k % 2, l % 2}) == 1, "F centring: h, k, l all even or all odd"),
    # obverse setting, hexagonal axes (International Tables standard)
    "R": (lambda h, k, l: (-h + k + l) % 3 == 0, "R centring (hexagonal axes, obverse): -h + k + l = 3n"),
}


@dataclass
class SpaceGroup:
    """A space group from its symbol ("R-3m", "Fd-3m", "P 63/m m c", "R-3m:R")
    or number. ``full`` tells whether all absences are known (gemmi) or only
    the centring ones."""
    symbol: str | int
    name: str = field(init=False)
    number: int = field(init=False)
    centring: str = field(init=False)
    full: bool = field(init=False)
    _sg: object = field(init=False, repr=False, default=None)

    def __post_init__(self):
        text = str(self.symbol).strip()
        if gemmi is not None:
            sg = gemmi.find_spacegroup_by_number(int(text)) if text.isdigit() else \
                gemmi.find_spacegroup_by_name(text)
            if sg is None:
                raise ValueError(f"unknown space group {text!r}")
            self._sg, self.name, self.number = sg, sg.xhm(), sg.number
            self.centring, self.full = sg.centring_type(), True
            return
        # without gemmi: the number gives the system, the leading letter the centring
        if text.isdigit():
            self.number, self.centring, self.name = int(text), "P", f"No. {text}"
            raise ValueError("without gemmi give the symbol (e.g. 'R-3m'), not only the number: "
                             "the centring letter is needed")
        letter = text.lstrip()[0].upper()
        if letter not in _CENTRING_RULES:
            raise ValueError(f"{text!r}: a space-group symbol starts with P, A, B, C, I, F or R")
        self.name, self.number, self.centring, self.full = text, 0, letter, False

    @property
    def system(self) -> str:
        if self._sg is not None:
            return self._sg.crystal_system_str()
        return {"R": "trigonal"}.get(self.centring, "unknown")

    @property
    def rhombohedral_axes(self) -> bool:
        return self.name.endswith(":R")

    @property
    def base_symbol(self) -> str:
        return self.name.split(":")[0]

    def is_allowed(self, hkl) -> bool:
        h, k, l = (int(round(x)) for x in hkl)
        if (h, k, l) == (0, 0, 0):
            return True
        if self._sg is not None:
            return not self._sg.operations().is_systematically_absent([h, k, l])
        if self.centring == "R" and self.rhombohedral_axes:
            return True
        return _CENTRING_RULES[self.centring][0](h, k, l)

    def why_forbidden(self, hkl) -> str:
        """'' if allowed, else which symmetry removes it (centring, or the
        glides/screws of the group)."""
        if self.is_allowed(hkl):
            return ""
        h, k, l = (int(round(x)) for x in hkl)
        rule, text = _CENTRING_RULES.get(self.centring, (None, ""))
        if rule is not None and not (self.centring == "R" and self.rhombohedral_axes) and not rule(h, k, l):
            return f"forbidden in {self.name} by {text}"
        return f"forbidden in {self.name} by a glide plane or screw axis"

    def condition_summary(self) -> str:
        rule = _CENTRING_RULES.get(self.centring)
        text = rule[1] if rule and not (self.centring == "R" and self.rhombohedral_axes) else "no centring condition"
        return text + ("" if self.full else " (only centring is checked: install gemmi for glides and screws)")

    def is_unique(self, hkl) -> bool:
        """One representative per set of symmetry-equivalent reflections."""
        if self._sg is not None:
            return gemmi.ReciprocalAsu(self._sg).is_in([int(round(x)) for x in hkl])
        return tuple(-x for x in hkl) < tuple(hkl)          # Friedel pairs only


# --- structures from CIF -------------------------------------------------------------


@dataclass
class Structure:
    """Cell, space group and atoms read from a CIF (needs gemmi)."""
    name: str
    cell: dict
    space_group: SpaceGroup
    formula: str
    density: float          # g/cm3
    z_electrons: float      # electrons per cell, = F(000)
    _small: object = field(repr=False, default=None)
    _calc: object = field(repr=False, default=None)

    def structure_factor(self, hkl) -> float:
        """abs(F(hkl)) in electrons (non-resonant X-ray form factors)."""
        return float(abs(self._calc.calculate_sf_from_small_structure(self._small, [int(round(x)) for x in hkl])))


def read_cif(text_or_path) -> Structure:
    """A CIF file (path or its text) -> ``Structure``."""
    if gemmi is None:
        raise ImportError("reading CIF files needs gemmi (pip install gemmi)")
    text = text_or_path
    if not ("\n" in str(text_or_path) or str(text_or_path).lstrip().startswith("data_")):
        with open(text_or_path) as f:
            text = f.read()
    block = gemmi.cif.read_string(text).sole_block()
    small = gemmi.make_small_structure_from_block(block)
    if small.spacegroup is None:
        raise ValueError("the CIF has no space group (_symmetry_space_group_name_H-M or _space_group_IT_number)")
    uc = small.cell
    cell = dict(a=uc.a, b=uc.b, c=uc.c, alpha=uc.alpha, beta=uc.beta, gamma=uc.gamma)
    # composition from the occupancies as written (chemical), before gemmi's
    # structure factors get them divided by the site multiplicities
    counts: dict = {}
    for site in small.get_all_unit_cell_sites():
        counts[site.element.name] = counts.get(site.element.name, 0.0) + site.occ
    mass = sum(n * gemmi.Element(el).weight for el, n in counts.items())
    electrons = sum(n * gemmi.Element(el).atomic_number for el, n in counts.items())
    gcd = _formula_divisor(list(counts.values()))
    formula = "".join(f"{el}{_num(n / gcd)}" for el, n in counts.items())
    density = mass / (uc.volume * 1e-24 * AVOGADRO)
    small.change_occupancies_to_crystallographic()
    sg = SpaceGroup(small.spacegroup.xhm())
    return Structure(block.name, cell, sg, formula, density, electrons, small,
                     gemmi.StructureFactorCalculatorX(small.cell))


def _num(x):
    return "" if abs(x - 1) < 1e-6 else (f"{x:.0f}" if abs(x - round(x)) < 1e-6 else f"{x:.3g}")


def _formula_divisor(values):
    ints = [int(round(v)) for v in values]
    if any(abs(v - i) > 1e-6 for v, i in zip(values, ints)) or not all(ints):
        return 1.0
    return float(np.gcd.reduce(ints))


# --- reflection lists ------------------------------------------------------------------


def reflection_list(cell: dict, energy: float, space_group=None, structure: Structure | None = None,
                    d_min: float | None = None, max_rows: int = 2000) -> pd.DataFrame:
    """Reflections reachable at ``energy`` (d >= lambda / 2, or ``d_min``),
    one per symmetry-equivalent set, sorted by d (largest first).

    Columns: h, k, l, d [A], 2theta at this energy, allowed (with the reason
    when not) and, with a structure, abs(F) and abs(F)^2 relative to the
    strongest. Without a space group every reflection is "allowed".
    """
    sg = None if space_group is None else (space_group if isinstance(space_group, SpaceGroup)
                                           else SpaceGroup(space_group))
    if structure is not None and sg is None:
        sg = structure.space_group
    lam = HC_EV_ANGSTROM / energy
    d_min = max(d_min or 0.0, lam / 2)
    B = b_matrix(*(float(cell[k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")))
    qmax = 2 * np.pi / d_min
    lim = [int(np.floor(qmax / np.linalg.norm(B @ e))) for e in np.eye(3)]
    if np.prod([2 * m + 1 for m in lim]) > 5e6:
        raise ValueError("too many reflections: raise d_min or lower the energy")
    rng = [np.arange(-m, m + 1) for m in lim]
    H = np.stack(np.meshgrid(*rng, indexing="ij"), -1).reshape(-1, 3)
    H = H[np.any(H != 0, axis=1)]
    d = 2 * np.pi / np.linalg.norm(H @ B.T, axis=1)
    keep = d >= d_min * (1 - 1e-9)
    H, d = H[keep], d[keep]
    rows = []
    for hkl, dd in sorted(zip(map(tuple, H), d), key=lambda x: (-x[1], [-abs(v) for v in x[0]], x[0])):
        if sg is not None and not sg.is_unique(hkl):
            continue
        row = {"h": int(hkl[0]), "k": int(hkl[1]), "l": int(hkl[2]), "d [A]": float(dd),
               "2theta": float(np.degrees(2 * np.arcsin(min(1.0, lam / (2 * dd)))))}
        if sg is not None:
            row["allowed"] = sg.is_allowed(hkl)
            row["why not"] = sg.why_forbidden(hkl)
        if structure is not None:
            row["|F|"] = structure.structure_factor(hkl)
        rows.append(row)
        if len(rows) >= max_rows:
            break
    df = pd.DataFrame(rows)
    if structure is not None and len(df):
        fmax = df["|F|"].max()
        df["I rel"] = (df["|F|"] / fmax) ** 2 if fmax > 0 else 0.0
        # accidentally weak (allowed by symmetry, ~0 by the atom positions)
        df.loc[df["allowed"] & (df["|F|"] < 1e-6 * max(fmax, 1)), "why not"] = "allowed by symmetry, F = 0 (atom positions)"
    return df


def target_check(hkl, cell: dict, space_group=None, structure: Structure | None = None) -> list[str]:
    """Warnings for an alignment target: forbidden, or zero/weak intensity."""
    out = []
    if any(abs(x - round(x)) > 1e-6 for x in hkl):
        return out
    sg = structure.space_group if structure is not None and space_group is None else space_group
    if sg is not None:
        sg = sg if isinstance(sg, SpaceGroup) else SpaceGroup(sg)
        why = sg.why_forbidden(hkl)
        if why:
            out.append(f"({' '.join(str(int(round(x))) for x in hkl)}) is {why}")
            return out
    if structure is not None:
        f = structure.structure_factor(hkl)
        if f < 1e-6 * structure.z_electrons:
            out.append(f"({' '.join(str(int(round(x))) for x in hkl)}) has F = 0 from the atom positions")
        elif f < 0.02 * structure.z_electrons:
            out.append(f"({' '.join(str(int(round(x))) for x in hkl)}) is weak: |F| = {f:.3g} of F(000) = "
                       f"{structure.z_electrons:.0f}")
    return out


def allowed_function(space_group=None, structure: Structure | None = None):
    """``hkl -> bool`` for filtering, or None if nothing is known."""
    sg = structure.space_group if structure is not None and space_group is None else space_group
    if sg is None:
        return None
    sg = sg if isinstance(sg, SpaceGroup) else SpaceGroup(sg)
    return sg.is_allowed


def to_cif_text(upload_bytes) -> str:
    """Bytes of an uploaded CIF -> text (UTF-8, latin-1 as a fallback)."""
    raw = bytes(upload_bytes)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return io.TextIOWrapper(io.BytesIO(raw), encoding="latin-1").read()
