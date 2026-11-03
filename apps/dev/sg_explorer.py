"""Space-group explorer (local test): symmetry, reciprocal lattice, absences.

What it is for: the link between a space group and which reflections a
diffraction experiment sees, for physicists rather than crystallographers.
Real-space symmetry is left to the Jmol Space Group Symmetry Visualizer
(https://spacegroups.symotter.org/), linked from the last tab; existing
tools give reflection conditions only as tables; this shows *why* a reflection
is missing: for every symmetry operation (R, t) that leaves the reflection's
indices unchanged (h R = h), the symmetry copies of each atom scatter with
an extra phase 2 pi h.t. If that phase is not a multiple of 2 pi the copies
cancel, for every structure with this symmetry. That is a systematic absence.

Needs gemmi (space groups, CIF, structure factors), numpy, matplotlib,
plotly, ipywidgets.
"""
from __future__ import annotations

import html
from fractions import Fraction

import gemmi
import ipywidgets as w
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from IPython.display import display

L = w.Layout
DEN = gemmi.Op.DEN

# --- example structures (CIF text, so gemmi expands the symmetry) ---------------


def _cif(name, cell, sg, atoms):
    a, b, c, al, be, ga = cell
    rows = "\n".join(f"{lab} {el} {x} {y} {z} 1" for lab, el, x, y, z in atoms)
    return (f"data_{name}\n_cell_length_a {a}\n_cell_length_b {b}\n_cell_length_c {c}\n"
            f"_cell_angle_alpha {al}\n_cell_angle_beta {be}\n_cell_angle_gamma {ga}\n"
            f"_symmetry_space_group_name_H-M '{sg}'\nloop_\n_atom_site_label\n_atom_site_type_symbol\n"
            f"_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_occupancy\n{rows}\n")


EXAMPLES = {
    "Bi2Se3 (R-3m, trigonal, hexagonal axes)": _cif(
        "Bi2Se3", (4.143, 4.143, 28.636, 90, 90, 120), "R -3 m",
        [("Bi1", "Bi", 0, 0, 0.4008), ("Se1", "Se", 0, 0, 0), ("Se2", "Se", 0, 0, 0.2117)]),
    "Si (Fd-3m, diamond)": _cif(
        "Si", (5.431, 5.431, 5.431, 90, 90, 90), "F d -3 m:1", [("Si", "Si", 0, 0, 0)]),
    "GaAs (F-43m, zinc blende)": _cif(
        "GaAs", (5.653, 5.653, 5.653, 90, 90, 90), "F -4 3 m",
        [("Ga", "Ga", 0, 0, 0), ("As", "As", 0.25, 0.25, 0.25)]),
    "SrTiO3 (Pm-3m, cubic perovskite)": _cif(
        "SrTiO3", (3.905, 3.905, 3.905, 90, 90, 90), "P m -3 m",
        [("Sr", "Sr", 0, 0, 0), ("Ti", "Ti", 0.5, 0.5, 0.5), ("O", "O", 0.5, 0.5, 0)]),
    "Mg (P63/mmc, hcp)": _cif(
        "Mg", (3.209, 3.209, 5.211, 90, 90, 120), "P 63/m m c", [("Mg", "Mg", 1 / 3, 2 / 3, 0.25)]),
    "a molecular crystal (P21/c, monoclinic)": _cif(
        "P21c", (7.27, 5.0, 5.55, 90, 96.8, 90), "P 1 21/c 1",
        [("C1", "C", 0.12, 0.23, 0.31), ("O1", "O", 0.31, 0.08, 0.17)]),
}


class Crystal:
    """Cell, space group and (optionally) atoms, from CIF text."""

    def __init__(self, cif_text):
        block = gemmi.cif.read_string(cif_text).sole_block()
        self.small = gemmi.make_small_structure_from_block(block)
        self.name = block.name
        self.sg = self.small.spacegroup
        self.cell = self.small.cell
        self.ops = self.sg.operations()
        self.atoms = [(s.label, s.element.name, np.array(s.fract.tolist())) for s in self.small.sites]
        sites = self.small.get_all_unit_cell_sites()
        self.formula_units = len(sites)
        self.small.change_occupancies_to_crystallographic()
        self.calc = gemmi.StructureFactorCalculatorX(self.cell)
        # real-space basis (columns a, b, c) and reciprocal (columns a*, b*, c*, 2 pi convention)
        self.A = np.array(self.cell.orth.mat.tolist())
        self.Bstar = 2 * np.pi * np.linalg.inv(self.A).T

    @property
    def system(self):
        return self.sg.crystal_system_str()

    def F(self, hkl):
        return abs(self.calc.calculate_sf_from_small_structure(self.small, list(hkl)))

    def F000(self):
        return self.F((0, 0, 0))

    def all_ops(self):
        """Every operation of the conventional cell: sym_ops x centring."""
        out = []
        for c in self.ops.cen_ops:
            for op in self.ops.sym_ops:
                out.append((np.array(op.rot) / DEN, (np.array(op.tran) + c) / DEN % 1.0, op, c))
        return out


# --- naming operations ---------------------------------------------------------------

_ROT_NAMES = {1: "identity", 2: "2-fold rotation", 3: "3-fold rotation", 4: "4-fold rotation",
              6: "6-fold rotation", -1: "inversion", -2: "mirror", -3: "-3 rotoinversion",
              -4: "-4 rotoinversion", -6: "-6 rotoinversion"}


def frac(x, maxden=24):
    f = Fraction(float(x)).limit_denominator(maxden)
    return "0" if f == 0 else str(f)


def _axis(R, value):
    """Integer-ish direction [uvw] of the eigenvector of R (fractional
    coordinates) with eigenvalue ``value``."""
    vals, vecs = np.linalg.eig(R)
    i = int(np.argmin(np.abs(vals - value)))
    v = np.real(vecs[:, i])
    v = v / np.max(np.abs(v))
    fr = [Fraction(float(x)).limit_denominator(6) for x in v]
    lcm = int(np.lcm.reduce([f.denominator for f in fr]))
    ints = [int(f * lcm) for f in fr]
    g = int(np.gcd.reduce([abs(i) for i in ints if i])) or 1
    return "[" + " ".join(str(i // g) for i in ints) + "]"


def describe(R, t):
    """Name of an operation x -> R x + t: kind, axis/plane, and the intrinsic
    (screw or glide) translation, the part of t that is not removable by
    moving the origin."""
    det = round(np.linalg.det(R))
    order = next(n for n in range(1, 13) if np.allclose(np.linalg.matrix_power(R, n), np.eye(3)))
    tr = round(np.trace(R))          # basis independent: 1 + 2 cos(rotation angle), times det
    kind = ({3: 1, -1: 2, 0: 3, 1: 4, 2: 6} if det > 0 else {-3: -1, 1: -2, 0: -3, -1: -4, -2: -6})[tr]
    t_int = sum(np.linalg.matrix_power(R, k) @ t for k in range(order)) / order
    t_int = np.where(np.abs(t_int) < 1e-9, 0.0, t_int)
    tstr = "(" + ", ".join(frac(x) for x in t_int) + ")"
    if kind == 1:
        return ("centring translation " + "(" + ", ".join(frac(x) for x in t) + ")") if np.any(t) else "identity", "centring" if np.any(t) else "identity"
    if kind == -1:
        return "inversion centre", "inversion"
    if kind > 1:
        ax = _axis(R, 1.0)
        if np.any(t_int):
            # screw n_m: translation m/n of the lattice vector along the axis
            return f"{kind}-fold screw axis along {ax}, translation {tstr} per turn", "screw"
        return f"{kind}-fold rotation axis along {ax}", "rotation"
    if kind == -2:
        n = _axis(R, -1.0)
        if np.any(t_int):
            return f"glide plane normal to {n}, glide {tstr}", "glide"
        return f"mirror plane normal to {n}", "mirror"
    return _ROT_NAMES.get(kind, "operation") + f" along {_axis(R, -1.0)}", "rotoinversion"


# --- why is a reflection absent --------------------------------------------------------


def invariant_ops(cr: Crystal, hkl):
    """Operations with h R = h, and the phase h.t (in turns) each adds."""
    h = np.asarray(hkl, float)
    out = []
    for R, t, op, c in cr.all_ops():
        if np.allclose(h @ R, h):
            out.append((R, t, float((h @ t) % 1.0)))
    return out


def absence_cause(cr: Crystal, hkl):
    """('' if allowed) or (kind, text): the simplest operation that cancels
    the reflection, centring first (it removes reflections everywhere in
    reciprocal space), then a screw or glide."""
    if (tuple(hkl) == (0, 0, 0)) or not cr.ops.is_systematically_absent(list(hkl)):
        return "", ""
    h = np.asarray(hkl, float)
    for c in cr.ops.cen_ops[1:]:
        t = np.array(c) / DEN
        if abs(((h @ t) + 1e-9) % 1.0) > 1e-6:
            return "centring", f"lattice centring ({', '.join(frac(x) for x in t)}): phase {frac((h @ t) % 1)} turn"
    for op in cr.ops.sym_ops:
        R, t = np.array(op.rot) / DEN, np.array(op.tran) / DEN
        if np.allclose(h @ R, h) and abs(((h @ t) + 1e-9) % 1.0) > 1e-6:
            text, kind = describe(R, t)
            return kind, f"{text} ({op.triplet()}): phase {frac((h @ t) % 1)} turn"
    for R, t, ph in invariant_ops(cr, hkl):
        if abs(ph) > 1e-6 and abs(ph - 1) > 1e-6:
            text, kind = describe(R, t)
            return kind, f"{text} combined with the centring: phase {frac(ph)} turn"
    return "other", "symmetry (combination of operations)"


COLORS = {"allowed": "#1f77b4", "centring": "#d62728", "screw": "#ff7f0e", "glide": "#9467bd",
          "other": "#8c564b", "weak": "#7f7f7f"}


# --- the widgets -------------------------------------------------------------------------


class Explorer:
    def __init__(self):
        self.crystal = None
        self.example = w.Dropdown(options=list(EXAMPLES), description="example", layout=L(width="420px"),
                                  style={"description_width": "70px"})
        self.upload = w.FileUpload(accept=".cif", multiple=False, description="own CIF ...")
        self.info = w.HTML()
        self.links = w.HTML()
        # reciprocal map
        self.plane = w.Dropdown(options=["h k 0", "h 0 l", "0 k l", "h h l", "h -h l"], value="h k 0",
                                description="layer", layout=L(width="200px"), style={"description_width": "50px"})
        self.offset = w.IntSlider(value=0, min=-6, max=6, description="third index", layout=L(width="300px"),
                                  style={"description_width": "80px"})
        self.extent = w.IntSlider(value=6, min=2, max=12, description="range", layout=L(width="260px"),
                                  style={"description_width": "50px"})
        self.with_f = w.Checkbox(value=True, description="size by |F|² (atoms)", indent=False)
        self.map_out = w.Output(layout=L(flex="1 1 520px"))
        # why
        self.hkl_text = w.Text(value="0 0 1", description="why (h k l)?", layout=L(width="230px"),
                               style={"description_width": "90px"})
        self.why_out = w.Output()
        # bases
        self.uvw_text = w.Text(value="1 0 0", description="compare (1 0 0) with [1 0 0]",
                               layout=L(width="330px"), style={"description_width": "200px"})
        self.basis_out = w.Output()
        self.example.observe(lambda c: self.load(EXAMPLES[c["new"]]), names="value")
        self.upload.observe(self._on_upload, names="value")
        for x in (self.plane, self.offset, self.extent, self.with_f):
            x.observe(lambda c: self.draw_map(), names="value")
        self.hkl_text.observe(lambda c: self.draw_why(), names="value")
        self.uvw_text.observe(lambda c: self.draw_basis(), names="value")

        tabs = w.Tab(children=[
            w.VBox([w.HTML(INTRO_MAP), w.HBox([self.plane, self.offset, self.extent, self.with_f],
                                              layout=L(flex_flow="row wrap")),
                    w.HBox([self.map_out, w.VBox([self.hkl_text, self.why_out], layout=L(flex="0 1 460px"))],
                           layout=L(flex_flow="row wrap"))]),
            w.VBox([w.HTML(INTRO_BASIS), self.uvw_text, self.basis_out]),
            w.VBox([self.links]),
        ])
        for i, t in enumerate(("reflections", "real vs reciprocal", "symmetry in 3D (symotter) and more")):
            tabs.set_title(i, t)
        self.ui = w.VBox([w.HBox([self.example, self.upload]), self.info, tabs])
        self.load(EXAMPLES[self.example.value])

    # --- loading ---

    def _on_upload(self, change):
        items = change["new"]
        if not items:
            return
        item = items[0] if isinstance(items, (list, tuple)) else list(items.values())[0]
        try:
            self.load(bytes(item["content"]).decode("utf-8", "replace"))
        except Exception as exc:
            self.info.value = f"<span style='color:#b00'>{html.escape(str(exc))}</span>"

    def load(self, cif_text):
        cr = Crystal(cif_text)
        self.crystal = cr
        c = cr.cell
        cen = {"P": "primitive", "C": "C-centred", "A": "A-centred", "B": "B-centred", "I": "body-centred",
               "F": "face-centred", "R": "rhombohedral (R) centring"}.get(cr.sg.centring_type(), "")
        atoms = ", ".join(f"{el} ({frac(x[0])}, {frac(x[1])}, {frac(x[2])})" for _, el, x in cr.atoms)
        self.info.value = (
            f"<div style='font-size:13px;line-height:1.5'><b>{html.escape(cr.name)}</b>: space group "
            f"<b>{html.escape(cr.sg.xhm())}</b> (No. {cr.sg.number}), {cr.system}, {cen}, point group "
            f"{cr.sg.point_group_hm()}; {len(cr.ops.sym_ops) * len(cr.ops.cen_ops)} operations in the "
            f"conventional cell.<br>cell a={c.a:g} b={c.b:g} c={c.c:g} Å, α={c.alpha:g} β={c.beta:g} "
            f"γ={c.gamma:g}°; atoms: {html.escape(atoms)}</div>")
        n = cr.sg.number
        self.links.value = LINKS.format(n=n, name=html.escape(cr.sg.xhm()))
        self.draw_map()
        self.draw_why()
        self.draw_basis()

    # --- reciprocal lattice layer ---

    def _layer(self):
        n, m = self.offset.value, self.extent.value
        e1, e2, o = {"h k 0": ((1, 0, 0), (0, 1, 0), (0, 0, 1)), "h 0 l": ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
                     "0 k l": ((0, 1, 0), (0, 0, 1), (1, 0, 0)), "h h l": ((1, 1, 0), (0, 0, 1), (1, -1, 0)),
                     "h -h l": ((1, -1, 0), (0, 0, 1), (1, 1, 0))}[self.plane.value]
        e1, e2, o = map(np.array, (e1, e2, o))
        pts = [tuple(int(x) for x in i * e1 + j * e2 + n * o) for i in range(-m, m + 1) for j in range(-m, m + 1)]
        return e1, e2, pts

    def draw_map(self):
        cr = self.crystal
        e1, e2, pts = self._layer()
        # 2D coordinates: the true geometry of the plane in reciprocal space
        q1 = cr.Bstar @ e1
        u = q1 / np.linalg.norm(q1)
        q2 = cr.Bstar @ e2
        v = q2 - (q2 @ u) * u
        v = v / np.linalg.norm(v)
        f000 = cr.F000()
        groups = {}
        for hkl in pts:
            q = cr.Bstar @ np.array(hkl, float)
            kind, why = absence_cause(cr, hkl)
            F = cr.F(hkl) if hkl != (0, 0, 0) else f000
            if not kind and hkl != (0, 0, 0) and F < 1e-6 * f000:
                kind, why = "weak", "allowed by symmetry, but F = 0 for these atom positions"
            key = kind or "allowed"
            d = 2 * np.pi / np.linalg.norm(q) if hkl != (0, 0, 0) else np.inf
            g = groups.setdefault(key, {"x": [], "y": [], "text": [], "size": []})
            g["x"].append(q @ u)
            g["y"].append(q @ v)
            g["text"].append(f"({' '.join(map(str, hkl))})  d = {d:.3f} Å<br>|F| = {F:.1f}"
                             + (f"<br>{html.escape(why)}" if why else ""))
            size = 7 + 18 * (F / f000) ** 2 if self.with_f.value and key == "allowed" else 9
            g["size"].append(size)
        fig = go.Figure()
        names = {"allowed": "allowed", "centring": "absent: lattice centring", "screw": "absent: screw axis",
                 "glide": "absent: glide plane", "other": "absent: other", "weak": "allowed, F = 0 (atoms)"}
        for key, g in groups.items():
            filled = key == "allowed"
            symbol = "circle" if filled else ("circle-open" if key == "weak" else "x")
            fig.add_trace(go.Scatter(x=g["x"], y=g["y"], mode="markers", name=names[key], text=g["text"],
                                     hoverinfo="text", marker=dict(
                                         size=g["size"], color=COLORS[key], symbol=symbol,
                                         line=dict(color=COLORS[key], width=1.5 if key == "weak" else 0))))
        a1, a2 = cr.Bstar @ e1, cr.Bstar @ e2
        for vec, lab in ((e1, self.plane.value.split()[0]), (e2, self.plane.value.split()[2])):
            q = cr.Bstar @ vec
            fig.add_annotation(x=q @ u, y=q @ v, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowcolor="#333",
                               text=f"({' '.join(map(str, vec))})", font=dict(size=11))
        ang = np.degrees(np.arccos(a1 @ a2 / np.linalg.norm(a1) / np.linalg.norm(a2)))
        fig.update_layout(width=600, height=640, margin=dict(l=50, r=10, t=10, b=40),
                          xaxis=dict(title="Å⁻¹", scaleanchor="y", zeroline=False),
                          yaxis=dict(title="Å⁻¹", zeroline=False),
                          legend=dict(font=dict(size=11), orientation="h", y=-0.12))
        self.map_out.clear_output()
        with self.map_out:
            display(w.HTML(f"<div style='font-size:12px'>layer {self.plane.value}, third index "
                           f"{self.offset.value}; the two directions are {ang:.1f}° apart in reciprocal space</div>"))
            display(fig)

    # --- why ---

    def draw_why(self):
        cr = self.crystal
        self.why_out.clear_output()
        try:
            hkl = tuple(int(x) for x in self.hkl_text.value.replace(",", " ").split())
            assert len(hkl) == 3
        except Exception:
            with self.why_out:
                print("give three integers, e.g. 0 0 1")
            return
        kind, why = absence_cause(cr, hkl)
        inv = invariant_ops(cr, hkl)
        F = cr.F(hkl)
        with self.why_out:
            if kind:
                display(w.HTML(f"<b>({' '.join(map(str, hkl))}) is systematically absent</b> in "
                               f"{html.escape(cr.sg.xhm())}: {html.escape(why)}.<br>"
                               "Every atom has symmetry copies whose waves reach this reflection with "
                               "these extra phases; they add to zero, whatever the atoms are."))
            elif F < 1e-6 * cr.F000():
                display(w.HTML(f"<b>({' '.join(map(str, hkl))}) is allowed by the symmetry</b> but has F = 0 "
                               "for these atom positions (a special position, not the space group); "
                               "another structure with the same space group may show it."))
            else:
                display(w.HTML(f"<b>({' '.join(map(str, hkl))}) is allowed</b>, |F| = {F:.1f} "
                               f"(F(000) = {cr.F000():.0f}). Its symmetry copies add in phase."))
            # phasors of the operations that leave h unchanged
            phases = np.array([ph for _, _, ph in inv])
            z = np.exp(2j * np.pi * phases)
            fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.4))
            ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#bbb"))
            uniq, counts = np.unique(np.round(phases % 1, 6), return_counts=True)
            for ph, cnt in zip(uniq, counts):
                zz = np.exp(2j * np.pi * ph)
                ax.annotate("", xy=(zz.real, zz.imag), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1 + cnt / 2))
                ax.text(1.15 * zz.real, 1.15 * zz.imag, f"{frac(ph)}\n×{cnt}", ha="center", va="center", fontsize=8)
            s = z.sum() / len(z)
            ax.annotate("", xy=(s.real, s.imag), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="#d62728", lw=2.5))
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{len(inv)} operations keep ({' '.join(map(str, hkl))});\nphases h·t in turns, "
                         f"red: their average = {abs(s):.2f}", fontsize=9)
            display(fig)
            plt.close(fig)

    # --- bases ---

    def draw_basis(self):
        cr = self.crystal
        self.basis_out.clear_output()
        try:
            v = np.array([float(x) for x in self.uvw_text.value.replace(",", " ").split()])
            assert v.shape == (3,)
        except Exception:
            with self.basis_out:
                print("give three numbers")
            return
        # directions only (unit length): the angles are the point, and a long
        # axis (c = 28.6 A for Bi2Se3) would squash everything else
        unit = lambda M: M / np.linalg.norm(M, axis=0)
        A, Bs = unit(cr.A), 0.8 * unit(cr.Bstar)
        fig = go.Figure()
        for i, (lab, col) in enumerate(zip("abc", ("#d62728", "#2ca02c", "#1f77b4"))):
            for M, name, dash in ((A, lab, "solid"), (Bs, lab + "*", "dot")):
                p = M[:, i]
                fig.add_trace(go.Scatter3d(x=[0, p[0]], y=[0, p[1]], z=[0, p[2]], mode="lines+text",
                                           line=dict(color=col, width=6, dash=dash), text=["", name],
                                           textfont=dict(size=14, color=col), name=name, showlegend=False))
        real = cr.A @ v
        real = 1.2 * real / np.linalg.norm(real)
        recip = cr.Bstar @ v
        recip = 1.2 * recip / np.linalg.norm(recip)
        for p, name, col in ((real, f"[{self.uvw_text.value}] real direction", "#000"),
                             (recip, f"({self.uvw_text.value}) reciprocal = plane normal", "#e377c2")):
            fig.add_trace(go.Scatter3d(x=[0, p[0]], y=[0, p[1]], z=[0, p[2]], mode="lines+text",
                                       line=dict(color=col, width=8), text=["", name], name=name))
        ang = np.degrees(np.arccos(np.clip(real @ recip / np.linalg.norm(real) / np.linalg.norm(recip), -1, 1)))
        fig.update_layout(width=720, height=560, margin=dict(l=0, r=0, t=10, b=0), scene=dict(
            aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
            legend=dict(orientation="h", y=-0.02))
        s = cr.Bstar
        angs = [np.degrees(np.arccos(s[:, i] @ s[:, j] / np.linalg.norm(s[:, i]) / np.linalg.norm(s[:, j])))
                for i, j in ((1, 2), (0, 2), (0, 1))]
        with self.basis_out:
            display(w.HTML(f"<div style='font-size:12.5px'><b>[{html.escape(self.uvw_text.value)}] and "
                           f"({html.escape(self.uvw_text.value)}) are {ang:.1f}° apart.</b> Directions only "
                           "(all drawn the same length): real a, b, c solid, reciprocal a*, b*, c* dotted.<br>"
                           f"Reciprocal cell: α* = {angs[0]:.1f}°, β* = {angs[1]:.1f}°, "
                           f"γ* = {angs[2]:.1f}° (real: α = {cr.cell.alpha:g}°, β = {cr.cell.beta:g}°, "
                           f"γ = {cr.cell.gamma:g}°). (h k l) is the normal of the lattice planes, "
                           "a*-b*-c* direction; [u v w] is the real-lattice direction "
                           "u a + v b + w c. They coincide only for orthogonal axes.</div>"))
            display(fig)


INTRO_MAP = """<div style="font-size:12.5px;line-height:1.45;max-width:1100px;margin-bottom:6px">
A layer of the reciprocal lattice in its true geometry (for a hexagonal crystal the h k 0 layer is a
60°/120° net, not a square). Filled dots are allowed reflections (size ~ |F|² for the example's atoms);
crosses are <b>systematically absent</b>, coloured by the symmetry that removes them; grey circles are allowed
by symmetry but zero for these particular atom positions. Hover for details, type a reflection on the right
to see <b>why</b>.</div>"""

INTRO_BASIS = """<div style="font-size:12.5px;line-height:1.45;max-width:1100px;margin-bottom:6px">
Real axes a, b, c and reciprocal axes a*, b*, c* drawn together (drag to rotate). a* is perpendicular to b and
c, not along a. A reflection (h k l) points along h a* + k b* + l c*, the normal of its lattice planes;
the real direction [h k l] with the same digits is generally somewhere else, e.g. hexagonal (1 0 0)
is 30° from the a axis, which is along (2 -1 0).</div>"""

LINKS = """<div style="font-size:13px;line-height:1.6;max-width:1000px">
<p>For <b>{name}</b> (No. {n}). The symmetry itself in real space (axes, mirror and glide planes,
screw axes, Wyckoff positions, copies of a draggable atom) is best explored in symotter, which this
notebook does not duplicate:</p>
<ul>
<li><a href="https://spacegroups.symotter.org/" target="_blank">Jmol Space Group Symmetry Visualizer</a>
(symotter, 2026): 3D cell with every symmetry element switchable, Wyckoff positions, draggable atoms, settings,
your own CIF. The best free real-space viewer; it does not show reflections.</li>
<li>Bilbao Crystallographic Server, the reference tables (enter group number <b>{n}</b>):
<a href="https://cryst.ehu.es/cryst/get_hkl.html" target="_blank">reflection conditions (HKLCOND)</a>,
<a href="https://cryst.ehu.es/cryst/get_wp.html" target="_blank">Wyckoff positions (WYCKPOS)</a>,
<a href="https://cryst.ehu.es/cryst/get_gen.html" target="_blank">general positions (GENPOS)</a>.</li>
<li><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11611277/" target="_blank">Bucio et al., J. Appl. Cryst. 57 (2024)</a>:
animations building the International Tables diagrams operation by operation (P4₃, R3, P6₃cm, P2₁/c, C1c1).</li>
<li><a href="https://www.doitpoms.ac.uk/tlplib/reciprocal_lattice/" target="_blank">DoITPoMS: reciprocal space</a>
and <a href="https://www.doitpoms.ac.uk/tlplib/crystallography3/" target="_blank">crystallography</a>: gentle introductions.</li>
<li><a href="https://it.iucr.org/resources/symmetrydatabase/" target="_blank">International Tables symmetry database</a>
(subscription): the official diagrams with animated screws and glides.</li>
</ul>
</div>"""
