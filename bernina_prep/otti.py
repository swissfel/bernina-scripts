"""Otti: the motorized detector arm on the THC lid, and what its detector sees.

The geometry is eco's ``calc_otti`` (``eco.endstations.bernina_sample_environments``):
a JUNGFRAU module (1024 x 512 pixels of 75 um) faces down at 179.287 mm
from the arm's pivot, which sits 46 mm to the side of and 77.436 mm above
the sample. Lab frame as there: x horizontal (towards the lid centre), y up,
z along the beam; azimuth = atan2(x, z), elevation above the horizontal
plane. The detector orientation is

    R = Ry(nu) Rx(del) Ry(det)

with ``nu`` (SARES20-EXP:MOT_RY) turning the arm about the vertical through
the pivot, ``del`` (MOT_RZ) tilting it, and ``det`` the detector's turn
about its own normal (encoder DET_RY plus a stored offset in eco).

``det`` is not driven: the detector hangs in a cardan (gimbal) mount on the
upward-pointing flange, so it follows from nu and del. Three models:

* ``"twist-free"`` - det = offset - nu, eco's ``otti_det="auto"``: the
  detector is carried along without turning about its normal (the tilt is
  a single rotation about a horizontal axis).
* ``"cardan"`` - a gimbal whose outer axis is fixed to the flange,
  horizontal at azimuth ``psi``. A gimbal never turns about the third axis,
  which is not the same as twist-free once both gimbal angles are non-zero:
  the detector turns by an extra ~del^2/2 * sin 2(nu - psi) (0.4 deg at
  del = 10, 4 deg at 30, 10 deg at 45). ``psi`` and the offset are not known
  yet; ``fit_cardan`` gets both from recorded (nu, del, encoder) triples.
* ``"encoder"`` - det as given (the measured value).

DEL_MAX = 44 deg is where the reach of this model matches the envelope from
the original thc_angles app (``data/otti.npy``, CAD/ray-traced) to 0.6 deg rms
over all azimuths; it is the tilt the lid allows, not a motor limit.
"""
from __future__ import annotations

import numpy as np

DISTANCE = 179.287                         # pivot to detector face [mm]
PIVOT = np.array([46.0, 77.436, 0.0])      # pivot relative to the sample [mm]
NX, NZ, PIXEL = 1024, 512, 0.075           # JUNGFRAU module
DEL_MAX = 44.0
SPIN_MODELS = ("twist-free", "cardan", "encoder")


def _rx(a):
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    t = np.radians(a)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def _wrap(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def cardan_det(nu, de, psi=0.0):
    """det (without offset) of a detector in a gimbal whose outer axis is
    horizontal at azimuth ``psi``: the detector normal n = Ry(nu) Rx(del) y is
    reached as R_u(a) R_v(b) y (u outer, v = y x u inner axis)."""
    y = np.array([0.0, 1.0, 0.0])
    n = _ry(nu) @ _rx(de) @ y
    u = _ry(psi) @ np.array([1.0, 0.0, 0.0])
    v = np.cross(y, u)
    b = np.degrees(np.arcsin(np.clip(n @ u, -1, 1)))
    a = np.degrees(np.arctan2(-(n @ v), n @ y))
    G = _rot(u, a) @ _rot(v, b)
    D = _rx(-de) @ _ry(-nu) @ G                # = Ry(det)
    return float(np.degrees(np.arctan2(D[0, 2], D[0, 0])))


def spin(nu, de, model="twist-free", psi=0.0, det=0.0, offset=0.0):
    """det [deg] under ``model``; ``det`` is the encoder value for "encoder"."""
    if model == "twist-free":
        return float(_wrap(offset - nu))
    if model == "cardan":
        return float(_wrap(offset + cardan_det(nu, de, psi)))
    if model == "encoder":
        return float(det)
    raise ValueError(f"spin model {model!r}: one of {SPIN_MODELS}")


def orientation(nu, de, det):
    return _ry(nu) @ _rx(de) @ _ry(det)


def pixel_positions(nu, de, det, step=1):
    """Lab positions (mm) of the pixel centres, shape (NX/step, NZ/step, 3)."""
    x = (np.arange(0, NX, step) - (NX - 1) / 2) * PIXEL
    z = (np.arange(0, NZ, step) - (NZ - 1) / 2) * PIXEL
    X, Z = np.meshgrid(x, z, indexing="ij")
    P = np.stack([X, np.full_like(X, DISTANCE), Z], -1)
    return P @ orientation(nu, de, det).T + PIVOT


def angles(P):
    """azimuth, elevation, 2 theta [deg] and distance [mm] of lab positions ``P`` (..., 3)."""
    P = np.asarray(P, float)
    r = np.linalg.norm(P, axis=-1)
    az = np.degrees(np.arctan2(P[..., 0], P[..., 2]))
    el = np.degrees(np.arctan2(P[..., 1], np.hypot(P[..., 0], P[..., 2])))
    tth = np.degrees(np.arccos(np.clip(P[..., 2] / r, -1, 1)))
    return az, el, tth, r


def outline(nu, de, det, n=40):
    """Lab positions of the module's edge, a closed loop starting at pixel (0, 0)."""
    s = np.linspace(-0.5, 0.5, n)
    w, h = NX * PIXEL, NZ * PIXEL
    edge = np.concatenate([np.stack([s * w, np.full(n, -h / 2)], 1), np.stack([np.full(n, w / 2), s * h], 1),
                           np.stack([-s * w, np.full(n, h / 2)], 1), np.stack([np.full(n, -w / 2), -s * h], 1)])
    edge = np.vstack([edge, edge[:1]])
    P = np.stack([edge[:, 0], np.full(len(edge), DISTANCE), edge[:, 1]], 1)
    return P @ orientation(nu, de, det).T + PIVOT


def summary(nu, de, det):
    """Numbers for one position: centre azimuth/elevation/2theta, ranges, distance."""
    P = pixel_positions(nu, de, det, step=8)
    az, el, tth, r = angles(P)
    normal = orientation(nu, de, det) @ np.array([0.0, 1.0, 0.0])
    caz, cel, ctth, cr = angles(PIVOT + DISTANCE * normal)
    # solid angle: pixel area * cos(incidence) / r^2, summed over the module
    omega = NX * NZ * PIXEL ** 2 * np.mean(np.abs(P @ normal) / r ** 3)
    return {"azimuth": float(caz), "elevation": float(cel), "2theta": float(ctth),
            "2theta min": float(tth.min()), "2theta max": float(tth.max()),
            "elevation min": float(el.min()), "elevation max": float(el.max()),
            "distance": float(cr), "solid angle [msr]": float(omega * 1e3)}


def reach(del_max=DEL_MAX, nu_step=2.0, bins=360):
    """Lowest elevation the module edge reaches per azimuth over all nu at
    del = del_max: (azimuth centres, elevation) - everything above is reachable."""
    az, el = [], []
    for nu in np.arange(-180, 180, nu_step):
        a, e, _, _ = angles(outline(nu, del_max, -nu, n=30))
        az.append(a)
        el.append(e)
    az, el = np.concatenate(az), np.concatenate(el)
    edges = np.linspace(-180, 180, bins + 1)
    idx = np.clip(np.digitize(az, edges) - 1, 0, bins - 1)
    low = np.full(bins, np.nan)
    np.fmin.at(low, idx, el)
    return 0.5 * (edges[1:] + edges[:-1]), low


def to_gamma_delta(az, el):
    """The THC app's (gamma, delta) plane, 0..180 each, as for data/otti.npy:
    azimuth >= 0 as is, azimuth < 0 folded to (-azimuth, 180 - elevation)."""
    az, el = np.asarray(az, float), np.asarray(el, float)
    neg = az < 0
    return np.where(neg, -az, az), np.where(neg, 180 - el, el)


def fit_cardan(nu, de, det_measured, psi_step=0.25):
    """Fit the cardan model to recorded positions: returns (psi, offset, rms)
    for det_measured ~ offset + cardan_det(nu, del, psi). det_measured is in
    eco's convention (otti_det readback + otti_det_rotation)."""
    nu, de, dm = (np.atleast_1d(np.asarray(a, float)) for a in (nu, de, det_measured))
    best = None
    for psi in np.arange(0, 180, psi_step):
        model = np.array([cardan_det(a, b, psi) for a, b in zip(nu, de)])
        d = np.radians(dm - model)
        offset = np.degrees(np.arctan2(np.sin(d).mean(), np.cos(d).mean()))
        rms = float(np.sqrt(np.mean(_wrap(dm - model - offset) ** 2)))
        if best is None or rms < best[2]:
            best = (float(psi), float(offset), rms)
    return best


# --- interactive panel ---------------------------------------------------------------


class OttiPanel:
    """Sliders for nu and del, the spin model, and two views: a sky map of
    what the module covers (polar, seen from above, beam to the right) and
    a top view of the arm. ``ax_top`` / ``ax_gd`` are optional axes of the THC
    app's grid that get the top view and the footprint in (gamma, delta);
    ``on_change`` is called after they were redrawn."""

    def __init__(self, envelope_npy=None, ax_top=None, ax_gd=None, on_change=None):
        import ipywidgets as w

        from bernina_prep.mpl_widget import FigureWidget

        self.envelope = None
        if envelope_npy is not None:
            try:
                x, _y1, _y2, y1s, _ = np.load(envelope_npy)
                self.envelope = (np.degrees(x), np.degrees(y1s))
            except OSError:
                pass
        self.ax_top, self.ax_gd, self.on_change = ax_top, ax_gd, on_change
        self._reach = reach(DEL_MAX, nu_step=3, bins=180)

        L = w.Layout
        st = {"description_width": "90px"}
        self.nu = w.FloatSlider(value=0, min=-180, max=180, step=1, description="otti_nu [°]",
                                continuous_update=False, layout=L(width="420px"), style=st)
        self.de = w.FloatSlider(value=20, min=0, max=60, step=0.5, description="otti_del [°]",
                                continuous_update=False, layout=L(width="420px"), style=st)
        self.model = w.ToggleButtons(options=[("twist-free (eco auto)", "twist-free"), ("cardan joint", "cardan"),
                                              ("encoder value", "encoder")], value="twist-free",
                                     description="detector turn", style={"description_width": "90px",
                                                                         "button_width": "150px"})
        self.psi = w.FloatSlider(value=0, min=0, max=180, step=1, description="gimbal axis ψ",
                                 continuous_update=False, layout=L(width="420px"), style=st)
        self.det = w.FloatText(value=0.0, description="det [°]", layout=L(width="180px"), style=st)
        self.offset = w.FloatText(value=0.0, description="det offset [°]", layout=L(width="180px"), style=st)
        self.colour = w.Dropdown(options=[("2θ", "2theta"), ("distance from sample", "distance"),
                                          ("elevation", "elevation")], value="2theta", description="colour",
                                 layout=L(width="260px"), style=st)
        self.grid = w.Checkbox(value=False, description="footprints over nu every 30°, del 0/15/30/44°",
                               indent=False, layout=L(width="auto"))
        self.readout = w.HTML()
        self.figure = FigureWidget((6.4, 6.0))
        self.fig = self.figure.fig
        for c in (self.nu, self.de, self.model, self.psi, self.det, self.offset, self.colour, self.grid):
            c.observe(lambda _c: self.update(), names="value")
        self.ui = w.VBox([
            w.HTML("<b>Otti detector</b> &mdash; where the JUNGFRAU module looks for given arm "
                   "rotations (geometry of eco's <code>calc_otti</code>)"),
            w.HBox([w.VBox([self.nu, self.de, self.model, self.psi,
                            w.HBox([self.det, self.offset]), self.colour, self.grid, self.readout,
                            w.HTML(f'<div style="font-size:11px;color:#666;white-space:normal;width:420px">'
                                   f'del above {DEL_MAX:g}° goes beyond the envelope of the original '
                                   'thc_angles app (orange), which this model reproduces at '
                                   f'del = {DEL_MAX:g}° (dashed). <i>twist-free</i>: the module does not '
                                   'turn about its normal (eco&#39;s <code>otti_det="auto"</code>); '
                                   '<i>cardan</i>: it hangs in a gimbal on the upward flange with the '
                                   'outer axis at azimuth ψ, and turns by ~del²/2·sin2(nu−ψ) more '
                                   '(ψ and the offset to be fitted from recorded encoder values: '
                                   '<code>otti.fit_cardan</code>); <i>encoder</i>: det as entered. '
                                   'The white dot is pixel (0, 0), the arrow the 1024-pixel '
                                   'direction.</div>')]),
                    self.figure.widget], layout=L(flex_flow="row wrap"))])
        self.update()

    def det_value(self):
        return spin(self.nu.value, self.de.value, self.model.value, self.psi.value,
                    self.det.value, self.offset.value)

    def update(self):
        nu, de = self.nu.value, self.de.value
        det = self.det_value()
        self.psi.disabled = self.model.value != "cardan"
        self.det.disabled = self.model.value != "encoder"
        self._draw_sky(nu, de, det)
        if self.ax_top is not None:
            self._draw_top(self.ax_top, nu, de, det)
        if self.ax_gd is not None:
            self._draw_gamma_delta(self.ax_gd, nu, de, det)
        s = summary(nu, de, det)
        twist = _wrap(det - spin(nu, de, "twist-free", offset=self.offset.value))
        self.readout.value = (
            '<div style="font-size:12px;line-height:1.5">'
            f'centre: azimuth {s["azimuth"]:.1f}°, elevation {s["elevation"]:.1f}°, 2θ {s["2theta"]:.1f}°<br>'
            f'2θ {s["2theta min"]:.1f}° &ndash; {s["2theta max"]:.1f}°, elevation '
            f'{s["elevation min"]:.1f}° &ndash; {s["elevation max"]:.1f}°<br>'
            f'distance {s["distance"]:.1f} mm, solid angle {s["solid angle [msr]"]:.1f} msr<br>'
            f'det {det:.2f}° (turn about the normal relative to twist-free: {twist:+.2f}°)</div>')
        self.figure.redraw()
        if self.on_change:
            self.on_change()

    # --- the sky map
    def _draw_sky(self, nu, de, det):
        fig = self.fig
        fig.clf()
        ax = fig.add_subplot(111, projection="polar")
        ax.set_theta_zero_location("E")          # seen from above, as the top view: beam (z) right, x up
        rmax = 62
        # 2 theta of every direction: cos 2theta = cos(el) cos(az)
        th, rr = np.meshgrid(np.radians(np.linspace(-180, 180, 241)), np.linspace(0, rmax, 63))
        tth = np.degrees(np.arccos(np.cos(np.radians(90 - rr)) * np.cos(th)))
        ax.contour(th, rr, tth, levels=np.arange(30, 160, 10), colors="0.85", linewidths=0.5)
        cs = ax.contour(th, rr, tth, levels=np.arange(40, 150, 20), colors="0.6", linewidths=0.7)
        ax.clabel(cs, fmt="2θ %d°", fontsize=7, inline_spacing=2)
        if self.envelope is not None:
            az, el = self.envelope
            order = np.argsort(az)
            ax.plot(np.radians(az[order]), 90 - el[order], color="darkorange", lw=2, label="envelope (thc_angles)")
        raz, rlow = self._reach
        ax.plot(np.radians(raz), 90 - rlow, color="0.3", ls="--", lw=1, label=f"model, del = {DEL_MAX:g}°")
        if self.grid.value:
            import matplotlib as mpl
            cmap = mpl.colormaps["plasma"]
            model = self.model.value if self.model.value != "encoder" else "twist-free"
            for d in (0, 15, 30, DEL_MAX):
                col = cmap(0.15 + 0.7 * d / DEL_MAX)
                for i, n in enumerate(np.arange(-180, 180, 30)):
                    a, e, _, _ = angles(outline(n, d, spin(n, d, model, self.psi.value, 0, self.offset.value)))
                    ax.plot(np.unwrap(np.radians(a)), 90 - e, color=col, lw=0.8, alpha=0.75,
                            label=f"del {d:g}°, nu every 30°" if i == 0 else None)
        P = pixel_positions(nu, de, det, step=16)
        az, el, tt, r = angles(P)
        val = {"2theta": tt, "distance": r, "elevation": el}[self.colour.value]
        thp = np.unwrap(np.unwrap(np.radians(az), axis=0), axis=1)
        mesh = ax.pcolormesh(thp, 90 - el, val, shading="gouraud", cmap="viridis")
        fig.colorbar(mesh, ax=ax, shrink=0.6, pad=0.08,
                     label={"2theta": "2θ [°]", "distance": "distance [mm]", "elevation": "elevation [°]"}[self.colour.value])
        a, e, _, _ = angles(outline(nu, de, det))
        ax.plot(np.unwrap(np.radians(a)), 90 - e, color="white", lw=3.5)
        ax.plot(np.unwrap(np.radians(a)), 90 - e, color="k", lw=1.5, label=f"nu {nu:g}°, del {de:g}°")
        a0, e0, _, _ = angles(P[0, 0])
        a1, e1, _, _ = angles(P[len(P) // 3, 0])
        ax.plot(np.radians(a0), 90 - e0, "o", mfc="white", mec="k", ms=6)
        ax.annotate("", xy=(np.radians(a1), 90 - e1), xytext=(np.radians(a0), 90 - e0),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
        ax.set_rlim(0, rmax)
        ax.set_rticks([10, 20, 30, 40, 50, 60])
        ax.set_yticklabels([f"{90 - t}°" for t in (10, 20, 30, 40, 50, 60)], fontsize=7, color="0.4")
        ax.set_rlabel_position(200)
        ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(["0°\nbeam →", "45°", "90° (x)", "135°", "180°", "-135°", "-90°", "-45°"], fontsize=8)
        ax.set_title(f"sky above the sample (elevation {90 - rmax}°…90°)", fontsize=9)
        ax.legend(loc="upper left", bbox_to_anchor=(-0.18, -0.07), fontsize=7, frameon=False, ncol=2)

    # --- in the THC grid: top view and (gamma, delta)
    def _draw_top(self, ax, nu, de, det):
        ax.clear()
        ax.set_title("Otti (motorized in-vacuum)")
        from bernina_prep.thc_angles import TOPFLANGE_R
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(PIVOT[2] + TOPFLANGE_R * np.sin(t), PIVOT[0] + TOPFLANGE_R * np.cos(t), color="grey", alpha=0.8)
        ax.annotate("", xy=(120, 0), xytext=(-120, 0), arrowprops=dict(arrowstyle="->", color="red", lw=1))
        ax.text(110, -14, "beam", color="red", fontsize=8, ha="right")
        ax.plot(0, 0, "k+", ms=10)
        ax.plot(PIVOT[2], PIVOT[0], "o", color="m", ms=5)
        normal = orientation(nu, de, det) @ np.array([0, DISTANCE, 0])
        ax.plot([PIVOT[2], PIVOT[2] + normal[2]], [PIVOT[0], PIVOT[0] + normal[0]], color="m", lw=1)
        o = outline(nu, de, det)
        ax.fill(o[:, 2], o[:, 0], color="darkorange", alpha=0.35)
        ax.plot(o[:, 2], o[:, 0], color="darkorange")
        ax.plot(o[0, 2], o[0, 0], "o", mfc="white", mec="k", ms=5)
        ax.set_xlabel("z, along the beam (mm)")
        ax.set_ylabel("x (mm), seen from above")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlim(-140, 140)
        ax.set_ylim(-110, 190)

    def _draw_gamma_delta(self, ax, nu, de, det):
        for line in getattr(self, "_gd_artists", []):
            line.remove()
        a, e, _, _ = angles(outline(nu, de, det))
        g, d = to_gamma_delta(a, e)
        jump = np.abs(np.diff(g)) > 20                 # where the fold changes branch
        g, d = np.insert(g, np.where(jump)[0] + 1, np.nan), np.insert(d, np.where(jump)[0] + 1, np.nan)
        self._gd_artists = ax.fill(g, d, color="darkorange", alpha=0.3) + \
            ax.plot(g, d, color="saddlebrown", lw=1.2, label="detector at nu, del")
