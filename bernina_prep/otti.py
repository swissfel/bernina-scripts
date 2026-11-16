"""Otti, the motorized detector arm on the THC lid: the interactive panel.

The geometry (arm, gimbal models, reach, gamma/delta) is
``escape.bernina.otti``; this module adds the panel the THC app shows.
"""
from __future__ import annotations

import numpy as np
from escape.bernina.otti import *  # noqa: F401,F403
from escape.bernina.otti import _wrap  # noqa: F401


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
