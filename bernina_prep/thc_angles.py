"""Accessible diffraction angles for Bernina's THC sample chamber.

Bernina's optional detector-arm hardware -- the manual in-vacuum detector,
the beryllium exit window, and the motorized in-vacuum arm ("Otti") -- each
physically obstructs part of the (gamma, delta) detector-angle range a
reflection could otherwise reach. This module reproduces the geometric
envelope calculation from the original ``thc_angles`` Binder app
(https://github.com/swissfel/thc_angles): given the sample-to-aperture
geometry and, for the rotatable hardware, the flange rotation angle beta, it
projects the aperture boundary onto scattering angles gamma (azimuth) /
delta (elevation) to find the accessible envelope.

The reference hardware photos shown in the original app are intentionally
not reproduced here -- they hot-link an externally, personally hosted image
store, which is both fragile and a poor fit for a fast, offline-friendly
Binder build. The geometry math and interactive rotation sliders are
unchanged; see the original notebook for the annotated photos.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import det
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# shared geometry primitives
# ---------------------------------------------------------------------------


def rotate_detector_2d(l=80.0, b=40.0, e_det=10.0, beta=0.0):
    """Corners of a rectangular detector, rotated by beta around the flange
    center; the long detector axis lies along x for beta = 0."""
    e1 = np.array([b / 2, l / 2 + e_det, 0])
    e2 = np.array([-b / 2, l / 2 + e_det, 0])
    e3 = np.array([-b / 2, -l / 2 + e_det, 0])
    e4 = np.array([b / 2, -l / 2 + e_det, 0])
    r_beta = Rotation.from_euler("z", [beta], degrees=False)
    e1, e2, e3, e4 = r_beta.apply([e1, e2, e3, e4])
    return np.array([e1[:2], e2[:2], e3[:2], e4[:2]])


def radii_rectangle(l=80.0, b=40.0, e_det=10.0, e_sample=np.array([0, 46]), gam=np.array([0]), beta=0.0):
    """Distance from the sample to where a line of sight at angle(s) ``gam``
    exits through the rotated rectangular detector aperture, for both the
    near and far intersection (min/max radius)."""
    gam = -gam  # x-axis defined in opposite direction
    s = e_sample
    s1 = np.zeros((len(gam), 2, 2)) + s

    s1[:, 0, 0] = s1[:, 0, 0] - 250 * np.array(np.cos(gam))
    s1[:, 0, 1] = s1[:, 0, 1] + 250 * np.array(np.sin(gam))
    s1[:, 1, 0] = s1[:, 1, 0] + 250 * np.array(np.cos(gam))
    s1[:, 1, 1] = s1[:, 1, 1] - 250 * np.array(np.sin(gam))

    e1, e2, e3, e4 = rotate_detector_2d(l=l, b=b, e_det=e_det, beta=beta)
    es = np.array([[e1, e2], [e1, e4], [e2, e3], [e3, e4]])
    M = np.array(
        [
            np.array([np.divide(det([p1 - p3, p3 - p4]), det([p1 - p2, p3 - p4])) for p3, p4 in zip(s1[:, 0], s1[:, 1])])
            for p1, p2 in es
        ]
    )
    M[~((M < 1.0) & (M > 0.0))] = np.nan

    p1s = es[:, 0]
    p2s = es[:, 1]
    intersects = np.array([m[:, None] * (p2s - p1s) + p1s for m in M.T])
    radii_2d = np.linalg.norm(intersects - s, axis=2) * (intersects - s)[:, :, 1] / -abs((intersects - s)[:, :, 1])
    # some gamma have no valid intersection at all (all-NaN slice); that's
    # expected here, not a bug -- it just means "nothing accessible there"
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        radii_2d = np.array([np.nanmax(radii_2d, axis=1), np.nanmin(radii_2d, axis=1)])
    return radii_2d


def radii(gamma, x0, y0, r):
    """Two intersections of a line from the origin at angle gamma with the
    circle (x0, y0, r) defining a circular aperture. Negative values under
    the sqrt (no intersection at that gamma) intentionally yield NaN."""
    with np.errstate(invalid="ignore"):
        r1 = np.cos(gamma) * x0 + np.sin(gamma) * y0 + np.sqrt((np.cos(gamma) * x0 + np.sin(gamma) * y0) ** 2 - (x0**2 + y0**2 - r**2))
        r2 = np.cos(gamma) * x0 + np.sin(gamma) * y0 - np.sqrt((np.cos(gamma) * x0 + np.sin(gamma) * y0) ** 2 - (x0**2 + y0**2 - r**2))
    return (r1, r2)


def excentricity(beta, e_window, e_sample=(0, -46)):
    """Offset of a circular aperture's center from the sample, for a flange
    rotated by beta."""
    x0 = np.cos(beta) * e_window - e_sample[0]
    y0 = np.sin(beta) * e_window - e_sample[1]
    return x0, y0


def phi(r, h):
    """Scattering angle measured from vertical, for a point at radius r on a
    plane height h above the sample."""
    return np.arctan(r / h)


def delta_from_beta(beta, gamma, r, e_window, e_sample, h):
    """Elevation angle delta (deg) of the near/far edges of a circular
    aperture, as seen from the sample, for detector azimuth(s) gamma."""
    x0, y0 = excentricity(beta, e_window, e_sample)
    radiis = radii(gamma, x0, y0, r)
    return [90 - np.rad2deg(phi(radiius, h)) for radiius in radiis]


def circle(beta, e_window, e_sample, r):
    """(x, y) outline of a circular aperture of radius r, offset by a flange
    rotated by beta, relative to the sample."""
    x0, y0 = excentricity(beta, e_window, e_sample)
    t = np.linspace(0, 2 * np.pi, 3601, endpoint=True)
    x = r * np.cos(t) - x0
    y = r * np.sin(t) - y0
    return x, y


# ---------------------------------------------------------------------------
# in-vacuum detector: rectangular aperture, rotates with the detector flange
# ---------------------------------------------------------------------------

INVAC_L = 80.0
INVAC_B = 40.0
INVAC_E_DET = 36.0
INVAC_H = 42.0
INVAC_E_SAMPLE = np.array([0.0, 46.0])


def invac_envelope(beta_deg=0.0, n_gamma=121):
    """Accessible (gamma, delta) envelope of the in-vacuum detector at a
    fixed flange rotation beta_deg."""
    gamma = np.linspace(0.001, np.pi + 0.001, n_gamma)
    r = radii_rectangle(l=INVAC_L, b=INVAC_B, e_det=INVAC_E_DET, e_sample=INVAC_E_SAMPLE, gam=gamma, beta=np.deg2rad(beta_deg))
    deltamin = 90 - np.rad2deg(phi(r[0], INVAC_H))
    deltamax = 90 - np.rad2deg(phi(r[1], INVAC_H))
    return np.rad2deg(gamma), deltamin, deltamax


def invac_footprint(beta_deg=0.0):
    """Detector rectangle outline (relative to the sample) at flange
    rotation beta_deg, for the top-view schematic plot."""
    edges = rotate_detector_2d(l=INVAC_L, b=INVAC_B, e_det=INVAC_E_DET, beta=np.deg2rad(beta_deg)) - INVAC_E_SAMPLE
    return np.vstack([edges, edges[0]])


def invac_envelope_full(n_beta=60, n_gamma=60):
    """Widest accessible envelope of the in-vacuum detector over all flange
    rotations (the dashed reference curve)."""
    betas = np.linspace(0, 2 * np.pi, n_beta)
    gammas = np.linspace(0.001, np.pi + 0.001, n_gamma)
    r = np.array(
        [radii_rectangle(l=INVAC_L, b=INVAC_B, e_det=INVAC_E_DET, e_sample=INVAC_E_SAMPLE, gam=gammas, beta=beta) for beta in betas]
    )
    rs = np.array([np.nanmax(r, axis=(0, 1)), np.nanmin(r, axis=(0, 1))])
    deltamin = 90 - np.rad2deg(phi(rs[0], INVAC_H))
    deltamax = 90 - np.rad2deg(phi(rs[1], INVAC_H))
    return np.rad2deg(gammas), deltamin, deltamax


# ---------------------------------------------------------------------------
# beryllium exit window: circular aperture, rotates with the top flange
# ---------------------------------------------------------------------------

WINDOW_H = 36.0
WINDOW_R = 70.0 / 2
WINDOW_E = 30.0
# shared by the Be window and the (fixed) top-flange boundary below
E_SAMPLE = (0.0, -46.0)


def window_envelope(beta_deg=0.0, n_gamma=361):
    """Accessible (gamma, delta) envelope through the Be window at a fixed
    top-flange rotation beta_deg."""
    gamma = np.linspace(0, np.pi, n_gamma)
    deltamin, deltamax = delta_from_beta(np.deg2rad(beta_deg - 90), gamma, WINDOW_R, WINDOW_E, E_SAMPLE, WINDOW_H)
    return np.rad2deg(gamma), np.asarray(deltamin), np.asarray(deltamax)


def window_circle(beta_deg=0.0):
    """Be window outline (relative to the sample) at flange rotation
    beta_deg, for the top-view schematic plot."""
    return circle(np.deg2rad(beta_deg - 90), WINDOW_E, E_SAMPLE, WINDOW_R)


def window_envelope_full(n_beta=360, n_gamma=361):
    """Widest accessible envelope through the Be window over all top-flange
    rotations (the dashed reference curve)."""
    gamma = np.linspace(0, np.pi, n_gamma)
    betas = np.linspace(0, 2 * np.pi, n_beta)
    ds = np.array([delta_from_beta(beta, gamma, WINDOW_R, WINDOW_E, E_SAMPLE, WINDOW_H) for beta in betas])
    dmins = np.array([np.nanmin(dmin) for dmin in ds[:, 0, :].T])
    dmaxs = np.array([np.nanmax(dmax) for dmax in ds[:, 1, :].T])
    return np.rad2deg(gamma), dmins, dmaxs


# ---------------------------------------------------------------------------
# top flange (CF250): fixed circular boundary, does not rotate
# ---------------------------------------------------------------------------

TOPFLANGE_H = 102.0
TOPFLANGE_R = 248.5 / 2
TOPFLANGE_E = 0.0


def top_flange_envelope(n_gamma=361):
    """(gamma, delta) boundary of the CF250 top-flange opening -- the
    hard limit the Be window envelope sits inside of."""
    gamma = np.linspace(0, np.pi, n_gamma)
    deltamin, deltamax = delta_from_beta(0.0, gamma, TOPFLANGE_R, TOPFLANGE_E, E_SAMPLE, TOPFLANGE_H)
    return np.rad2deg(gamma), np.asarray(deltamin), np.asarray(deltamax)


def top_flange_circle():
    """Top-flange opening outline (relative to the sample), for the
    top-view schematic plot."""
    return circle(0.0, TOPFLANGE_E, E_SAMPLE, TOPFLANGE_R)


# ---------------------------------------------------------------------------
# Otti (motorized in-vacuum arm): precomputed envelope
# ---------------------------------------------------------------------------


def load_otti_envelope(npy_path):
    """Load the precomputed accessible-angle envelope for Otti.

    Unlike the other two, Otti's envelope isn't a simple aperture
    projection with a closed-form geometry (it comes from CAD/raytracing in
    the original thc_angles tool), so it ships as a precomputed array
    (``data/otti.npy``) rather than being derived live here. Returns
    ``(gamma1, delta1, gamma2, delta2)`` in degrees, split at gamma = 0 the
    same way the original app does.
    """
    x, _y1, _y2, y1s, _y2_1 = np.load(npy_path)
    gamma2 = np.rad2deg(-x[x < 0])
    delta2 = 180 - np.rad2deg(y1s)[x < 0]
    gamma1 = np.rad2deg(x[~(x < 0)])
    delta1 = np.rad2deg(y1s[~(x < 0)])
    return gamma1, delta1, gamma2, delta2


# ---------------------------------------------------------------------------
# interactive app
# ---------------------------------------------------------------------------


class ThcAnglesApp:
    """Accessible-diffraction-angle explorer for Bernina's THC detector-arm
    hardware.

    Three columns -- in-vacuum detector, Be window, Otti (motorized
    in-vacuum arm) -- each show the aperture geometry relative to the
    sample (top row) and the resulting accessible (gamma, delta) envelope
    (bottom row, solid curve = current rotation, dashed = widest envelope
    over all rotations). The first two columns are rotatable via sliders;
    Otti's envelope is precomputed (``data/otti.npy``) and shown as-is.
    """

    def __init__(self, otti_npy: str | Path | None = None):
        self.otti_npy = Path(otti_npy) if otti_npy else None

        plt.ioff()
        self.fig, self.ax = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
        self.canvas = self.fig.canvas

        self.slider_invac = widgets.IntSlider(
            value=0, min=0, max=360, step=15, description="Rotation (°)", continuous_update=False
        )
        self.slider_window = widgets.IntSlider(
            value=0, min=0, max=360, step=15, description="Rotation (°)", continuous_update=False
        )
        self.slider_invac.observe(self._on_invac_change, "value")
        self.slider_window.observe(self._on_window_change, "value")

        self._setup_axes()
        self._draw_invac(0.0)
        self._draw_window(0.0)
        self._draw_static()

        col_window = widgets.VBox([widgets.Label("Be window flange rotation"), self.slider_window])
        col_invac = widgets.VBox([widgets.Label("In-vacuum detector rotation"), self.slider_invac])
        controls = widgets.HBox([col_window, col_invac], layout=widgets.Layout(justify_content="center"))
        self.ui = widgets.VBox([self.canvas, controls])

    def _setup_axes(self):
        titles = ["In-vacuum detector", "Be window", "Otti (motorized in-vacuum)"]
        for a, title in zip(self.ax[0], titles):
            a.set_title(title)
            a.set_xlabel("x (mm)")
            a.set_ylabel("y (mm)")
            a.set_aspect("equal", adjustable="datalim")
            a.plot(0, 0, "k+")
        for a in self.ax[1]:
            a.set_xlabel("Azimuth γ (°)")
            a.set_ylabel("Elevation δ (°)")
            a.set_xlim(0, 180)
            a.set_ylim(0, 180)
            a.set_xticks([0, 30, 60, 90, 120, 150, 180])
            a.set_yticks([0, 45, 90, 135, 180])
            a.tick_params(direction="in")

    def _draw_invac(self, beta_deg):
        gamma, deltamin, deltamax = invac_envelope(beta_deg)
        footprint = invac_footprint(beta_deg)
        if not hasattr(self, "_invac_lines"):
            (l1,) = self.ax[1][0].plot(gamma, deltamin, color="seagreen", label="Accessible diffraction angles")
            (l2,) = self.ax[1][0].plot(gamma, deltamax, color="seagreen")
            (rect,) = self.ax[0][0].plot(footprint[:, 0], footprint[:, 1], color="seagreen")
            self._invac_lines = (l1, l2, rect)
        else:
            l1, l2, rect = self._invac_lines
            l1.set_ydata(deltamin)
            l2.set_ydata(deltamax)
            rect.set_xdata(footprint[:, 0])
            rect.set_ydata(footprint[:, 1])
        self.ax[0][0].relim()
        self.ax[0][0].autoscale_view()

    def _draw_window(self, beta_deg):
        gamma, deltamin, deltamax = window_envelope(beta_deg)
        x, y = window_circle(beta_deg)
        if not hasattr(self, "_window_lines"):
            (l1,) = self.ax[1][1].plot(gamma, deltamin, color="royalblue", label="Accessible diffraction angles")
            (l2,) = self.ax[1][1].plot(gamma, deltamax, color="royalblue")
            (circ,) = self.ax[0][1].plot(x, y, color="royalblue")
            self._window_lines = (l1, l2, circ)
        else:
            l1, l2, circ = self._window_lines
            l1.set_ydata(deltamin)
            l2.set_ydata(deltamax)
            circ.set_xdata(x)
            circ.set_ydata(y)
        self.ax[0][1].relim()
        self.ax[0][1].autoscale_view()

    def _draw_static(self):
        # CF250 top-flange boundary: hard limit the Be window envelope sits inside of
        x_t, y_t = top_flange_circle()
        self.ax[0][1].plot(x_t, y_t, color="grey", alpha=0.8)
        g_t, dmin_t, dmax_t = top_flange_envelope()
        self.ax[1][1].plot(g_t, dmin_t, color="grey", alpha=0.8, label="CF250 flange boundary")
        self.ax[1][1].plot(g_t, dmax_t, color="grey", alpha=0.8)

        # dashed reference: widest envelope over all rotations
        g_fi, dmin_fi, dmax_fi = invac_envelope_full()
        self.ax[1][0].plot(g_fi, dmin_fi, color="seagreen", linestyle=(5, (4, 4)))
        self.ax[1][0].plot(g_fi, dmax_fi, color="seagreen", linestyle=(5, (4, 4)))
        g_fw, dmin_fw, dmax_fw = window_envelope_full()
        self.ax[1][1].plot(g_fw, dmin_fw, color="royalblue", linestyle=(5, (4, 4)))
        self.ax[1][1].plot(g_fw, dmax_fw, color="royalblue", linestyle=(5, (4, 4)))

        self.ax[0][2].axis("off")
        self.ax[0][2].text(
            0.5, 0.5, "geometry is CAD-derived,\nnot parametrized here",
            ha="center", va="center", fontsize=9, wrap=True, transform=self.ax[0][2].transAxes,
        )
        if self.otti_npy and self.otti_npy.exists():
            g1, d1, g2, d2 = load_otti_envelope(self.otti_npy)
            self.ax[1][2].plot(g1, d1, color="darkorange", label="Accessible diffraction angles")
            self.ax[1][2].plot(g2, d2, color="darkorange")
        else:
            self.ax[1][2].text(
                0.5, 0.5, f"{self.otti_npy or 'otti.npy'} not found",
                ha="center", va="center", transform=self.ax[1][2].transAxes,
            )

        self.ax[1][0].legend(frameon=False, loc="upper center")
        self.ax[1][1].legend(frameon=False, loc="upper center")
        self.ax[1][2].legend(frameon=False, loc="upper center")

    def _on_invac_change(self, change):
        self._draw_invac(change["new"])
        self.canvas.draw_idle()

    def _on_window_change(self, change):
        self._draw_window(change["new"])
        self.canvas.draw_idle()
