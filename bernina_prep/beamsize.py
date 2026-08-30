"""Gaussian x-ray beam size / divergence calculations and an ipywidgets app.

Given a measured beam FWHM at one screen position, this infers the focus
waist and Rayleigh range and propagates a proper Gaussian beam profile
w(z) = w0 * sqrt(1 + ((z - z0) / zR)**2) through the beamline, rather than
linearly extrapolating the measured point to zero width at the focus (which
is not physical: a real Gaussian beam has finite width w0 at its waist).
"""
from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

W_TO_FWHM = 2 * np.sqrt(2 * np.log(2))  # FWHM = W_TO_FWHM * sigma
# far-field divergence of the FWHM *diameter* is sqrt(2 ln 2) times the
# divergence of the 1/e^2 *radius* w(z); NOT the sigma->FWHM factor above.
FWHM_DIAM_TO_1E2_RADIUS = np.sqrt(2 * np.log(2))


def energy_to_wavelength_mm(energy_kev: float) -> float:
    return 1.23984193e-6 / energy_kev


def solve_focus_from_fwhm(w_test_um: float, z_test: float, z0: float, energy_kev: float):
    """Given a measured FWHM spot size at distance z_test from focus z0,
    infer (w0_um, zr_mm, div_fwhm_mrad) assuming diffraction-limited,
    far-field propagation from the focus.
    """
    dist = abs(z_test - z0) or 1e-9
    div_fwhm_rad = 2 * np.arctan((w_test_um * 1e-3 / 2) / dist)
    wl_mm = energy_to_wavelength_mm(energy_kev)
    theta_1e2_rad = div_fwhm_rad / FWHM_DIAM_TO_1E2_RADIUS
    w0_mm = wl_mm / (np.pi * theta_1e2_rad) if theta_1e2_rad != 0 else 0.0
    zr_mm = (np.pi * w0_mm**2) / wl_mm if wl_mm != 0 else 0.0
    return w0_mm * 1e3, zr_mm, div_fwhm_rad * 1000


def beam_width(z, z0: float, w0_mm: float, zr_mm: float):
    """Gaussian beam 1/e^2 width at position(s) z, in the same units as w0_mm."""
    if zr_mm == 0:
        return np.zeros_like(np.asarray(z, dtype=float)) if hasattr(z, "__len__") else 0.0
    return w0_mm * np.sqrt(1 + ((np.asarray(z, dtype=float) - z0) / zr_mm) ** 2)


def load_beamline_defaults(csv_path) -> list[tuple[str, float]]:
    """Read a beamline.csv (columns Comp, Z_mm, X_um, Y_um) into
    [(name, z_mm), ...] component markers for the plot."""
    df = pd.read_csv(csv_path)
    return list(zip(df["Comp"], df["Z_mm"]))


class GeometricBeamApp:
    """Interactive x/y beam-size-vs-position tool.

    Enter a measured FWHM spot size at a known distance from focus for each
    axis; the app infers the waist/Rayleigh range and plots the beam
    envelope alongside named beamline components (loaded from a CSV, or
    added manually).
    """

    def __init__(self, beamline_csv: str | Path | None = None):
        self.is_updating = False
        self.beamline_csv = Path(beamline_csv) if beamline_csv else None

        self.lbl_layout = widgets.Layout(width="66%")
        self.inp_layout = widgets.Layout(width="33%")

        self.inputs = {}
        self.derived = {}

        for dim in ["x", "y"]:
            z_t_def = -2600.0 if dim == "x" else -3350.0
            self.inputs[dim] = {
                "z0": widgets.FloatText(value=0.0),
                "z_test": widgets.FloatText(value=z_t_def),
                "w_test": widgets.FloatText(value=500.0),
                "energy": widgets.FloatText(value=12.4),
            }
            self.derived[dim] = {
                "div": widgets.FloatText(value=0.0, disabled=True),
                "w0": widgets.FloatText(value=0.0, disabled=True),
                "zr": widgets.FloatText(value=0.0, disabled=True),
            }
            for w in self.inputs[dim].values():
                w.observe(self.on_global_input_change, "value")

        self.z_rows_container = widgets.VBox([], layout={"max_height": "400px", "overflow_y": "auto"})
        self.z_rows = []

        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        self.canvas = self.fig.canvas

        self.setup_ui()
        self.load_defaults()
        self.update_plot()

    def on_global_input_change(self, _):
        if self.is_updating:
            return
        self.update_plot()

    def solve_optics(self, dim):
        d = self.inputs[dim]
        w0_um, zr_mm, div_mrad = solve_focus_from_fwhm(
            d["w_test"].value, d["z_test"].value, d["z0"].value, d["energy"].value
        )
        self.derived[dim]["div"].value = round(div_mrad, 4)
        self.derived[dim]["w0"].value = round(w0_um, 4)
        self.derived[dim]["zr"].value = round(zr_mm, 4)
        return w0_um, zr_mm

    def create_param_row(self, label, widget):
        return widgets.HBox([widgets.Label(label, layout=self.lbl_layout), widget], layout={"width": "100%"})

    def setup_ui(self):
        self.add_row_btn = widgets.Button(description="+ Add Component", button_style="success")
        self.rem_row_btn = widgets.Button(description="- Remove Last", button_style="danger")
        self.export_btn = widgets.Button(description="Export CSV", button_style="info")

        self.add_row_btn.on_click(lambda _: self.add_z_row())
        self.rem_row_btn.on_click(lambda _: self.remove_z_row())
        self.export_btn.on_click(self.export_to_csv)

        panels = []
        labels_in = ["Focus $z_0$ [mm]", "Test $z$ [mm]", "Test FWHM [µm]", "Energy [keV]"]
        labels_der = ["Div (FWHM) [mrad]", "Waist $w_0$ [µm]", "Rayleigh $z_R$ [mm]"]

        for dim in ["x", "y"]:
            title = "Horizontal (X)" if dim == "x" else "Vertical (Y)"
            rows = [widgets.HTML(f"<b>{title}</b>")]
            for lbl, key in zip(labels_in, self.inputs[dim].keys()):
                rows.append(self.create_param_row(lbl, self.inputs[dim][key]))
            rows.append(widgets.HTML("<i>Calculated Optics</i>"))
            for lbl, key in zip(labels_der, self.derived[dim].keys()):
                rows.append(self.create_param_row(lbl, self.derived[dim][key]))
            panels.append(
                widgets.VBox(
                    rows,
                    layout={"border": "1px solid #ccc", "padding": "10px", "margin": "5px", "width": "320px"},
                )
            )

        left_box = widgets.VBox(
            [
                widgets.HBox(panels),
                widgets.HTML("<b>Beamline Components</b>"),
                self.z_rows_container,
                widgets.HBox([self.add_row_btn, self.rem_row_btn, self.export_btn]),
            ],
            layout={"min_width": "660px"},
        )

        self.ui = widgets.HBox([left_box, self.canvas], layout={"width": "100%", "flex_flow": "row wrap"})

    def add_z_row(self, initial_z=0.0, initial_name="New"):
        idx = len(self.z_rows)
        name_in = widgets.Text(value=initial_name, layout={"width": "140px"})
        z_in = widgets.FloatText(value=initial_z, layout={"width": "80px"})
        x_in = widgets.FloatText(value=0, layout={"width": "80px"}, disabled=True)
        y_in = widgets.FloatText(value=0, layout={"width": "80px"}, disabled=True)

        z_in.observe(lambda _: self.update_plot(), "value")

        row_ui = widgets.HBox([name_in, z_in, x_in, y_in])
        self.z_rows.append({"row": row_ui, "name": name_in, "z": z_in, "x": x_in, "y": y_in})
        self.z_rows_container.children = [r["row"] for r in self.z_rows]
        self.update_plot()

    def remove_z_row(self, _=None):
        if self.z_rows:
            self.z_rows.pop()
            self.z_rows_container.children = [r["row"] for r in self.z_rows]
            self.update_plot()

    def export_to_csv(self, _=None):
        data = [[r["name"].value, r["z"].value, r["x"].value, r["y"].value] for r in self.z_rows]
        pd.DataFrame(data, columns=["Comp", "Z_mm", "X_um", "Y_um"]).to_csv("beamline.csv", index=False)

    def update_plot(self, _=None):
        self.ax.clear()
        z_range = np.linspace(-4500, 4500, 2000)
        self.is_updating = True

        for dim, color in zip(["x", "y"], ["#1f77b4", "#d62728"]):
            w0_um, zr_mm = self.solve_optics(dim)
            z0 = self.inputs[dim]["z0"].value
            widths_um = beam_width(z_range, z0, w0_um, zr_mm)

            self.ax.plot(z_range, widths_um, color=color, lw=2, label=f"{dim.upper()} Beam")
            for row in self.z_rows:
                row[dim].value = round(float(beam_width(row["z"].value, z0, w0_um, zr_mm)), 2)

        self.is_updating = False

        colors = plt.get_cmap("tab20", max(len(self.z_rows), 1))
        for i, row in enumerate(self.z_rows):
            self.ax.axvline(row["z"].value, color=colors(i), alpha=0.5, label=row["name"].value)

        self.ax.set_xlabel("z [mm]")
        self.ax.set_ylabel("Size [µm]")
        self.ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def load_defaults(self):
        if self.beamline_csv and self.beamline_csv.exists():
            for name, z in load_beamline_defaults(self.beamline_csv):
                self.add_z_row(z, name)
