"""The diffraction reflections app (``apps/diffraction_reflections.ipynb``).

``ReflectionsApp().ui`` is the whole interface: sample and constraint
inputs, the solutions table, the 3D view and the tools (reflection table,
reachable reflections, detector map, crystal and grazing-incidence checks).
All calculations are in ``bernina_prep.diffraction``, ``crystal``,
``grazing`` and ``thc_angles``; this module only builds and wires widgets.

Rules that keep it rendering under voila: px sizes only, core
widgets only, explicit flex basis for an Output beside something, no LaTeX
in descriptions, no blocking in callbacks.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display

from bernina_prep import crystal as cry
from bernina_prep import detector_view as dv
from bernina_prep import diffraction as dif
from bernina_prep import grazing as gra
from bernina_prep.mpl_widget import FigureWidget

HERE = Path(__file__).resolve().parent
VIEWER_HTML = HERE / "diffractometer_3d.html"
PRESETS_JSON = HERE / "presets.json"

L = widgets.Layout
UC_W, FIELD_W, FILTER_W, HKL_W, LABEL_W = "150px", "86px", "52px", "98px", "70px"
ROW_LABEL_W, COL_W = 70, 84
SELECTED_BG = "#cfe8fb"
HKL_DECIMALS = 4
INPUT_W = "420px"
SEL_CONFIG, SEL_ZERO = "config", "zero"

DEFAULT_SAMPLE = {
    "uc_a": 5.431, "uc_b": 5.431, "uc_c": 5.431, "uc_alpha": 90.0, "uc_beta": 90.0, "uc_gamma": 90.0,
    "surface_normal": "(1,1,1)", "reference": "(1,0,0)", "space_group": "Fd-3m",
    "orientations": "(1,1,1) > (1,0,0)\n(1,1,-1) > (0,1,0)",
}
DEFAULT_HELD = {"eta": 0.0, "chi": 0.0, "phi": 0.0}

HELP_HTML = """
<div style="font-size:12.5px;line-height:1.45;max-width:1100px">
<p><b>What it does.</b> Diffractometer angles for an (h,k,l) reflection in Bernina's
six-circle You geometry (mu, eta, chi, phi on the sample; gamma, delta on the
detector), solved with <a href="https://github.com/swissfel/hklforge" target="_blank">hklforge</a>
&mdash; the same calculation as eco's <code>recspace</code> on the beamline.</p>

<p><b>Axes.</b> Lab directions are SwissFEL/Bernina axes: z along the beam, y up,
x to the left looking along the beam. An orientation line
<code>(h,k,l) &gt; (x,y,z)</code> (or <code>&gt; beam</code>, <code>&gt; up</code>,
<code>&gt; x+</code>) says where that crystal direction points with all angles at 0.
A sample mounted on the phi stage faces along x+ with all angles 0.</p>

<p><b>Surface normal and reference</b> are separate (h,k,l) directions: the surface
normal sets betain/betaout (incidence and exit angles to the surface), the
reference sets alpha/beta/psi. Leave the surface normal empty if there is none.</p>

<p><b>Crystal.</b> A hexagonal or trigonal cell given by a and c has gamma = 120 &mdash;
<i>crystal system</i> fills the angles in. The <i>space group</i> (or a CIF file)
lets the app warn when the target reflection is forbidden and list the allowed
ones; with a CIF also their intensities.</p>

<p><b>Solving.</b> Tick <i>fix</i> for the three angles you hold (circles and/or
virtual angles under <i>more constraints</i>); the others follow. Each solution
branch is a column. <i>min/max</i> drop branches outside a range; a circle with
both bounds may be shifted by 360. <i>two reflections</i> holds a second reflection
in Bragg condition too (it counts as one constraint).</p>

<p><b>3D view.</b> Click a solution column to drive the drawing. <b>config</b> shows
the angle values entered under <i>angles</i>, whether or not they solve anything,
and the line under the table says which (h,k,l) is in Bragg condition there
&mdash; a simulator for any position. <b>zero</b> sets every circle to 0. The view's
panel has the reciprocal vectors, the reciprocal cell, the Ewald sphere and the
patch of it the detector sees.</p>

<p><b>Tools.</b> <i>table</i> solves a list of reflections; <i>reachable</i> checks
which allowed reflections a setup can reach (optionally through a THC aperture);
<i>detector</i> shows the detector face in reciprocal space: coloured by h, k, l,
|Q|, 2&theta;, Brillouin zone, distance from the nearest lattice point, a diffuse
(TDS-like) estimate, the exit angle or Q along and across the surface, with
overlays for integer hkl lines, cell and Brillouin-zone boundaries, Bragg spots,
powder rings, the direct beam, horizon and Yoneda line and truncation-rod
crossings; hover or click for the values at a point; <i>crystal</i> lists the allowed reflections;
<i>grazing</i> gives critical angle, penetration depth and footprint.</p>

<p><b>Config.</b> <i>load setup</i> applies a Bernina setup (held angles and ranges,
the same as eco's <code>set_constraints</code>); <i>store config&hellip;</i> writes the
inputs to a JSON file, <i>load config</i> reads one back (also files of the
earlier diffcalc version).</p>
</div>
"""


def _num_or_none(text):
    try:
        return float(str(text).strip())
    except ValueError:
        return None


def _fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    return f"{float(v):.3f}"


def _err(text):
    return f'<span style="color:#b00;font-size:12px">{html.escape(str(text))}</span>'


def _note(text, color="#a60"):
    return f'<span style="color:{color};font-size:12px">{text}</span>'


def _show_df(out, df, floatfmt="{:.3f}", max_rows=200):
    with out:
        with pd.option_context("display.float_format", floatfmt.format, "display.max_rows", max_rows):
            display(df)


def load_presets(path=PRESETS_JSON) -> dict:
    data = json.loads(Path(path).read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


class Panel:
    """A titled card whose body can be collapsed."""

    def __init__(self, title, body, is_open=True):
        self.body = body
        self.toggle = widgets.Button(description="▾", tooltip="collapse / expand",
                                     layout=L(width="30px", height="26px", padding="0", margin="0 6px 0 0"))
        self.toggle.on_click(lambda _b: self.set_open(not self.is_open))
        head = widgets.HBox([self.toggle, widgets.HTML(f"<b>{title}</b>")], layout=L(align_items="center"))
        self.box = widgets.VBox([head, body], layout=L(border="1px solid #d0d7de", padding="6px 8px",
                                                       margin="0 8px 8px 0"))
        self.is_open = True
        self.set_open(is_open)

    def set_open(self, value):
        self.is_open = bool(value)
        self.body.layout.display = "flex" if self.is_open else "none"
        self.toggle.description = "▾" if self.is_open else "▸"


class ReflectionsApp:
    """The interactive reflections calculator; display ``.ui``."""

    LAYOUTS = {"large 3D": "large", "side by side": "side", "stacked": "stacked"}

    def __init__(self, presets_path=PRESETS_JSON):
        self._suspend = False
        self._pushes = 0
        self.sample = None
        self.structure = None
        self.df = None
        self.cols = []
        self.selected = SEL_CONFIG
        self.display_settings = None
        self._restore_done = False  # until the 3D page has offered the state it remembers
        self._forget = False
        self.results = {}           # last result of each tool, also for scripts and tests
        self.presets = load_presets(presets_path)
        self._build_inputs()
        self._build_config()
        self._build_solutions()
        self._build_view()
        self._build_tools()
        self._build_shell()
        self._wire()
        self.recompute()

    # ================================================================ inputs

    def _build_inputs(self):
        w = widgets
        s = DEFAULT_SAMPLE
        self.ctrl = {}
        for key, desc in (("uc_a", "a [Å]"), ("uc_b", "b [Å]"), ("uc_c", "c [Å]"),
                          ("uc_alpha", "alpha"), ("uc_beta", "beta"), ("uc_gamma", "gamma")):
            self.ctrl[key] = w.FloatText(value=s[key], description=desc, layout=L(width=UC_W),
                                         style={"description_width": "48px"})
        self.system = w.Dropdown(options=[("crystal system ...", "")] + [(k, k) for k in cry.SYSTEMS],
                                 layout=L(width="240px"))
        self.system_btn = w.Button(description="apply", tooltip="set the angles (and equal lengths) "
                                   "this crystal system requires, from a (and c)", layout=L(width="70px"))
        self.ctrl["space_group"] = w.Text(value=s["space_group"], description="space group",
                                          placeholder="e.g. R-3m, Fd-3m, 166", layout=L(width="240px"),
                                          style={"description_width": "84px"})
        self.cif = w.FileUpload(accept=".cif", multiple=False, description="CIF ...", layout=L(width="100px"))
        self.crystal_msg = w.HTML()
        self.ctrl["surface_normal"] = w.Text(value=s["surface_normal"], description="surface normal (h,k,l)",
                                             placeholder="empty: none", layout=L(width="360px"),
                                             style={"description_width": "160px"})
        self.ctrl["reference"] = w.Text(value=s["reference"], description="reference (h,k,l)",
                                        placeholder="for alpha, beta, psi", layout=L(width="360px"),
                                        style={"description_width": "160px"})
        self.ctrl["orientations"] = w.Textarea(
            value=s["orientations"], description="orientations (h,k,l) > lab",
            placeholder='one "(h,k,l) > (x,y,z)" per line; or > beam, > up, > x+',
            layout=L(width="360px", height="66px"), style={"description_width": "150px"})

        for n in "hkl":
            self.ctrl[n] = w.FloatText(value=1.0, step=0.1, description=n, layout=L(width=HKL_W),
                                       style={"description_width": "20px"})
        self.ctrl["energy"] = w.BoundedFloatText(value=9000, min=1000, max=30000, step=10,
                                                 description="energy [eV]", layout=L(width="190px"),
                                                 style={"description_width": "80px"})
        self.energy_slider = w.FloatSlider(value=9000, min=1000, max=30000, step=10, readout=False,
                                           continuous_update=False, layout=L(width="170px"))
        w.link((self.ctrl["energy"], "value"), (self.energy_slider, "value"))
        self.ctrl["two"] = w.Checkbox(value=False, description="keep a second reflection in Bragg condition",
                                      indent=False, layout=L(width="320px"))
        for n in ("h2", "k2", "l2"):
            self.ctrl[n] = w.FloatText(value=0.0 if n != "l2" else 1.0, step=1, description=n[0],
                                       layout=L(width=HKL_W), style={"description_width": "20px"})
        self.ctrl["detector"] = w.Dropdown(options=[(v, k) for k, v in dif.DETECTOR_POLICIES.items()],
                                           value="h1", layout=L(width="240px"))
        self.target_msg = w.HTML()

        # angles: value, min, max, fix
        self.filters = {}
        self.angle_rows = {}
        for name in dif.CIRCLES + dif.VIRTUAL_CONSTRAINTS:
            self.ctrl[name] = w.FloatText(value=DEFAULT_HELD.get(name, 0.0), layout=L(width=FIELD_W))
            self.ctrl[f"{name}_fixed"] = w.Checkbox(value=name in DEFAULT_HELD, description="fix", indent=False,
                                                    layout=L(width="48px"))
            lo = w.Text(placeholder="min", layout=L(width=FILTER_W))
            hi = w.Text(placeholder="max", layout=L(width=FILTER_W))
            self.filters[name] = (lo, hi)
            self.angle_rows[name] = w.HBox([w.Label(name, layout=L(width=LABEL_W)), self.ctrl[name], lo, hi,
                                            self.ctrl[f"{name}_fixed"]])
        for name in dif.BOOL_CONSTRAINTS:
            self.ctrl[f"{name}_fixed"] = w.Checkbox(value=False, description=f"hold {name}", indent=False,
                                                    layout=L(width="200px"))
        header = w.HTML(f'<div style="font-size:11px;color:#666;margin-top:4px">'
                        f'<span style="display:inline-block;width:{LABEL_W}"></span>'
                        f'<span style="display:inline-block;width:{FIELD_W}">value</span>'
                        f'<span style="display:inline-block;width:110px">solution min / max</span></div>')
        more = w.VBox([self.angle_rows[n] for n in dif.VIRTUAL_CONSTRAINTS]
                      + [self.ctrl[f"{n}_fixed"] for n in dif.BOOL_CONSTRAINTS])
        self.more_constraints = w.Accordion(children=[more], selected_index=None)
        self.more_constraints.set_title(0, "more constraints (virtual angles)")
        self.held_msg = w.HTML()

        c = self.ctrl
        self.tab_sample = w.VBox([
            w.HBox([c[k] for k in ("uc_a", "uc_b", "uc_c", "uc_alpha", "uc_beta", "uc_gamma")],
                   layout=L(flex_flow="row wrap")),
            w.HBox([self.system, self.system_btn]),
            w.HBox([c["space_group"], self.cif]), self.crystal_msg,
            c["surface_normal"], c["reference"], c["orientations"],
        ])
        self.tab_reflection = w.VBox([
            w.HBox([c["h"], c["k"], c["l"]]), w.HBox([c["energy"], self.energy_slider]), self.target_msg,
            w.HTML("<b>two reflections</b>"), c["two"], w.HBox([c["h2"], c["k2"], c["l2"]]), c["detector"],
        ])
        self.tab_angles = w.VBox([header] + [self.angle_rows[n] for n in dif.CIRCLES]
                                 + [self.more_constraints, self.held_msg])

    # ================================================================ config

    def _build_config(self):
        w = widgets
        self.preset = w.Dropdown(options=[("load setup ...", "")] + [(f"{k}: {v.get('description', '')}", k)
                                                                      for k, v in self.presets.items()],
                                 layout=L(width="360px"))
        self.preset_msg = w.HTML()
        self.include_groups = [
            ("sample", "sample (cell, space group, orientations, surface normal, reference)"),
            ("hkl", "h, k, l and the second reflection"), ("energy", "energy"),
            ("angles", "angle values"), ("fixed", "fix conditions"), ("filters", "solution min/max"),
            ("display", "3D view display (toggles, reciprocal scale, detector)"),
            ("layout", "panel layout (arrangement, open panels, 3D height)"),
        ]
        self.include = {k: w.Checkbox(value=True, description=t, indent=False, layout=L(width="360px"))
                        for k, t in self.include_groups}
        self.store_btn = w.Button(description="store config…", layout=L(width="130px"))
        self.make_btn = w.Button(description="create file", button_style="primary", layout=L(width="110px"))
        self.file_name = w.Text(value="bernina_diffraction_config", description="file name",
                                layout=L(width="330px"), style={"description_width": "70px"})
        self.upload = w.FileUpload(accept=".json", multiple=False, description="load config",
                                   layout=L(width="150px"))
        self.download_link = w.HTML()
        self.config_msg = w.HTML()
        self.store_panel = w.VBox(
            [w.HTML('<div style="font-size:11px;color:#666">include in the file (unchecked groups are '
                    'left out, so a file can be a partial template):</div>')]
            + [self.include[k] for k, _ in self.include_groups]
            + [self.file_name, self.make_btn, self.download_link],
            layout=L(display="none", border="1px solid #ddd", padding="6px", margin="4px 0 0 0"))
        self.tab_config = w.VBox([self.preset, self.preset_msg, w.HBox([self.store_btn, self.upload]),
                                  self.store_panel, self.config_msg])

    # ================================================================ solutions, view

    def _build_solutions(self):
        self.solution_buttons = widgets.HBox([])
        self.table_html = widgets.HTML()
        self.solve_msg = widgets.HTML()
        self.config_hkl = widgets.HTML()

    def _build_view(self):
        self.view_frame = widgets.HTML(
            f'<iframe srcdoc="{html.escape(VIEWER_HTML.read_text())}" '
            'style="width:100%; height:760px; border:1px solid #ddd;"></iframe>', layout=L(width="100%"))
        self.payload_holder = widgets.HTML()
        # the 3D page writes its detector choice here (see its sendDetector)
        self.det_from_view = widgets.Text(layout=L(display="none"))
        self.det_from_view.add_class("diff3d-detector-in")
        # ... and, once, the state it remembers from the last session in this browser
        self.state_from_view = widgets.Text(layout=L(display="none"))
        self.state_from_view.add_class("diff3d-state-in")

    # ================================================================ tools

    def _build_tools(self):
        w = widgets
        # reflection table
        self.t_hkls = w.Textarea(value="1,1,1\n2,2,0\n3,1,1\n0,0,4", description="h,k,l list",
                                 layout=L(width="220px", height="120px"))
        self.t_sol = w.IntText(value=0, description="solution #", layout=L(width="170px"))
        self.t_btn = w.Button(description="compute table", button_style="primary")
        self.t_out = w.Output()
        tool_table = w.VBox([w.HBox([self.t_hkls, self.t_sol]), self.t_btn, self.t_out])

        # reachable reflections
        self.r_source = w.Dropdown(options=[("allowed reflections of the crystal", "allowed"),
                                            ("the h,k,l list of the table tab", "list")],
                                   description="reflections", layout=L(width="380px"),
                                   style={"description_width": "80px"})
        self.r_dmin = w.FloatText(value=1.0, description="d min [Å]", layout=L(width="170px"),
                                  style={"description_width": "80px"})
        self.r_max = w.IntText(value=60, description="at most", layout=L(width="150px"),
                               style={"description_width": "60px"})
        from bernina_prep.thc_angles import APERTURES
        self.r_aperture = w.Dropdown(options=[("no aperture", "")] + [(f"THC {k}", k) for k in APERTURES],
                                     description="aperture", layout=L(width="300px"),
                                     style={"description_width": "80px"})
        self.r_beta = w.FloatSlider(value=0, min=0, max=360, step=5, description="flange β",
                                    layout=L(width="300px"), style={"description_width": "80px"})
        self.r_only = w.Checkbox(value=False, description="only reachable", indent=False)
        self.r_btn = w.Button(description="check", button_style="primary")
        self.r_out = w.Output()
        tool_reach = w.VBox([self.r_source, w.HBox([self.r_dmin, self.r_max]),
                             w.HBox([self.r_aperture]), self.r_beta, w.HBox([self.r_btn, self.r_only]),
                             w.HTML('<span style="font-size:11px;color:#666">uses the constraints, the '
                                    'min/max and the energy above; THC envelopes from the THC app</span>'),
                             self.r_out])

        # detector
        self.d_preset = w.Dropdown(options=list(dif.DETECTOR_PRESETS), description="detector",
                                   layout=L(width="380px"), style={"description_width": "70px"})
        self.d_dist = w.FloatText(value=500.0, description="distance [mm]", layout=L(width="190px"),
                                  style={"description_width": "96px"})
        self.d_w = w.FloatText(value=77.0, description="width [mm]", layout=L(width="170px"),
                               style={"description_width": "76px"})
        self.d_h = w.FloatText(value=77.0, description="height [mm]", layout=L(width="170px"),
                               style={"description_width": "76px"})
        self.d_tol = w.FloatText(value=0.5, description="band [%]", layout=L(width="150px"),
                                 style={"description_width": "60px"})
        self.d_qty = w.Dropdown(options=[(v[0], k) for k, v in dv.COLOURS.items()], value="l",
                                description="colour", layout=L(width="380px"), style={"description_width": "50px"})
        self.d_res = w.IntSlider(value=161, min=61, max=401, step=20, description="points across",
                                 layout=L(width="300px"), style={"description_width": "90px"})
        self.d_btn = w.Button(description="map the detector", button_style="primary")
        self.d_follow = w.Checkbox(value=False, description="follow the 3D view", indent=False,
                                   layout=L(width="auto"))
        self.d_over = {k: w.Checkbox(value=k in dv.DEFAULT_OVERLAYS, description=label, indent=False,
                                     layout=L(width="auto", margin="0 12px 0 0"))
                       for k, label in dv.OVERLAYS.items()}
        self.d_readout = w.HTML(layout=L(min_height="18px", width="720px"))
        self.d_figure = FigureWidget((7.0, 4.5))
        self.d_fig, self.d_canvas, self.d_plot = self.d_figure.fig, self.d_figure.canvas, self.d_figure.widget
        self.d_out = w.Output(layout=L(flex="1 1 380px"))
        self._det = None             # (pose, grid, spots, allowed, label) of the last map
        tool_det = w.VBox([self.d_preset, w.HBox([self.d_dist, self.d_tol]), w.HBox([self.d_w, self.d_h]),
                           w.HBox([self.d_res, self.d_btn, self.d_follow]),
                           w.HTML('<div style="font-size:11px;color:#666;white-space:normal;line-height:1.3">'
                                  'at the pose shown in the 3D view (a solution or config); band: how far '
                                  'from the Ewald sphere a lattice point may be (bandwidth, mosaicity). '
                                  'Hover or click the image for hkl, |Q|, zone and exit angle at that point; '
                                  'colour and overlays redraw at once, the map is recalculated by the '
                                  'button (or with each new pose when following the 3D view).</div>',
                                  layout=L(width="720px")),
                           self.d_qty, w.HBox(list(self.d_over.values()), layout=L(flex_flow="row wrap", width="720px")),
                           self.d_plot, self.d_readout, self.d_out], layout=L(width="100%"))

        # crystal
        self.c_dmin = w.FloatText(value=1.0, description="d min [Å]", layout=L(width="170px"),
                                  style={"description_width": "70px"})
        self.c_forbidden = w.Checkbox(value=False, description="show forbidden too", indent=False)
        self.c_btn = w.Button(description="list reflections", button_style="primary")
        self.c_out = w.Output()
        tool_crystal = w.VBox([
            w.HTML('<span style="font-size:11px;color:#666">one reflection per symmetry-equivalent set, '
                   'from the space group (and the CIF atoms for |F|); see the sample tab</span>'),
            w.HBox([self.c_dmin, self.c_forbidden]), self.c_btn, self.c_out])

        # grazing incidence
        self.g_formula = w.Text(value="Si", description="formula", layout=L(width="190px"),
                                style={"description_width": "60px"})
        self.g_density = w.FloatText(value=2.33, description="ρ [g/cm³]", layout=L(width="170px"),
                                     style={"description_width": "70px"})
        self.g_delta = w.Text(placeholder="from formula", description="δ", layout=L(width="170px"),
                              style={"description_width": "20px"})
        self.g_beta = w.Text(placeholder="from formula", description="β", layout=L(width="170px"),
                             style={"description_width": "20px"})
        self.g_beam = w.FloatText(value=20.0, description="beam [µm]", layout=L(width="170px"),
                                  style={"description_width": "70px"},
                                  tooltip="beam size in the plane of incidence")
        self.g_btn = w.Button(description="calculate", button_style="primary")
        self.g_out = w.Output()
        tool_grazing = w.VBox([w.HBox([self.g_formula, self.g_density]), w.HBox([self.g_delta, self.g_beta]),
                               self.g_beam, self.g_btn,
                               w.HTML('<span style="font-size:11px;color:#666">for the energy above and, '
                                      'if a solution is selected, its betain/betaout (needs a surface '
                                      'normal)</span>'),
                               self.g_out])

        self.tool_tabs = w.Tab(children=[tool_table, tool_reach, tool_det, tool_crystal, tool_grazing])
        for i, t in enumerate(("table", "reachable", "detector", "crystal", "grazing")):
            self.tool_tabs.set_title(i, t)

    # ================================================================ shell

    def _build_shell(self):
        w = widgets
        self.input_tabs = w.Tab(children=[self.tab_sample, self.tab_reflection, self.tab_angles, self.tab_config],
                                layout=L(width=INPUT_W))
        for i, t in enumerate(("sample", "reflection", "angles", "config")):
            self.input_tabs.set_title(i, t)
        self.input_tabs.selected_index = 1
        self.panels = {
            "inputs": Panel("inputs", self.input_tabs),
            "solutions": Panel("solutions", w.VBox([
                w.HTML('<span style="font-size:11px;color:#666">click a column to show it in the 3D view</span>'),
                self.solve_msg, self.solution_buttons, self.table_html, self.config_hkl])),
            "view": Panel("3D view", w.VBox([self.view_frame, self.payload_holder, self.det_from_view,
                                             self.state_from_view])),
            "tools": Panel("tools", self.tool_tabs, is_open=False),
        }
        self.layout_switch = w.ToggleButtons(options=list(self.LAYOUTS), value="large 3D",
                                             style={"button_width": "96px"})
        self.view_height = w.BoundedIntText(value=760, min=300, max=1600, step=20, description="3D height",
                                            layout=L(width="150px"), style={"description_width": "62px"})
        self.help_btn = w.ToggleButton(value=False, description="help", icon="question", layout=L(width="80px"))
        self.help_box = w.HTML(HELP_HTML, layout=L(display="none", border="1px solid #d0d7de",
                                                   padding="4px 12px", margin="0 0 8px 0"))
        header = w.HBox(
            [w.HTML('<span style="font-size:15px;font-weight:600">Diffraction reflections '
                    '<span style="font-weight:400;color:#666">&middot; Bernina, You geometry</span></span>',
                    layout=L(margin="0 16px 0 0")),
             self.layout_switch, self.view_height, self.help_btn],
            layout=L(align_items="center", flex_flow="row wrap", margin="0 0 8px 0"))
        self.restore_note = w.HTML()
        self.fresh_btn = w.Button(description="start fresh", layout=L(width="110px"),
                                  tooltip="forget the remembered state and reload with the defaults")
        self.restore_box = w.HBox([self.restore_note, self.fresh_btn],
                                  layout=L(display="none", align_items="center", margin="0 0 8px 0"))
        self.body = w.VBox(layout=L(width="100%"))
        self.ui = w.VBox([header, self.restore_box, self.help_box, self.body], layout=L(width="100%"))
        self.arrange()
        self.display_settings = {"controlsBelow": False}

    def arrange(self, *_):
        w = widgets
        mode = self.LAYOUTS[self.layout_switch.value]
        p = {k: v.box for k, v in self.panels.items()}
        p["view"].layout.flex = "1 1 640px" if mode == "large" else "1 1 520px"
        p["view"].layout.min_width = "0"
        for k in ("inputs", "solutions", "tools"):
            p[k].layout.flex = "0 0 auto"
        if mode == "side":
            row = w.HBox([p["inputs"], p["solutions"], p["view"]],
                         layout=L(flex_flow="row wrap", align_items="flex-start"))
            self.body.children = [w.VBox([row, p["tools"]])]
        elif mode == "large":
            left = w.VBox([p["inputs"], p["solutions"], p["tools"]], layout=L(flex="0 0 auto"))
            self.body.children = [w.HBox([left, p["view"]], layout=L(flex_flow="row wrap",
                                                                     align_items="flex-start"))]
        else:
            top = w.HBox([p["inputs"], p["solutions"], p["tools"]],
                         layout=L(flex_flow="row wrap", align_items="flex-start"))
            self.body.children = [w.VBox([top, p["view"]])]

    # ================================================================ wiring

    def _wire(self):
        for n in ("h", "k", "l"):
            self.ctrl[n].observe(self._round_hkl, names="value")
        for key, wdg in self.ctrl.items():
            wdg.observe(self.recompute, names="value")
        for lo, hi in self.filters.values():
            lo.observe(self.recompute, names="value")
            hi.observe(self.recompute, names="value")
        self.system_btn.on_click(self._apply_system)
        self.cif.observe(self._on_cif, names="value")
        self.preset.observe(lambda ch: self.load_preset(ch["new"]), names="value")
        self.store_btn.on_click(self._toggle_store)
        self.make_btn.on_click(self._on_make)
        self.file_name.observe(lambda _c: setattr(self.download_link, "value", ""), names="value")
        self.upload.observe(self._on_upload, names="value")
        self.layout_switch.observe(self.arrange, names="value")
        self.layout_switch.observe(self._view_controls_for_layout, names="value")
        self.view_height.observe(lambda _c: self.push_view(), names="value")
        self.state_from_view.observe(self._state_from_view, names="value")
        self.fresh_btn.on_click(self._start_fresh)
        self.help_btn.observe(lambda c: setattr(self.help_box.layout, "display", "flex" if c["new"] else "none"),
                              names="value")
        self.t_btn.on_click(self.run_table)
        self.r_btn.on_click(self.run_reachable)
        self.d_preset.observe(self._detector_preset, names="value")
        for wid in (self.d_w, self.d_h, self.d_dist):
            wid.observe(lambda _c: self.push_view(), names="value")
        self.det_from_view.observe(self._detector_from_view, names="value")
        self.d_btn.on_click(self.run_detector)
        self.d_qty.observe(lambda _c: self.draw_detector(), names="value")
        for cb in self.d_over.values():
            cb.observe(lambda _c: self.draw_detector(), names="value")
        if self.d_canvas is not None:
            self.d_canvas.mpl_connect("button_press_event", self._detector_click)
        self.c_btn.on_click(self.run_crystal)
        self.g_btn.on_click(self.run_grazing)

    def _round_hkl(self, change):
        rounded = round(float(change["new"]), HKL_DECIMALS)
        if rounded != change["new"]:
            self._suspend = True
            try:
                change["owner"].value = rounded
            finally:
                self._suspend = False

    def _view_controls_for_layout(self, change):
        self.display_settings = {"controlsBelow": self.LAYOUTS[change["new"]] != "large"}
        self.push_view()

    # ================================================================ reading the inputs

    def cell(self) -> dict:
        return {k[3:]: float(self.ctrl[k].value) for k in ("uc_a", "uc_b", "uc_c", "uc_alpha", "uc_beta", "uc_gamma")}

    def constraints(self) -> dict:
        cons = {}
        for name in dif.CIRCLES + dif.VIRTUAL_CONSTRAINTS:
            if self.ctrl[f"{name}_fixed"].value:
                cons[name] = float(self.ctrl[name].value)
        for name in dif.BOOL_CONSTRAINTS:
            if self.ctrl[f"{name}_fixed"].value:
                cons[name] = True
        if self.ctrl["two"].value:
            cons["h2"] = [float(self.ctrl[n].value) for n in ("h2", "k2", "l2")]
        return cons

    def ranges(self) -> dict:
        out = {}
        for name, (lo, hi) in self.filters.items():
            a, b = _num_or_none(lo.value), _num_or_none(hi.value)
            if a is not None or b is not None:
                out[name] = (a, b)
        return out

    def hkl(self):
        return [float(self.ctrl[n].value) for n in "hkl"]

    def energy(self) -> float:
        return float(self.ctrl["energy"].value)

    def config_angles(self) -> dict:
        return {n: float(self.ctrl[n].value) for n in dif.CIRCLES}

    def space_group(self):
        text = self.ctrl["space_group"].value.strip()
        if not text:
            return None
        return cry.SpaceGroup(text)

    # ================================================================ solving

    def recompute(self, *_):
        if self._suspend:
            return
        self._check_crystal()
        try:
            self.sample = dif.Sample.from_text(self.cell(), self.ctrl["orientations"].value,
                                               self.ctrl["surface_normal"].value, self.ctrl["reference"].value)
        except Exception as exc:
            self.sample = None
            self._no_solutions(_err(f"sample: {exc}"))
            return
        cons = self.constraints()
        held = len(cons)
        self.held_msg.value = ("" if held == 3 else
                               _note(f"{held} constraint(s) held; three are needed for a finite set "
                                     "of solutions (the second reflection counts as one)"))
        try:
            df = self.sample.solve(self.hkl(), self.energy(), cons, self.ctrl["detector"].value)
        except Exception as exc:
            self._no_solutions(_err(exc))
            return
        if self.ctrl["two"].value and len(df.columns):
            h2 = cons["h2"]
            extra = {}
            for c in df.columns:
                sp = self.sample.spots(dif.solution_angles(df, c), [self.hkl(), h2], self.energy())
                extra[c] = {"separation": sp["separation"], "tdet max [mm]": sp["tdet_max"]}
            df = pd.concat([df, pd.DataFrame(extra)])
        self.df = df
        self.cols = [c for c in df.columns if dif.passes(df[c], self.ranges())]
        hidden = len(df.columns) - len(self.cols)
        if df.empty:
            beyond = np.linalg.norm(self.sample.B @ np.asarray(self.hkl())) > 2 * dif.k_magnitude(self.energy())
            self.solve_msg.value = _note("no solution: 2 theta would exceed 180, raise the energy" if beyond
                                         else "no solution under these constraints")
        elif not self.cols:
            self.solve_msg.value = _note(f"no solution passes the min/max ({hidden} hidden)")
        else:
            self.solve_msg.value = _note(f"{hidden} more hidden by the min/max", "#666") if hidden else ""
        if isinstance(self.selected, int) and self.selected >= len(self.cols):
            self.selected = SEL_CONFIG
        self._rebuild_buttons()
        self.select(self.selected)

    def _no_solutions(self, message):
        self.solve_msg.value = message
        self.df, self.cols = None, []
        if isinstance(self.selected, int):
            self.selected = SEL_CONFIG
        self._rebuild_buttons()
        self.select(self.selected)

    def _check_crystal(self):
        msgs, target = [], []
        try:
            sg = self.space_group()
        except Exception as exc:
            sg = None
            msgs.append(_err(exc))
        try:
            msgs += [_note(html.escape(m)) for m in cry.check_cell(self.cell(), sg)]
            if sg is not None:
                msgs.append(_note(f"{html.escape(sg.name)} ({sg.system}): "
                                  f"{html.escape(sg.condition_summary())}", "#666"))
            target = cry.target_check(self.hkl(), self.cell(), sg, self.structure)
        except Exception as exc:
            msgs.append(_err(exc))
        if self.structure is not None:
            st = self.structure
            msgs.append(_note(f"CIF {html.escape(st.name)}: {html.escape(st.formula)}, "
                              f"{st.density:.3f} g/cm³", "#282"))
        self.crystal_msg.value = "<br>".join(msgs)
        self.target_msg.value = "<br>".join(_note(html.escape(t), "#b00") for t in target)

    # ================================================================ table and view

    def _rebuild_buttons(self):
        w = widgets
        buttons = []
        for i, col in enumerate(self.cols):
            b = w.Button(description=str(col), tooltip=f"show {col} in the 3D view",
                         layout=L(width=f"{COL_W}px", margin="0"))
            b.on_click(lambda _b, idx=i: self.select(idx))
            b._sel = i
            buttons.append(b)
        config = w.Button(description="config", tooltip="show the angle values entered under 'angles'",
                          layout=L(width=f"{ROW_LABEL_W}px", margin="0"))
        config.on_click(lambda _b: self.select(SEL_CONFIG))
        config._sel = SEL_CONFIG
        zero = w.Button(description="zero", tooltip="all circles at 0", layout=L(width="60px", margin="0 0 0 8px"))
        zero.on_click(lambda _b: self.select(SEL_ZERO))
        zero._sel = SEL_ZERO
        self.solution_buttons.children = [config] + buttons + [zero]

    def _render_table(self):
        if self.df is None or not self.cols:
            self.table_html.value = ""
            return
        df, sel = self.df, self.selected
        bg = lambda i: f"background:{SELECTED_BG};" if i == sel else ""
        head = "".join(f'<th style="width:{COL_W}px;text-align:right;padding:2px 6px;border-bottom:1px solid #ccc;'
                       f'{bg(i)}">{html.escape(str(c))}</th>' for i, c in enumerate(self.cols))
        body = []
        for name in df.index:
            cells = "".join(f'<td style="width:{COL_W}px;text-align:right;padding:1px 6px;'
                            f'font-variant-numeric:tabular-nums;{bg(i)}">{_fmt(df.loc[name, c])}</td>'
                            for i, c in enumerate(self.cols))
            body.append(f'<tr><td style="width:{ROW_LABEL_W}px;text-align:right;padding:1px 6px;font-weight:600">'
                        f'{html.escape(str(name))}</td>{cells}</tr>')
        self.table_html.value = ('<table style="border-collapse:collapse;font-size:12px">'
                                 f'<tr><td style="width:{ROW_LABEL_W}px"></td>{head}</tr>' + "".join(body) + "</table>")

    def shown_angles(self):
        """(angles, label) of what the 3D view shows."""
        if self.selected == SEL_CONFIG:
            return self.config_angles(), "config"
        if isinstance(self.selected, int) and self.df is not None and self.selected < len(self.cols):
            col = self.cols[self.selected]
            return dif.solution_angles(self.df, col), str(col)
        return {}, "zero"

    def select(self, sel):
        self.selected = sel
        for b in self.solution_buttons.children:
            b.button_style = "info" if getattr(b, "_sel", None) == sel else ""
        self._render_table()
        self._render_config_hkl()
        self.push_view()

    def _render_config_hkl(self):
        if self.sample is None:
            self.config_hkl.value = ""
            return
        angles = self.config_angles()
        h = np.round(self.sample.hkl_at(angles, self.energy()), 6) + 0.0
        near = np.round(h)
        self.config_hkl.value = (
            '<div style="font-size:12px;margin-top:6px"><b>config</b> (the angles entered) puts '
            f'({", ".join(f"{x:.3f}" for x in h)}) in Bragg condition; nearest '
            f'({" ".join(str(int(x)) for x in near + 0.0)}), off by {np.linalg.norm(h - near):.3f}</div>')

    def push_view(self):
        angles, label = self.shown_angles()
        targets = [self.hkl()] + ([[float(self.ctrl[n].value) for n in ("h2", "k2", "l2")]]
                                  if self.ctrl["two"].value else [])
        msg = dif.view_payload(self.sample, angles, self.energy(), label, targets, n=self._pushes)
        try:
            msg["centring"] = dv.lattice_centring(self.space_group())   # for the Brillouin zones
        except Exception:
            msg["centring"] = "P"
        msg["frameHeight"] = int(self.view_height.value) if hasattr(self, "view_height") else 760
        if hasattr(self, "d_w"):
            msg["detectors"] = [[k, v[0], v[1]] for k, v in dif.DETECTOR_PRESETS.items()]
            msg["detector"] = {"w": float(self.d_w.value), "h": float(self.d_h.value)}
        if self.display_settings is not None:
            msg["display"] = self.display_settings          # sent once
            self.display_settings = None
        if self._forget:
            msg["forgetState"] = True
        elif self._restore_done:
            msg["autosave"] = self.autosave_state()
        self._pushes += 1
        self.payload_holder.value = ('<span id="diff3d-payload" style="display:none">'
                                     + html.escape(json.dumps(msg)) + "</span>")
        if getattr(self, "d_follow", None) is not None and self.d_follow.value and self._det is not None:
            self.run_detector()

    # ================================================================ crystal inputs

    def _apply_system(self, _b=None):
        system = self.system.value
        if not system:
            return
        c = self.cell()
        cell = cry.cell_for_system(system, c["a"], c["b"], c["c"], c["alpha"], c["beta"], c["gamma"])
        self._set_cell(cell)
        self.crystal_msg.value = _note(html.escape(f"{system}: {cry.SYSTEMS[system][2]}"), "#282")

    def _set_cell(self, cell):
        self._suspend = True
        try:
            for k, v in cell.items():
                self.ctrl[f"uc_{k}"].value = round(float(v), 6)
        finally:
            self._suspend = False
        self.recompute()

    def _on_cif(self, change):
        items = change["new"]
        if not items:
            return
        item = items[0] if isinstance(items, (list, tuple)) else list(items.values())[0]
        try:
            st = cry.read_cif(cry.to_cif_text(item["content"]))
        except Exception as exc:
            self.crystal_msg.value = _err(f"CIF: {exc}")
            return
        self.structure = st
        self._suspend = True
        try:
            self.ctrl["space_group"].value = st.space_group.name
            self.g_formula.value = st.formula
            self.g_density.value = round(st.density, 4)
        finally:
            self._suspend = False
        self._set_cell(st.cell)

    # ================================================================ presets and config files

    def load_preset(self, name):
        if not name:
            return
        p = self.presets[name]
        cons, ranges = p.get("constraints", {}), p.get("ranges") or {}
        self._suspend = True
        try:
            for n in dif.CIRCLES + dif.VIRTUAL_CONSTRAINTS:
                self.ctrl[f"{n}_fixed"].value = n in cons
                if n in cons:
                    self.ctrl[n].value = float(cons[n])
            for n in dif.BOOL_CONSTRAINTS:
                self.ctrl[f"{n}_fixed"].value = bool(cons.get(n))
            for n, (lo, hi) in self.filters.items():
                lo.value, hi.value = (f"{v:g}" for v in ranges[n]) if n in ranges else ("", "")
            self.ctrl["two"].value = False
            self.preset.value = ""
        finally:
            self._suspend = False
        self.preset_msg.value = _note(f"setup <b>{html.escape(name)}</b>: {html.escape(p.get('description', ''))}; "
                                      "sample and hkl kept", "#282")
        self.recompute()

    def _toggle_store(self, _b=None):
        self.store_panel.layout.display = "none" if self.store_panel.layout.display != "none" else "flex"

    def current_config(self, everything: bool = False) -> dict:
        inc = {k: everything or b.value for k, b in self.include.items()}
        cfg = {"version": 2}
        if inc["sample"]:
            cfg["sample"] = {k: self.ctrl[k].value for k in DEFAULT_SAMPLE}
        if inc["hkl"]:
            cfg["hkl"] = {n: self.ctrl[n].value for n in ("h", "k", "l", "two", "h2", "k2", "l2", "detector")}
        if inc["energy"]:
            cfg["energy"] = self.ctrl["energy"].value
        if inc["angles"]:
            cfg["angles"] = {n: self.ctrl[n].value for n in dif.CIRCLES + dif.VIRTUAL_CONSTRAINTS}
        if inc["fixed"]:
            cfg["fixed"] = {n: self.ctrl[f"{n}_fixed"].value for n in dif.CONSTRAINTS}
        if inc["filters"]:
            cfg["filters"] = {n: [lo.value, hi.value] for n, (lo, hi) in self.filters.items()}
        if inc["layout"]:
            cfg["layout"] = {"arrangement": self.layout_switch.value,
                             "open": {k: v.is_open for k, v in self.panels.items()},
                             "input_tab": self.input_tabs.selected_index, "view_height": self.view_height.value}
        return cfg

    # ---------------------------------------------------------- remembered in the browser
    # The 3D page keeps the last autosave_state() in the browser's storage and
    # offers it back once when a new session starts (state_from_view), so a
    # Binder session that timed out continues where it was. Nothing is saved
    # before that offer, or a fresh session's defaults would overwrite it.

    def autosave_state(self) -> dict:
        state = self.current_config(everything=True)
        state["tools"] = {"detector": self.d_preset.value, "width": self.d_w.value, "height": self.d_h.value,
                          "distance": self.d_dist.value}
        state["selected"] = self.selected
        return state

    def _state_from_view(self, change):
        if self._restore_done:
            return
        self._restore_done = True
        try:
            offered = json.loads(change["new"] or "{}")
            if offered.get("state"):
                self.restore_state(offered["state"], offered.get("display"), offered.get("savedText", ""))
        except Exception as exc:
            self.restore_note.value = _err(f"could not restore the remembered state: {exc}")
            self.restore_box.layout.display = "flex"
        self.push_view()

    def restore_state(self, state: dict, display: dict | None = None, saved_text: str = ""):
        """Apply a state from autosave_state() (and the 3D page's display settings)."""
        cfg = dict(state)
        if display:
            cfg["display"] = display
        self._suspend = True
        try:
            self.apply_config(cfg)
            tools = state.get("tools") or {}
            if tools.get("detector") in dif.DETECTOR_PRESETS:
                self.d_preset.value = tools["detector"]
            for key, wid in (("width", self.d_w), ("height", self.d_h), ("distance", self.d_dist)):
                if isinstance(tools.get(key), (int, float)):
                    wid.value = tools[key]
        finally:
            self._suspend = False
        self.recompute()
        sel = state.get("selected")
        if sel in (SEL_CONFIG, SEL_ZERO) or (isinstance(sel, int) and 0 <= sel < len(self.cols)):
            self.select(sel)
        when = f" (saved {html.escape(saved_text)})" if saved_text else ""
        self.restore_note.value = _note(f"continued from your last state in this browser{when} "
                                        "(a CIF's atoms are not remembered)", "#282")
        self.restore_box.layout.display = "flex"

    def _start_fresh(self, _b=None):
        self._forget = True
        self.restore_note.value = _note("forgetting the remembered state, the page reloads ...", "#a60")
        self.push_view()

    def _on_make(self, _b=None):
        cfg = self.current_config()
        with_display = self.include["display"].value
        blob = base64.b64encode(json.dumps(cfg, indent=2).encode()).decode()
        # the 3D page rewrites this link's href to add its own display block
        hook = f'id="diff3d-config-download" data-cfg="{blob}" ' if with_display else ""
        stem = self.file_name.value.strip()
        stem = stem[:-5] if stem.lower().endswith(".json") else stem
        stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem).strip("._")
        name = (stem or "bernina_diffraction_config") + ".json"
        self.download_link.value = (f'<a download="{name}" {hook}href="data:application/json;base64,{blob}" '
                                    f'style="font-size:12px">&#11015; download {name}</a>')

    def apply_config(self, cfg: dict) -> list:
        """Apply a stored config (this version's or the diffcalc version's)."""
        applied, notes = [], []
        legacy = lambda n: dif.LEGACY_NAMES.get(n, n)
        for name, value in (cfg.get("sample") or {}).items():
            if name in self.ctrl:
                self.ctrl[name].value = value
                applied.append(name)
        if "sample" in cfg and "reference" not in cfg["sample"] and cfg.get("version", 1) < 2:
            # the diffcalc version used the surface normal as the reference too
            self.ctrl["reference"].value = cfg["sample"].get("surface_normal", self.ctrl["reference"].value)
            notes.append("reference = surface normal, as the earlier version used it")
        for name, value in (cfg.get("hkl") or {}).items():
            if name in self.ctrl:
                self.ctrl[name].value = value
                applied.append(name)
        if "energy" in cfg:
            self.ctrl["energy"].value = cfg["energy"]
            applied.append("energy")
        for name, value in (cfg.get("angles") or {}).items():
            name = legacy(name)
            if name in self.ctrl and name not in dif.BOOL_CONSTRAINTS and not isinstance(value, bool):
                self.ctrl[name].value = float(value)
                applied.append(name)
        for name, value in (cfg.get("fixed") or {}).items():
            name = legacy(name)
            if name in dif.UNSUPPORTED:
                if value:
                    notes.append(f"{name} is not supported any more and was dropped")
                continue
            if f"{name}_fixed" in self.ctrl:
                self.ctrl[f"{name}_fixed"].value = bool(value)
                applied.append(f"{name}_fixed")
        for name, pair in (cfg.get("filters") or {}).items():
            name = legacy(name)
            if name in self.filters:
                self.filters[name][0].value = pair[0] or ""
                self.filters[name][1].value = pair[1] or ""
                applied.append(f"{name}_filter")
        lay = cfg.get("layout")
        if isinstance(lay, dict):
            if lay.get("arrangement") in self.LAYOUTS:
                self.layout_switch.value = lay["arrangement"]
            for k, is_open in (lay.get("open") or {}).items():
                if k in self.panels:
                    self.panels[k].set_open(is_open)
            if isinstance(lay.get("input_tab"), int):
                self.input_tabs.selected_index = lay["input_tab"]
            if isinstance(lay.get("view_height"), int):
                self.view_height.value = lay["view_height"]
            applied.append("layout")
        if isinstance(cfg.get("display"), dict):
            self.display_settings = cfg["display"]
            applied.append("display")
        self._config_notes = notes
        return applied

    def _on_upload(self, change):
        items = change["new"]
        if not items:
            return
        item = items[0] if isinstance(items, (list, tuple)) else list(items.values())[0]
        try:
            cfg = json.loads(bytes(item["content"]).decode())
            self._suspend = True
            try:
                applied = self.apply_config(cfg)
            finally:
                self._suspend = False
            self.recompute()
            extra = "; ".join(self._config_notes)
            self.config_msg.value = _note(f"loaded {len(applied)} settings; groups absent from the file "
                                          f"were left as they are{'; ' + html.escape(extra) if extra else ''}",
                                          "#282")
        except Exception as exc:
            self.config_msg.value = _err(exc)

    # ================================================================ tools

    def _need_sample(self, out):
        if self.sample is None:
            with out:
                print("No valid sample yet (sample tab).")
            return False
        return True

    def run_table(self, _b=None):
        self.t_out.clear_output()
        if not self._need_sample(self.t_out):
            return
        try:
            hkls = [dif.parse_vector(ln) for ln in self.t_hkls.value.splitlines() if ln.strip()]
        except Exception as exc:
            with self.t_out:
                print(exc)
            return
        df = dif.reflection_table(self.sample, self.constraints(), hkls, self.energy(), self.ranges(),
                                  self.t_sol.value, self.ctrl["detector"].value)
        self.results["table"] = df
        _show_df(self.t_out, df)

    def run_reachable(self, _b=None):
        self.r_out.clear_output()
        if not self._need_sample(self.r_out):
            return
        try:
            if self.r_source.value == "list":
                hkls = [dif.parse_vector(ln) for ln in self.t_hkls.value.splitlines() if ln.strip()]
            else:
                sg = self.space_group()
                lst = cry.reflection_list(self.cell(), self.energy(), sg, self.structure, d_min=self.r_dmin.value)
                if "allowed" in lst:
                    lst = lst[lst["allowed"] & (lst["why not"] == "")]
                hkls = [tuple(r) for r in lst[["h", "k", "l"]].head(self.r_max.value).itertuples(index=False)]
                if sg is None:
                    with self.r_out:
                        print("No space group given: every reflection counts as allowed.")
            aperture = None
            if self.r_aperture.value:
                from bernina_prep.thc_angles import aperture_check
                aperture = aperture_check(self.r_aperture.value, self.r_beta.value)
            df = dif.reachable_reflections(self.sample, self.energy(), self.constraints(), hkls, self.ranges(),
                                           aperture, self.ctrl["detector"].value)
        except Exception as exc:
            self.results["reachable"] = exc
            with self.r_out:
                print(f"error: {exc}")
            return
        self.results["reachable"] = df
        if self.r_only.value and len(df):
            df = df[df["reachable"]]
        with self.r_out:
            n_ok = int(df["reachable"].sum()) if len(df) else 0
            print(f"{n_ok} of {len(df)} reachable")
        _show_df(self.r_out, df)

    def _detector_preset(self, change):
        self.d_w.value, self.d_h.value = dif.DETECTOR_PRESETS[change["new"]]

    def _detector_modules(self):
        """The module rectangles of the chosen preset, while width and height are still the preset's."""
        name = self.d_preset.value
        if not np.allclose((self.d_w.value, self.d_h.value), dif.DETECTOR_PRESETS.get(name, (np.nan, np.nan)),
                           atol=1e-3):
            return None
        return dif.DETECTOR_MODULES.get(name)

    def _detector_from_view(self, change):
        """The detector chosen in the 3D view: same preset and size here."""
        try:
            d = json.loads(change["new"])
            width, height = float(d["w"]), float(d["h"])
        except (ValueError, KeyError, TypeError):
            return
        if d.get("name") in dif.DETECTOR_PRESETS:
            self.d_preset.value = d["name"]
        self.d_w.value, self.d_h.value = width, height

    def run_detector(self, _b=None):
        """Map the detector face at the pose of the 3D view, then draw it."""
        self.d_out.clear_output()
        if not self._need_sample(self.d_out):
            return
        angles, label = self.shown_angles()
        try:
            try:
                sg = self.space_group()
            except Exception:
                sg = None
            pose = dv.Pose(self.sample, angles, self.energy(), self.d_dist.value, self.d_w.value,
                           self.d_h.value, centring=dv.lattice_centring(sg), alpha_c=self._critical_angle(),
                           modules=self._detector_modules())
            grid = dv.face_grid(pose, self.d_res.value)
            allowed = cry.allowed_function(sg, self.structure)
            spots = dif.reflections_on_detector(self.sample, angles, self.energy(), self.d_dist.value,
                                                self.d_w.value, self.d_h.value, self.d_tol.value / 100, allowed)
            if pose.modules and len(spots):
                spots["on module"] = dif.on_modules(pose.modules, spots["u [mm]"], spots["v [mm]"])
        except Exception as exc:
            self.results["detector"] = exc
            with self.d_out:
                print(f"error: {exc}")
            return
        self.results["detector"] = (grid, spots)
        self._det = (pose, grid, spots, allowed, label)
        self.draw_detector()
        with self.d_out:
            print(f"{label}; distance {self.d_dist.value:g} mm; zones from the {pose.centring} lattice"
                  + ("" if sg else " (no space group given)"))
        _show_df(self.d_out, dif.detector_summary(grid))
        with self.d_out:
            print(f"{len(spots)} reflection(s) within ±{self.d_tol.value:g} % of the Ewald sphere on the detector"
                  + ("" if allowed else " (no space group: forbidden ones included)"))
        if len(spots):
            _show_df(self.d_out, spots.head(50), "{:.4g}")

    def draw_detector(self):
        """Redraw the last map with the current colour and overlays (no recalculation)."""
        if self._det is None:
            return
        pose, grid, spots, allowed, label = self._det
        side = max(pose.width, pose.height)
        self.d_fig.set_size_inches(7.0, max(2.5, 6.0 * pose.height / side) + 0.6)
        overlays = [k for k, cb in self.d_over.items() if cb.value]
        notes = dv.draw(self.d_fig, pose, grid, self.d_qty.value, overlays, spots, allowed,
                        title=f"{label}, {pose.energy:g} eV, {pose.distance:g} mm")
        self.results["detector notes"] = notes
        self.d_readout.value = "".join(f'<div style="font-size:11px;color:#a60;white-space:normal">{html.escape(n)}</div>'
                                       for n in notes)
        self.d_figure.redraw()

    def _detector_click(self, event):
        if self._det is None or event.inaxes is None or event.xdata is None:
            return
        self.d_readout.value = (f'<code style="font-size:11px;white-space:normal">'
                                f'{html.escape(dv.point_text(self._det[0], event.xdata, event.ydata))}</code>')

    def _critical_angle(self):
        """Critical angle [deg] from the grazing tab (delta, or formula and density), or None."""
        try:
            d = _num_or_none(self.g_delta.value)
            if d is None:
                d, _ = gra.optical_constants(self.g_formula.value.strip(), self.g_density.value, self.energy())
            return gra.critical_angle(d)
        except Exception:
            return None

    def run_crystal(self, _b=None):
        self.c_out.clear_output()
        try:
            sg = self.space_group()
            df = cry.reflection_list(self.cell(), self.energy(), sg, self.structure, d_min=self.c_dmin.value)
        except Exception as exc:
            self.results["crystal"] = exc
            with self.c_out:
                print(f"error: {exc}")
            return
        self.results["crystal"] = df
        with self.c_out:
            if sg is None:
                print("No space group: every reflection is listed as allowed.")
            else:
                print(f"{sg.name} ({sg.system}); {sg.condition_summary()}")
        if "allowed" in df and not self.c_forbidden.value:
            df = df[df["allowed"]]
        _show_df(self.c_out, df, "{:.4g}")

    def run_grazing(self, _b=None):
        self.g_out.clear_output()
        import matplotlib.pyplot as plt

        e = self.energy()
        try:
            d, b = _num_or_none(self.g_delta.value), _num_or_none(self.g_beta.value)
            if d is None or b is None:
                d, b = gra.optical_constants(self.g_formula.value.strip(), self.g_density.value, e)
        except Exception as exc:
            self.results["grazing"] = exc
            with self.g_out:
                print(f"error: {exc}")
            return
        ac = gra.critical_angle(d)
        self.results["grazing"] = {"delta": d, "beta": b, "critical angle": ac}
        with self.g_out:
            print(f"{self.g_formula.value} at {e:g} eV: δ = {d:.3e}, β = {b:.3e}, critical angle "
                  f"{ac:.3f}° ({ac * np.pi / 180 * 1e3:.2f} mrad), attenuation length "
                  f"{gra.penetration_depth(90, d, b, e) / 1e4:.3g} µm at normal incidence")
        angles = np.linspace(0.01, max(5 * ac, 1.0), 400)
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.semilogy(angles, gra.penetration_depth(angles, d, b, e) / 10)
        ax.axvline(ac, color="r", lw=0.8, ls="--", label=f"α_c = {ac:.3f}°")
        ax.set_xlabel("incidence angle to the surface [deg]")
        ax.set_ylabel("1/e depth [nm]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        with self.g_out:
            display(fig)
        plt.close(fig)
        _show_df(self.g_out, gra.depth_table(d, b, e), "{:.4g}")
        if isinstance(self.selected, int) and self.df is not None and self.sample is not None \
                and self.sample.surface is not None and "betain" in self.df.index:
            col = self.cols[self.selected]
            bi, bo = float(self.df.loc["betain", col]), float(self.df.loc["betaout", col])
            with self.g_out:
                if bi <= 0 or bo <= 0:
                    print(f"{col}: betain {bi:.3f}°, betaout {bo:.3f}°: the beam enters or leaves "
                          "through the back of the sample")
                else:
                    z = gra.probed_depth(bi, bo, d, b, e)
                    print(f"{col}: betain {bi:.3f}°, betaout {bo:.3f}° -> probed depth {z / 10:.3g} nm; "
                          f"footprint {gra.footprint(self.g_beam.value, bi) / 1e3:.3g} mm "
                          f"for a {self.g_beam.value:g} µm beam")
