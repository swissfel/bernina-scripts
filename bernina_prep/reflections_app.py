"""The diffraction reflections app (``apps/diffraction_reflections.ipynb``).

``ReflectionsApp().ui`` is the whole interface: sample and constraint
inputs, the solutions table, the 3D view and the tools (reflection table,
reachable reflections, detector map, crystal and grazing-incidence checks).
The calculations are escape's ``escape.bernina`` (``diffraction``,
``detector_view``) and hklforge's (``crystal`` here wraps
``hklforge.crystal``; Brillouin-zone points, the solution scan), plus
``grazing`` and ``thc_angles``; the 3D view is hklforge's
(``hklforge.view``) with the Bernina geometry; this module only builds and
wires widgets.

Rules that keep it rendering under voila: px sizes only, core
widgets only, explicit flex basis for an Output beside something, no LaTeX
in descriptions, no blocking in callbacks.
"""
from __future__ import annotations

import base64
import contextlib
import html
import io
import json
import os
import time
import zlib
from datetime import datetime
from urllib.parse import parse_qs, quote, urlencode, urlsplit, urlunsplit
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pandas as pd
from hklforge import Diffractometer
from hklforge.crystal import bz as hbz
from hklforge.view import iframe as view_iframe
from hklforge.view import message as view_message
from hklforge.view import page as view_page
from hklforge.view import payload_html

try:
    from hklforge.view import structure_factor_nodes
except ImportError:                    # hklforge without the structure-factor spheres
    structure_factor_nodes = None
try:
    from hklforge.view import structure_atoms
except ImportError:                    # hklforge without the real-space structure
    structure_atoms = None
try:
    from hklforge.crystal import examples as cif_examples
except ImportError:                    # hklforge without example CIFs
    cif_examples = None
from hklforge.virtual import virtual_angles

from escape.bernina import detector_view as dv
from escape.bernina import diffraction as dif

from bernina_prep import crystal as cry
from bernina_prep import grazing as gra
from bernina_prep.mpl_widget import FigureWidget

HERE = Path(__file__).resolve().parent
PRESETS_JSON = HERE / "presets.json"

# The 3D view: hklforge's page with the Bernina geometry (SwissFEL frame,
# gamma for nu). The storage prefix keeps what browsers remembered with the
# earlier page (bernina-reflections-state, -snapshots, -remember).
VIEW_ID = "diff3d"
VIEW_STORAGE = "bernina-reflections"
VIEW_TITLE = "Bernina diffractometer (You geometry, SwissFEL frame)"


def bernina_view_page() -> str:
    """The 3D page (HTML text) with Bernina's geometry and detectors built in."""
    return view_page(dif.GEOMETRY, view_id=VIEW_ID, storage=VIEW_STORAGE, title=VIEW_TITLE,
                     detectors=[[k, v[0], v[1]] for k, v in dif.DETECTOR_PRESETS.items()])

L = widgets.Layout
UC_W, FIELD_W, FILTER_W, HKL_W, LABEL_W = "150px", "86px", "52px", "98px", "70px"
ROW_LABEL_W, COL_W = 70, 84
SELECTED_BG = "#cfe8fb"
HKL_DECIMALS = 4
INPUT_W = "420px"
ENERGY_LIMITS = (1000.0, 30000.0)               # eV, the energy field's range
ENERGY_HINT = "; or let the energy vary: mode['energy'] = (lo, hi) in eV"   # hklforge's ModeError
CONFIG_VERSION = 3        # 2: separate reference; 3: CIF structure, free energy
SEL_CONFIG, SEL_ZERO = "config", "zero"          # keys kept for stored states
POSITION_NAMES = {SEL_CONFIG: "constraint configuration", SEL_ZERO: "zero position"}

DEFAULT_SAMPLE = {
    "uc_a": 5.431, "uc_b": 5.431, "uc_c": 5.431, "uc_alpha": 90.0, "uc_beta": 90.0, "uc_gamma": 90.0,
    "surface_normal": "(1,1,1)", "reference": "(1,0,0)", "space_group": "Fd-3m",
    "orientations": "(1,1,1) > (1,0,0)\n(1,1,-1) > (0,1,0)",
}
DEFAULT_HELD = {"eta": 0.0, "chi": 0.0, "phi": 0.0}

TABLE_GROUPS = {
    "": "no grouping",
    "sym": "symmetry-equivalent (same family)",
    "phi": "same |Q| and angle to the phi axis",
    "surface": "same |Q| and angle to the surface normal",
    "dir": "same |Q| and angle to a direction",
}

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
lets the app warn when the target reflection is forbidden (and by which centring,
screw axis or glide plane) and list the allowed ones; with a CIF also |F| and
|F|<sup>2</sup>, including f&prime; and f&Prime; at the chosen energy. The angles table
(tools) can list all allowed reflections and group them: symmetry-equivalent ones, or
those a rotation about the phi axis, the surface normal or a direction maps onto each
other, with the rotation (for phi: the change of phi) between them.</p>

<p><b>Solving.</b> Tick <i>fix</i> for the three angles you hold (circles and/or
virtual angles under <i>more constraints</i>); the others follow. Each solution
branch is a column. <i>min/max</i> drop branches outside a range; a circle with
both bounds may be shifted by 360. A second reflection (<i>2nd (h,k,l)</i>, normally
not held) can be kept in Bragg condition too; it counts as one constraint. The table then adds the angle
between the two scattering vectors and between their outgoing beams, and, for the
reflection the detector is not on, where gamma and delta would put it there with the
virtual angles of that reflection (2theta, qaz, beta, betaout, psi, tau, omega; the
sample angles, alpha, betain and naz are the same for both). The 3D view draws that
second detector position translucent, with its own cut of the Ewald sphere.</p>

<p><b>Energy and second reflection as constraints.</b> Both are rows in the
<i>angles</i> tab. The energy is normally held (<i>fix</i>) at the value set; unticked,
it is one more unknown within its min/max (empty: 1 - 30 keV), and one more angle has
to be held - e.g. the whole crystal (Laue-like: which energy brings
(h,k,l) into Bragg condition, and where the detector goes) or the detector (which
energy and sample angles reach it there). Each solution then has its own energy, and the
energy field follows the selected solution, so the 3D view and the tools use it. When
a mode has one held angle too many, the solutions panel offers <i>let the energy
vary</i> directly.</p>

<p><b>3D view.</b> Click a solution column to drive the drawing. <b>constraint
configuration</b> shows the angle values entered under <i>angles</i> (also the values of
the held constraints), whether or not they solve anything, and the line under the table
says which (h,k,l) is in Bragg condition there &mdash; a simulator for any position.
<b>zero position</b> sets every circle to 0, where the orientation ((h k l) &gt; lab
direction) is defined. The view's
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
inputs to a JSON file (with a CIF's atoms), <i>load config</i> reads one back (also
files of the earlier diffcalc version). <i>link to this state</i> gives web addresses
that open the app as it is now (on Binder, in the browser version), with everything in
the address itself - nothing is stored anywhere. <i>In this browser</i>: the app remembers its state
(switch it off there) and keeps snapshots - take named ones, or go back to an
automatic one (last session, every 10 minutes, before "start fresh" or a
restore); each can be downloaded as a config file. Nothing leaves the browser.</p>
</div>
"""


# ---------------------------------------------------------------- links
# A link carries the whole state (autosave_state(), with the CIF's atoms) as
# ?state=<base64url of zlib-compressed JSON>. It is read from the address of
# the page the app runs in, which the 3D page reports (hklforge >= 0.2.3),
# or under voila from the kernel's QUERY_STRING. Binder passes it on inside
# urlpath; the browser version's static server ignores it.

DISPLAY_HOLD = 3.0        # s: how long a display block rides on the 3D view's pushes
LINK_PARAM = "state"
APP_NOTEBOOK = "apps/diffraction_reflections.ipynb"
BINDER_URL = "https://mybinder.org/v2/gh/swissfel/bernina-scripts/HEAD"
BROWSER_URL = "https://swissfel.github.io/bernina-scripts/voici/render/diffraction_reflections.html"
LINK_LONG = 6000          # characters: longer links may be refused by some servers


def encode_link_state(state: dict) -> str:
    raw = json.dumps(state, separators=(",", ":"), ensure_ascii=False).encode()
    return base64.urlsafe_b64encode(zlib.compress(raw, 9)).rstrip(b"=").decode()


def decode_link_state(text: str) -> dict:
    text = str(text).strip()
    data = zlib.decompress(base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)))
    state = json.loads(data.decode())
    if not isinstance(state, dict):
        raise ValueError("not an app state")
    return state


def state_param(address: str | None) -> str | None:
    """The ?state=... (or #state=...) of an address, or None."""
    if not address:
        return None
    parts = urlsplit(str(address))
    for part in (parts.query, parts.fragment):
        values = parse_qs(part).get(LINK_PARAM)
        if values:
            return values[0]
    return None


def with_state(address: str, param: str) -> str:
    """``address`` with its ?state= set to ``param`` (other query parameters kept, no fragment)."""
    parts = urlsplit(address)
    query = [(k, v) for k, vs in parse_qs(parts.query).items() for v in vs if k != LINK_PARAM]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query + [(LINK_PARAM, param)]), ""))


def without_state(address: str) -> str:
    """``address`` without its ?state= / #state= (other query parameters kept)."""
    parts = urlsplit(address)
    query = [(k, v) for k, vs in parse_qs(parts.query).items() for v in vs if k != LINK_PARAM]
    frag = "" if LINK_PARAM in parse_qs(parts.fragment) else parts.fragment
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), frag))


def state_links(param: str, address: str | None = None) -> dict:
    """{name: url} opening the app with this state: in the browser version
    (first: no server, opens fastest), on Binder and, when known, at the
    address it runs at now."""
    links = {"in the browser": with_state(BROWSER_URL, param)}
    here = urlsplit(address or "")
    # not a JupyterHub session (Binder's /user/...: gone with the session), not the browser version
    if (here.scheme in ("http", "https") and "/user/" not in here.path
            and here.netloc != urlsplit(BROWSER_URL).netloc):
        links["this address"] = with_state(address, param)
    links["Binder"] = f"{BINDER_URL}?urlpath=" + quote(f"voila/render/{APP_NOTEBOOK}?{LINK_PARAM}={param}", safe="")
    return links


# the tools' settings kept in the remembered state, snapshots and links
# (autosave_state "tools"; the detector preset and size have their own keys)
TOOL_SETTINGS = ("t_hkls", "t_sol", "t_source", "t_dmin", "t_max", "t_group", "t_dir", "t_one",
                 "r_source", "r_dmin", "r_max", "r_aperture", "r_beta", "r_only",
                 "d_tol", "d_qty", "d_res", "d_follow", "c_dmin", "c_forbidden",
                 "g_formula", "g_density", "g_delta", "g_beta", "g_beam", "s_from", "s_to", "s_steps")


def _set_if_valid(widget, value):
    """Set a widget from a stored value, leaving it alone if the value does
    not fit (another type, an option that no longer exists, out of bounds)."""
    current = widget.value
    if isinstance(current, bool):
        if isinstance(value, bool):
            widget.value = value
        return
    if isinstance(current, (int, float)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return
    elif isinstance(current, str) and not isinstance(value, str):
        return
    if isinstance(widget, (widgets.Dropdown, widgets.ToggleButtons)):
        values = [o[1] if isinstance(o, tuple) else o for o in widget.options]
        if value not in values:
            return
    with contextlib.suppress(Exception):            # bounds, traits
        widget.value = value


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
    out.show_df(df, floatfmt, max_rows)


def _progress_bar():
    return widgets.IntProgress(value=0, min=0, max=1, layout=L(width="220px", display="none"))


@contextlib.contextmanager
def _working(area, button=None, progress=None, text="computing ..."):
    """While a tool runs: a "computing" line, its button disabled, and a
    progress bar the tool advances with the ``step(i, n)`` this yields."""
    area.busy(text)
    if button is not None:
        button.disabled = True

    def step(i, n):
        if progress is not None:
            progress.max, progress.value = max(n, 1), i
            progress.layout.display = ""

    try:
        yield step
    finally:
        area.done()
        if button is not None:
            button.disabled = False
        if progress is not None:
            progress.layout.display = "none"


# Column groups a table's view can fold away (ResultArea's bar): the
# diffractometer's own circles ("real" angles), the virtual (physical) angles
# and the crystal columns. Every other column always shows.
REAL_ANGLES = tuple(dif.GEOMETRY.names) + ("energy",)
VIRTUAL_ANGLES = ("theta", "ttheta", "qaz", "naz", "alpha", "beta", "psi", "tau", "betain", "betaout", "omega")
CRYSTAL_COLUMNS = ("d [A]", "2theta", "allowed", "why not", "mult", "|F|", "|F|²", "I rel")
TRANSPOSE_SYMBOL = "\u21c4\u1d40"          # ⇄ᵀ


class ResultArea:
    """Where a tool writes its result: an HTML widget whose value is set
    directly - ``with area: print(...)`` collects text, ``show_df`` adds a
    table, ``figure`` an image, ``busy`` a "computing" line.

    Not an Output widget: in the browser version (xeus-python) output printed
    from a button callback never reaches an Output widget, so the tools
    seemed to do nothing. Setting a widget's value works in every front-end.

    Above a table a small bar folds column groups away (real angles,
    virtual angles, crystal columns) and transposes it (``TRANSPOSE_SYMBOL``);
    it only changes how the stored tables are drawn.
    """

    def __init__(self, layout=None):
        self.html = widgets.HTML(layout=L(width="100%"))
        chk = lambda d, tip: widgets.Checkbox(value=True, description=d, indent=False, tooltip=tip,  # noqa: E731
                                              layout=L(width="auto", margin="0 10px 0 0"))
        self.show_real = chk("real angles", "the diffractometer circles (" + ", ".join(REAL_ANGLES) + ")")
        self.show_virtual = chk("virtual angles", "theta, 2theta, qaz, naz, alpha, beta, psi, tau, betain, betaout, omega")
        self.show_crystal = chk("crystal", "d, 2theta, allowed, why not, multiplicity, |F|, |F|², I rel")
        self.transpose = widgets.ToggleButton(value=False, description=TRANSPOSE_SYMBOL, tooltip="transpose the table",
                                              layout=L(width="44px", height="24px"))
        for c in (self.show_real, self.show_virtual, self.show_crystal, self.transpose):
            c.observe(lambda _c: self._render(), names="value")
        self.bar = widgets.HBox([widgets.HTML('<span style="font-size:12px;color:#555;margin-right:6px">columns:</span>'),
                                 self.show_real, self.show_virtual, self.show_crystal, self.transpose],
                                layout=L(display="none", align_items="center"))
        self.widget = widgets.VBox([self.bar, self.html], layout=layout or L(width="100%"))
        self._parts = []
        self._busy = ""

    def view(self, df):
        """``df`` with the folded column groups left out, transposed if asked."""
        hide = set()
        for box, names in ((self.show_real, REAL_ANGLES), (self.show_virtual, VIRTUAL_ANGLES),
                           (self.show_crystal, CRYSTAL_COLUMNS)):
            if not box.value:
                hide.update(names)
        keep = [c for c in df.columns if str(c) not in hide]
        out = df[keep]
        return out.T if self.transpose.value else out

    def _table_html(self, df, floatfmt, max_rows):
        df = self.view(df)
        with pd.option_context("display.max_rows", max_rows):
            table = df.to_html(float_format=floatfmt.format, max_rows=max_rows, border=0, na_rep="")
        return ('<div style="overflow-x:auto;font-size:12px" class="bernina-table">'
                '<style>.bernina-table table{border-collapse:collapse}.bernina-table th,'
                '.bernina-table td{padding:1px 6px;text-align:right;white-space:nowrap}'
                '.bernina-table tr:nth-child(even){background:#f6f8fa}</style>' + table + "</div>")

    def _render(self):
        tables = [p for p in self._parts if not isinstance(p, str)]
        self.bar.layout.display = "flex" if tables else "none"
        self.html.value = self._busy + "".join(p if isinstance(p, str) else self._table_html(*p)
                                               for p in self._parts)

    def clear_output(self):
        self._parts = []
        self._render()

    def busy(self, text="computing ..."):
        """A "computing" line above the results, until ``done``."""
        self._busy = f'<div style="font-size:12px;color:#a60">&#8987; {html.escape(text)}</div>'
        self._render()

    def done(self):
        self._busy = ""
        self._render()

    def add_html(self, fragment: str):
        self._parts.append(fragment)
        self._render()

    def __enter__(self):
        self._buf = io.StringIO()
        self._redirect = contextlib.redirect_stdout(self._buf)
        self._redirect.__enter__()
        return self

    def __exit__(self, *exc):
        self._redirect.__exit__(*exc)
        text = self._buf.getvalue()
        if text:
            self.add_html('<pre style="font-size:12px;margin:2px 0;white-space:pre-wrap">'
                          f'{html.escape(text.rstrip())}</pre>')
        return False

    def show_df(self, df, floatfmt="{:.3f}", max_rows=200):
        self._parts.append((df, floatfmt, max_rows))      # drawn by _render, through view()
        self._render()

    def figure(self, fig, dpi=90):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        self.add_html(f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">')


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
        self._display_out = None      # (display, until): sent with every push for DISPLAY_HOLD seconds
        self._restore_done = False  # until the 3D page has offered the state it remembers
        self._forget = False
        self._memory_ops = []       # snapshot requests for the 3D page, numbered (see push_view)
        self._memory_seq = 0
        self._memory_loading = False
        self.snapshots = []         # as the 3D page reports them (oldest first)
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
        names = cif_examples.names() if cif_examples is not None else []
        self.cif_example = w.Dropdown(options=[("example CIF ...", "")] + [(cif_examples.title(n), n) for n in names],
                                      value="", layout=L(width="190px", display="flex" if names else "none"),
                                      tooltip="load one of hklforge's example structures, as if uploaded")
        self.cif_clear = w.Button(description="remove CIF", layout=L(width="100px", display="none"),
                                  tooltip="forget the CIF's atoms (|F|); the cell and space group stay")
        self.crystal_msg = w.HTML()
        self.ctrl["surface_normal"] = w.Text(value=s["surface_normal"], description="surface normal (h,k,l)",
                                             placeholder="empty: none", layout=L(width="360px"),
                                             style={"description_width": "160px"})
        self.ctrl["reference"] = w.Text(value=s["reference"], description="reference (h,k,l)",
                                        placeholder="for alpha, beta, psi", layout=L(width="360px"),
                                        style={"description_width": "160px"})
        self.ctrl["orientations"] = w.Textarea(
            value=s["orientations"], description="orientations (h,k,l) > lab@zero angles",
            placeholder='one "(h,k,l) > (x,y,z)" per line; or > beam, > up, > x+',
            layout=L(width="410px", height="66px"), style={"description_width": "200px"})

        for n in "hkl":
            self.ctrl[n] = w.FloatText(value=1.0, step=0.1, description=n, layout=L(width=HKL_W),
                                       style={"description_width": "20px"})
        self.ctrl["energy"] = w.BoundedFloatText(value=9000, min=ENERGY_LIMITS[0], max=ENERGY_LIMITS[1], step=10,
                                                 description="energy [eV]", layout=L(width="190px"),
                                                 style={"description_width": "80px"})
        self.energy_slider = w.FloatSlider(value=9000, min=ENERGY_LIMITS[0], max=ENERGY_LIMITS[1], step=10,
                                           readout=False,
                                           continuous_update=False, layout=L(width="170px"))
        w.link((self.ctrl["energy"], "value"), (self.energy_slider, "value"))
        self.energy_free_msg = w.HTML()
        self.target_msg = w.HTML()

        # constraints (angles tab): value, min, max, fix
        self.filters = {}
        self.angle_rows = {}
        # the energy is a constraint too, held at the energy above unless "fix" is
        # unticked: then it is one more unknown within min/max (hklforge >= 0.2.2), and
        # the energy above follows the selected solution (select), so every tool uses it
        self.ctrl["energy_fixed"] = w.Checkbox(value=True, description="fix", indent=False, layout=L(width="48px"))
        energy_value = w.BoundedFloatText(value=9000, min=ENERGY_LIMITS[0], max=ENERGY_LIMITS[1],
                                          layout=L(width=FIELD_W))
        w.link((self.ctrl["energy"], "value"), (energy_value, "value"))
        self.filters["energy"] = (w.Text(placeholder=f"{ENERGY_LIMITS[0]:g}", layout=L(width=FILTER_W)),
                                  w.Text(placeholder=f"{ENERGY_LIMITS[1]:g}", layout=L(width=FILTER_W)))
        self.energy_row = w.HBox([w.Label("energy", layout=L(width=LABEL_W)), energy_value, *self.filters["energy"],
                                  self.ctrl["energy_fixed"], w.Label("eV")])
        # a second reflection kept in Bragg condition: a constraint, normally not held
        self.ctrl["two"] = w.Checkbox(value=False, description="fix", indent=False, layout=L(width="48px"),
                                      tooltip="keep this second reflection in Bragg condition too (counts as one "
                                              "constraint)")
        for n in ("h2", "k2", "l2"):
            self.ctrl[n] = w.FloatText(value=0.0 if n != "l2" else 1.0, step=1, layout=L(width="52px"))
        self.ctrl["detector"] = w.Dropdown(options=[(v, k) for k, v in dif.DETECTOR_POLICIES.items()],
                                           value="h1", layout=L(width="220px"))
        self.h2_row = w.VBox([
            w.HBox([w.Label("2nd (h,k,l)", layout=L(width=LABEL_W)), self.ctrl["h2"], self.ctrl["k2"],
                    self.ctrl["l2"], self.ctrl["two"]], layout=L(align_items="center")),
            w.HBox([w.Label("", layout=L(width=LABEL_W)), self.ctrl["detector"]])])
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
            w.HBox([c["space_group"], self.cif, self.cif_example, self.cif_clear]), self.crystal_msg,
            c["surface_normal"], c["reference"], c["orientations"],
        ])
        self.tab_reflection = w.VBox([
            w.HBox([c["h"], c["k"], c["l"]]), w.HBox([c["energy"], self.energy_slider]),
            self.energy_free_msg, self.target_msg,
            w.HTML('<div style="font-size:11px;color:#666;width:400px;margin-top:6px">the energy and a '
                   'second reflection are constraints as well: <i>angles</i> tab</div>'),
        ])
        # scan one held value: how many solutions (passing the min/max) along it
        self.s_name = w.Dropdown(options=[("(hold a value first)", "")], description="scan",
                                 layout=L(width="200px"), style={"description_width": "36px"})
        self.s_from = w.FloatText(value=-180.0, description="from", layout=L(width="120px"),
                                  style={"description_width": "36px"})
        self.s_to = w.FloatText(value=180.0, description="to", layout=L(width="110px"),
                                style={"description_width": "22px"})
        self.s_steps = w.BoundedIntText(value=181, min=3, max=2001, description="points",
                                        layout=L(width="130px"), style={"description_width": "46px"})
        self.s_btn = w.Button(description="scan", button_style="primary", layout=L(width="70px"),
                              tooltip="solve at every point and plot how many solutions pass the min/max")
        self.s_figure = FigureWidget((4.6, 2.3))
        self.s_fig = self.s_figure.fig
        self.s_msg = w.HTML()
        self._scan = None                 # (name, values, counts) of the last scan
        self.scan_box = w.VBox([
            w.HTML('<div style="font-size:13px;font-weight:600;margin-top:10px">scan a held value</div>'
                   '<div style="font-size:11px;color:#666;width:400px">steps one held angle and counts the '
                   'solutions passing the min/max at each value. Click in the plot to set the angle to '
                   'that value.</div>'),
            w.HBox([self.s_name, self.s_btn]), w.HBox([self.s_from, self.s_to, self.s_steps]),
            self.s_msg, self.s_figure.widget], layout=L(border_top="1px solid #ddd", margin="10px 0 0 0"))
        self.tab_angles = w.VBox([header] + [self.angle_rows[n] for n in dif.CIRCLES]
                                 + [self.energy_row, self.h2_row, self.more_constraints, self.held_msg,
                                    self.scan_box])

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
        self.make_btn = w.Button(description="download", icon="download", button_style="primary",
                                 layout=L(width="120px"),
                                 tooltip="write the file and download it (also kept in this browser's history)")
        self._downloads = 0               # numbers the downloads the 3D page starts (push_view)
        self.file_name = w.Text(value="bernina_diffraction_config", description="file name",
                                layout=L(width="330px"), style={"description_width": "70px"})
        self.upload = w.FileUpload(accept=".json", multiple=False, description="load config",
                                   layout=L(width="150px"))
        self.download_link = w.HTML()
        self.config_msg = w.HTML()
        # a link that opens the app in this state (everything, with the CIF's atoms)
        self.link_btn = w.Button(description="link to this state", icon="link", layout=L(width="150px"),
                                 tooltip="a web address that opens the app as it is now (sample with "
                                         "the CIF, reflection, angles, min/max, tools)")
        self.link_html = w.HTML()
        self.link_text = w.Textarea(layout=L(width="400px", height="60px", display="none"))
        self.page_address = None          # where the app runs, as the 3D page reports it
        self._link_pending = None         # the state of a link waiting for the 3D page's display
        self.store_panel = w.VBox(
            [w.HTML('<div style="font-size:11px;color:#666">include in the file (unchecked groups are '
                    'left out, so a file can be a partial template):</div>')]
            + [self.include[k] for k, _ in self.include_groups]
            + [self.file_name, self.make_btn, self.download_link],
            layout=L(display="none", border="1px solid #ddd", padding="6px", margin="4px 0 0 0"))
        # kept in this browser (the 3D page does the storing, see restore/snapshots below)
        self.remember = w.Checkbox(value=True, indent=False, layout=L(width="auto"),
                                   description="remember my state in this browser, with automatic snapshots")
        self.snap_name = w.Text(placeholder="snapshot name (optional)", layout=L(width="220px"))
        self.snap_btn = w.Button(description="take snapshot", icon="camera", layout=L(width="130px"))
        self.snap_list = w.Dropdown(options=[("(no snapshots yet)", "")], layout=L(width="360px"))
        self.snap_restore = w.Button(description="restore", layout=L(width="80px"))
        self.snap_delete = w.Button(description="delete", layout=L(width="70px"))
        self.snap_download = w.HTML()
        self.fresh_btn2 = w.Button(description="start fresh", layout=L(width="110px"),
                                   tooltip="reload with the defaults (the current state is kept as a snapshot)")
        self.forget_btn = w.Button(description="forget everything", layout=L(width="140px"),
                                   tooltip="delete the remembered state and every snapshot in this browser")
        self.memory_msg = w.HTML(_note("waiting for the 3D view, which keeps them ...", "#888"))
        self.memory_box = w.VBox(
            [w.HTML('<div style="font-size:13px;font-weight:600;margin-top:10px">in this browser</div>'
                    '<div style="font-size:11px;color:#666;max-width:560px">kept in this browser\'s '
                    'storage for this web address only, never sent anywhere; a backup in case nobody '
                    'stored a file. Automatic snapshots: the last session\'s state when a new one starts, '
                    'every 10 minutes of work, before "start fresh" and before a restore (the newest 20). '
                    'Every config file you download is kept here too ("config:", with its date), so '
                    'it can be restored from this list.'
                    '</div>'),
             w.HBox([self.remember]),
             w.HBox([self.snap_name, self.snap_btn]),
             w.HBox([self.snap_list, self.snap_restore, self.snap_delete]),
             self.snap_download,
             w.HBox([self.fresh_btn2, self.forget_btn]),
             self.memory_msg],
            layout=L(border_top="1px solid #ddd", margin="10px 0 0 0"))
        self._set_memory_enabled(False)
        self.tab_config = w.VBox([self.preset, self.preset_msg, w.HBox([self.store_btn, self.upload]),
                                  self.store_panel, self.config_msg, self.link_btn, self.link_html,
                                  self.link_text, self.memory_box])

    # ================================================================ solutions, view

    def _build_solutions(self):
        self.solution_buttons = widgets.HBox([])
        # the two positions that are not solutions, on their own row above them
        self.position_buttons = widgets.HBox([], layout=L(margin="0 0 4px 0"))
        self.table_html = widgets.HTML()
        self.solve_msg = widgets.HTML()
        # offered when hklforge says the energy would make the mode complete (or the opposite)
        self.energy_btn = widgets.Button(layout=L(width="auto", display="none"))
        self.config_hkl = widgets.HTML()

    def _build_view(self):
        self.view_frame = widgets.HTML(view_iframe(bernina_view_page(), 760, VIEW_ID), layout=L(width="100%"))
        self._bz_key, self._bz = None, None
        self._sf_key, self._sf = None, None
        self.payload_holder = widgets.HTML()
        # the 3D page writes its detector choice here (see its sendDetector)
        self.det_from_view = widgets.Text(layout=L(display="none"))
        self.det_from_view.add_class("diff3d-detector-in")
        # ... and what it keeps in this browser: once the state it remembers from the
        # last session ({"kind": "offer"}), then the snapshots whenever they change
        self.state_from_view = widgets.Text(layout=L(display="none"))
        self.state_from_view.add_class("diff3d-state-in")

    # ================================================================ tools

    def _build_tools(self):
        w = widgets
        # reflection table
        self.t_hkls = w.Textarea(value="1,1,1\n2,2,0\n3,1,1\n0,0,4", description="h,k,l list",
                                 layout=L(width="220px", height="120px"))
        self.t_sol = w.IntText(value=0, description="solution #", layout=L(width="170px"))
        self.t_source = w.Dropdown(options=[("allowed reflections down to d min", "all"), ("the h,k,l list", "list")],
                                   description="reflections", layout=L(width="330px"),
                                   style={"description_width": "70px"})
        self.t_dmin = w.FloatText(value=1.5, description="d min [Å]", layout=L(width="150px"),
                                  style={"description_width": "64px"})
        self.t_max = w.IntText(value=100, description="at most", layout=L(width="130px"),
                               style={"description_width": "52px"})
        self.t_group = w.Dropdown(options=[(label, key) for key, label in TABLE_GROUPS.items()],
                                  description="group", layout=L(width="330px"), style={"description_width": "70px"})
        self.t_dir = w.Text(value="[0 0 1]", placeholder="[u v w] or (h k l)", layout=L(width="110px", display="none"),
                            tooltip="a real-space direction [u v w] or the normal of the planes (h k l)")
        self.t_one = w.Checkbox(value=False, description="one row per group", indent=False, layout=L(width="auto"))
        self.t_btn = w.Button(description="compute table", button_style="primary")
        self.t_progress = _progress_bar()
        self.t_out = ResultArea()
        tool_table = w.VBox([
            w.HBox([self.t_source, self.t_dmin, self.t_max], layout=L(flex_flow="row wrap")),
            w.HBox([self.t_hkls, self.t_sol]),
            w.HBox([self.t_group, self.t_dir, self.t_one], layout=L(flex_flow="row wrap", align_items="center")),
            w.HBox([self.t_btn, self.t_progress], layout=L(align_items="center")),
            w.HTML('<div style="font-size:11px;color:#666;white-space:normal;line-height:1.3;width:640px">'
                   'one row per reflection at the solution # among those passing the min/max. With a space '
                   'group: allowed and why not; with a CIF: |F|, |F|² (with f′, f″ at this energy) and I rel. '
                   '<i>Same |Q| and angle to an axis</i>: reflections a rotation about it maps onto each other '
                   '(same 2θ, same exit direction relative to the axis); Δ is the turn about the axis (for '
                   'phi: the change of phi) that brings each to where the group\'s first one is.</div>'),
            self.t_out.widget])

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
        self.r_progress = _progress_bar()
        self.r_out = ResultArea()
        tool_reach = w.VBox([self.r_source, w.HBox([self.r_dmin, self.r_max]),
                             w.HBox([self.r_aperture]), self.r_beta, w.HBox([self.r_btn, self.r_only, self.r_progress], layout=L(align_items="center")),
                             w.HTML('<span style="font-size:11px;color:#666">uses the constraints, the '
                                    'min/max and the energy above; THC envelopes from the THC app</span>'),
                             self.r_out.widget])

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
        self.d_out = ResultArea(layout=L(flex="1 1 380px"))
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
                           self.d_plot, self.d_readout, self.d_out.widget], layout=L(width="100%"))

        # crystal
        self.c_dmin = w.FloatText(value=1.0, description="d min [Å]", layout=L(width="170px"),
                                  style={"description_width": "70px"})
        self.c_forbidden = w.Checkbox(value=False, description="show forbidden too", indent=False)
        self.c_btn = w.Button(description="list reflections", button_style="primary")
        self.c_out = ResultArea()
        tool_crystal = w.VBox([
            w.HTML('<span style="font-size:11px;color:#666">one reflection per symmetry-equivalent set, '
                   'from the space group (and the CIF atoms for |F|); see the sample tab</span>'),
            w.HBox([self.c_dmin, self.c_forbidden]), self.c_btn, self.c_out.widget])

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
        self.g_out = ResultArea()
        tool_grazing = w.VBox([w.HBox([self.g_formula, self.g_density]), w.HBox([self.g_delta, self.g_beta]),
                               self.g_beam, self.g_btn,
                               w.HTML('<span style="font-size:11px;color:#666">for the energy above and, '
                                      'if a solution is selected, its betain/betaout (needs a surface '
                                      'normal)</span>'),
                               self.g_out.widget])

        # geometry: how a diffractometer geometry is built (hklforge's explorer),
        # made when the tab is first opened - it carries a second 3D page
        self.geometry_box = w.VBox([w.HTML('<span style="font-size:12px;color:#666">loading ...</span>')],
                                   layout=L(width="100%"))
        self.explorer = None

        self.tool_tabs = w.Tab(children=[tool_table, tool_reach, tool_det, tool_crystal, tool_grazing,
                                         self.geometry_box])
        for i, t in enumerate(("table", "reachable", "detector", "crystal", "grazing", "geometry")):
            self.tool_tabs.set_title(i, t)
        self.tool_tabs.observe(self._on_tool_tab, names="selected_index")

    def _on_tool_tab(self, change):
        if change["new"] == 5:
            self.build_geometry_tab()

    def build_geometry_tab(self):
        """The geometry tab's content (once): Bernina's geometry first, then hklforge's stored ones."""
        if self.explorer is not None:
            return self.explorer
        from hklforge.view.explorer import GeometryExplorer
        self.explorer = GeometryExplorer({"bernina (You, SwissFEL frame)": dif.GEOMETRY}, view_id="geo3d",
                                         height=560)
        self.geometry_box.children = [
            widgets.HTML('<div style="font-size:11px;color:#666;width:720px;white-space:normal">How a '
                         'diffractometer geometry is built, circle by circle - Bernina\'s and others '
                         '(from libhkl, diffcalc). For understanding only: the app solves with '
                         'Bernina\'s geometry.</div>'),
            self.explorer.widget]
        return self.explorer

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
                widgets.HBox([self.solve_msg, self.energy_btn], layout=L(align_items="center")),
                self.position_buttons, self.solution_buttons, self.table_html,
                self.config_hkl])),
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
        self.cif_example.observe(self._on_cif_example, names="value")
        self.cif_clear.on_click(self._clear_cif)
        self.preset.observe(lambda ch: self.load_preset(ch["new"]), names="value")
        self.store_btn.on_click(self._toggle_store)
        self.make_btn.on_click(self._on_make)
        self.file_name.observe(lambda _c: setattr(self.download_link, "value", ""), names="value")
        self.upload.observe(self._on_upload, names="value")
        self.link_btn.on_click(self.make_link)
        self.layout_switch.observe(self.arrange, names="value")
        self.layout_switch.observe(self._view_controls_for_layout, names="value")
        self.view_height.observe(lambda _c: self.push_view(), names="value")
        self.state_from_view.observe(self._state_from_view, names="value")
        self.fresh_btn.on_click(self._start_fresh)
        self.fresh_btn2.on_click(self._start_fresh)
        self.remember.observe(self._on_remember, names="value")
        self.snap_btn.on_click(self._take_snapshot)
        self.snap_list.observe(self._show_snapshot_link, names="value")
        self.snap_restore.on_click(self._restore_snapshot)
        self.snap_delete.on_click(self._delete_snapshot)
        self.forget_btn.on_click(self._forget_everything)
        self.help_btn.observe(lambda c: setattr(self.help_box.layout, "display", "flex" if c["new"] else "none"),
                              names="value")
        self.t_btn.on_click(self.run_table)
        self.t_group.observe(lambda c: setattr(self.t_dir.layout, "display", "" if c["new"] == "dir" else "none"),
                             names="value")
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
        self.s_btn.on_click(self.run_scan)
        self.s_name.observe(self._scan_defaults, names="value")
        if self.s_figure.canvas is not None:
            self.s_figure.canvas.mpl_connect("button_press_event", self._scan_click)
        self.g_btn.on_click(self.run_grazing)
        self.energy_btn.on_click(lambda _b: setattr(self.ctrl["energy_fixed"], "value", self.energy_free()))

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
        if self.energy_free():
            cons["energy"] = self.energy_limits()
        return cons

    def energy_free(self) -> bool:
        return not self.ctrl["energy_fixed"].value

    def energy_limits(self) -> tuple:
        """(lo, hi) eV within which a free energy may go (the energy row's min/max);
        empty fields: the energy field's range."""
        lo, hi = (_num_or_none(t.value) for t in self.filters["energy"])
        lo = ENERGY_LIMITS[0] if lo is None else min(max(lo, ENERGY_LIMITS[0]), ENERGY_LIMITS[1])
        hi = ENERGY_LIMITS[1] if hi is None else min(max(hi, ENERGY_LIMITS[0]), ENERGY_LIMITS[1])
        return (lo, hi) if lo <= hi else (hi, lo)

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
        self._refresh_scan_names()
        self._check_crystal()
        try:
            self.sample = dif.Sample.from_text(self.cell(), self.ctrl["orientations"].value,
                                               self.ctrl["surface_normal"].value, self.ctrl["reference"].value)
        except Exception as exc:
            self.sample = None
            self._no_solutions(_err(f"sample: {exc}"))
            return
        cons = self.constraints()
        free = "energy" in cons
        held, need = len(cons) - free, 3 + free
        self.held_msg.value = ("" if held == need else
                               _note(f"{held} constraint(s) held; {'four' if free else 'three'} are needed for "
                                     "a finite set of solutions (the second reflection counts as one"
                                     f"{'; one more because the energy may vary' if free else ''})"))
        self.energy_free_msg.value = (_note(f"solved with the energy free in {cons['energy'][0]:g} - "
                                            f"{cons['energy'][1]:g} eV (one more angle held); the energy above "
                                            "follows the selected solution", "#666") if free else "")
        self.energy_btn.layout.display = "none"
        try:
            df = self.sample.solve(self.hkl(), self.energy(), cons, self.ctrl["detector"].value)
        except Exception as exc:
            text = str(exc)
            if ENERGY_HINT in text:          # hklforge: over-determined, the energy could take one more
                text = text.replace(ENERGY_HINT, "")
                self.energy_btn.description, self.energy_btn.layout.display = "let the energy vary", "flex"
            elif free and "needs one more circle or constraint held" in text:
                self.energy_btn.description, self.energy_btn.layout.display = "fix the energy", "flex"
            self._no_solutions(_err(text))
            return
        if self.ctrl["two"].value and len(df.columns):
            df = pd.concat([df, pd.DataFrame(self._two_reflection_rows(df, cons["h2"]))])
        self.df = df
        self.cols = [c for c in df.columns if dif.passes(df[c], self.ranges())]
        hidden = len(df.columns) - len(self.cols)
        if df.empty and free:
            self.solve_msg.value = _note(f"no solution with the energy in {cons['energy'][0]:g} - "
                                         f"{cons['energy'][1]:g} eV")
        elif df.empty:
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

    def _ghost_detectors(self) -> list:
        """For the 3D view: the detector on each reflection it is not on, for a
        solution shown in two-reflection mode (read from the table's rows)."""
        if not (self.ctrl["two"].value and isinstance(self.selected, int) and self.df is not None
                and self.selected < len(self.cols)):
            return []
        col = self.df[self.cols[self.selected]]
        hkls = {1: self.hkl(), 2: [float(self.ctrl[n].value) for n in ("h2", "k2", "l2")]}
        out = []
        for n in self._other_reflections():
            g, d = col.get(f"gamma @{n}"), col.get(f"delta @{n}")
            if g is not None and d is not None and np.isfinite(g) and np.isfinite(d):
                out.append({"gamma": float(g), "delta": float(d), "which": n,
                            "label": "(" + " ".join(f"{x:g}" for x in hkls[n]) + ")"})
        return out

    def _other_reflections(self):
        """Which reflections (1, 2) the detector is not on, in two-reflection mode."""
        return (2,) if self.ctrl["detector"].value == "h1" else (1, 2)

    def _two_reflection_rows(self, df, h2) -> dict:
        """Rows added per solution with two reflections: the angle between
        them, between their outgoing beams, the largest detector distance
        that keeps both on the face, and - for each reflection the detector
        is not on - where the detector would sit on it and the virtual angles
        of its outgoing beam ("<angle> @<n>"; the sample angles are shared)."""
        h1 = self.hkl()
        q_angle = dif.angle_between(self.sample, h1, h2)
        rows = {}
        for c in df.columns:
            a = dif.solution_angles(df, c)
            energy = self._solution_energy(df, c)
            sp = self.sample.spots(a, [h1, h2], energy)
            r = {"∠(Q1,Q2)": q_angle, "∠(k′1,k′2)": sp["separation"], "tdet max [mm]": sp["tdet_max"]}
            for n in self._other_reflections():
                for name, v in dif.reflection_angles(self.sample, a, (h1, h2)[n - 1], energy).items():
                    if name != "theta" and (name in df.index or name in ("gamma", "delta")):
                        r[f"{name} @{n}"] = v
            rows[c] = r
        return rows

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
        config = w.Button(description=POSITION_NAMES[SEL_CONFIG], layout=L(width="190px", margin="0"),
                          tooltip="the angle values entered under 'angles' (also the values of the held "
                                  "constraints), whether or not they solve anything")
        config.on_click(lambda _b: self.select(SEL_CONFIG))
        config._sel = SEL_CONFIG
        zero = w.Button(description=POSITION_NAMES[SEL_ZERO], layout=L(width="120px", margin="0 0 0 8px"),
                        tooltip="every circle at 0: where the orientation (h k l) > lab direction is defined")
        zero.on_click(lambda _b: self.select(SEL_ZERO))
        zero._sel = SEL_ZERO
        self.position_buttons.children = [config, zero]
        spacer = w.HTML(layout=L(width=f"{ROW_LABEL_W}px"))          # keeps them over the table's columns
        self.solution_buttons.children = [spacer] + buttons

    def _render_table(self):
        if self.df is None or not self.cols:
            self.table_html.value = ""
            return
        df, sel = self.df, self.selected
        bg = lambda i: f"background:{SELECTED_BG};" if i == sel else ""
        head = "".join(f'<th style="width:{COL_W}px;text-align:right;padding:2px 6px;border-bottom:1px solid #ccc;'
                       f'{bg(i)}">{html.escape(str(c))}</th>' for i, c in enumerate(self.cols))
        body, group = [], None
        span = len(self.cols) + 1
        hkls = {1: self.hkl(), 2: [self.ctrl[n].value for n in ("h2", "k2", "l2")]}
        for name in df.index:
            label, _, which = str(name).partition(" @")
            if which and which != group:
                group = which
                hkl = " ".join(f"{float(x):g}" for x in hkls[int(which)])
                body.append(f'<tr><td colspan="{span}" style="padding:6px 6px 1px;color:#555;'
                            f'border-top:1px solid #ddd">detector on reflection {which} ({hkl}), '
                            'same sample angles</td></tr>')
            elif label == "∠(Q1,Q2)":
                body.append(f'<tr><td colspan="{span}" style="padding:6px 6px 1px;color:#555;'
                            'border-top:1px solid #ddd">both reflections</td></tr>')
            cells = "".join(f'<td style="width:{COL_W}px;text-align:right;padding:1px 6px;'
                            f'font-variant-numeric:tabular-nums;{bg(i)}">{_fmt(df.loc[name, c])}</td>'
                            for i, c in enumerate(self.cols))
            body.append(f'<tr><td style="width:{ROW_LABEL_W}px;text-align:right;padding:1px 6px;font-weight:600">'
                        f'{html.escape(label)}</td>{cells}</tr>')
        self.table_html.value = ('<table style="border-collapse:collapse;font-size:12px">'
                                 f'<tr><td style="width:{ROW_LABEL_W}px"></td>{head}</tr>' + "".join(body) + "</table>")

    def shown_angles(self):
        """(angles, label) of what the 3D view shows."""
        if self.selected == SEL_CONFIG:
            return self.config_angles(), POSITION_NAMES[SEL_CONFIG]
        if isinstance(self.selected, int) and self.df is not None and self.selected < len(self.cols):
            col = self.cols[self.selected]
            return dif.solution_angles(self.df, col), str(col)
        return {}, POSITION_NAMES[SEL_ZERO]

    def _solution_energy(self, df, col) -> float:
        """A solution's own energy (the energy free), else the energy field's."""
        if "energy" in df.index and np.isfinite(df.loc["energy", col]):
            return float(df.loc["energy", col])
        return self.energy()

    def select(self, sel):
        self.selected = sel
        if (self.energy_free() and isinstance(sel, int) and self.df is not None and sel < len(self.cols)
                and "energy" in self.df.index):
            e = round(self._solution_energy(self.df, self.cols[sel]), 3)
            if ENERGY_LIMITS[0] <= e <= ENERGY_LIMITS[1] and e != self.ctrl["energy"].value:
                self._suspend, was = True, self._suspend
                try:
                    self.ctrl["energy"].value = e   # every tool and the 3D view now use it
                finally:
                    self._suspend = was
        for b in self.position_buttons.children + self.solution_buttons.children:
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
            '<div style="font-size:12px;margin-top:6px"><b>constraint configuration</b> (the angles entered) puts '
            f'({", ".join(f"{x:.3f}" for x in h)}) in Bragg condition; nearest '
            f'({" ".join(str(int(x)) for x in near + 0.0)}), off by {np.linalg.norm(h - near):.3f}</div>')

    def push_view(self):
        angles, label = self.shown_angles()
        targets = [self.hkl()] + ([[float(self.ctrl[n].value) for n in ("h2", "k2", "l2")]]
                                  if self.ctrl["two"].value else [])
        s = self.sample
        msg = view_message(None, dif._complete(angles), k=dif.k_magnitude(self.energy()), label=label,
                           targets=targets, n=self._pushes, ub=None if s is None else s.UB,
                           special_points=self._special_points())
        if structure_factor_nodes is not None:   # the 3D view's spheres (None: no structure)
            msg["structureFactors"] = self._structure_factors()
        if structure_atoms is not None:          # the 3D view's real-space atoms (None: lattice points)
            msg["structure"] = self._structure_atoms()
        if s is not None:      # sent as None too, so that emptying a field clears it in the view
            msg["nPhi"] = None if s.n_surface is None else [float(x) for x in s.n_surface]
            msg["refPhi"] = None if s.n_reference is None else [float(x) for x in s.n_reference]
        if self._downloads:
            msg["download"] = self._downloads              # the page clicks the config link once per number
        try:
            msg["centring"] = dv.lattice_centring(self.space_group())   # for the Brillouin zones
        except Exception:
            msg["centring"] = "P"
        msg["frameHeight"] = int(self.view_height.value) if hasattr(self, "view_height") else 760
        msg["ghosts"] = self._ghost_detectors()
        if hasattr(self, "d_w"):
            msg["detectors"] = [[k, v[0], v[1]] for k, v in dif.DETECTOR_PRESETS.items()]
            msg["detector"] = {"w": float(self.d_w.value), "h": float(self.d_h.value)}
        # The page reads the newest payload every 150 ms, so a display block sent
        # in only one push of a burst (a restore pushes several) can be missed:
        # it rides on every push for DISPLAY_HOLD seconds (applying it again is harmless).
        now = time.monotonic()
        if self.display_settings is not None:
            held = self._display_out[0] if self._display_out and now < self._display_out[1] else {}
            self._display_out = ({**held, **self.display_settings}, now + DISPLAY_HOLD)
            self.display_settings = None
        if self._display_out and now < self._display_out[1]:
            msg["display"] = self._display_out[0]
        if self._restore_done:
            msg["memory"] = {"remember": bool(self.remember.value), "ops": list(self._memory_ops)}
        if self._forget:
            msg["forgetState"] = True
            if state_param(self.page_address) is not None:   # opened from a link: reload without it
                msg["reloadTo"] = without_state(self.page_address)
        elif self._restore_done:
            msg["autosave"] = self.autosave_state()
        self._pushes += 1
        self.payload_holder.value = payload_html(msg, VIEW_ID)
        if getattr(self, "d_follow", None) is not None and self.d_follow.value and self._det is not None:
            self.run_detector()

    def _structure_factors(self) -> dict | None:
        """|F|^2 of every reflection reachable at the energy, for the 3D view's
        structure-factor spheres (cached per structure and energy), or None."""
        if self.structure is None:
            return None
        key = (self.structure, round(self.energy(), 1))     # the object itself: a new CIF is a new one
        if self._sf_key is None or key[0] is not self._sf_key[0] or key[1] != self._sf_key[1]:
            try:
                sf = structure_factor_nodes(self.structure, self.energy())
            except Exception:                  # xraydb missing, or far too many reflections
                sf = None
            self._sf_key, self._sf = key, sf
        return self._sf

    def _structure_atoms(self) -> dict | None:
        """The atoms of one cell for the 3D view's real-space drawing (cached), or None."""
        if self.structure is None:
            return None
        if getattr(self, "_atoms_for", None) is not self.structure:
            try:
                self._atoms = structure_atoms(self.structure)
            except Exception:
                self._atoms = None
            self._atoms_for = self.structure
        return self._atoms

    def _special_points(self) -> dict | None:
        """Brillouin-zone special points of the cell and space group (cached), or None."""
        key = (tuple(self.cell().values()), self.ctrl["space_group"].value.strip(),
               None if self.structure is None else self.structure.name)
        if key != self._bz_key:
            try:
                sg = self.space_group()
                if sg is None and self.structure is not None:
                    sg = self.structure.space_group
                # without a space group the lattice centring, and so the points, are unknown
                sp = None if sg is None else hbz.special_points(self.cell(), sg)
            except Exception:
                sp = None
            self._bz_key, self._bz = key, sp
        return self._bz

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
        self._load_cif_text(cry.to_cif_text(item["content"]))

    def _on_cif_example(self, change):
        name = change["new"]
        if not name:
            return
        try:
            self._load_cif_text(cif_examples.cif_text(name))
        finally:
            self.cif_example.value = ""          # back to the prompt, so the same one can be chosen again

    def _load_cif_text(self, text: str):
        """A CIF (uploaded or an example): structure, space group and cell."""
        try:
            st = cry.read_cif(text)
        except Exception as exc:
            self.crystal_msg.value = _err(f"CIF: {exc}")
            return
        self._suspend = True
        try:
            self._set_structure(st)
            self.ctrl["space_group"].value = st.space_group.name if st.space_group is not None else ""
        finally:
            self._suspend = False
        self._set_cell(st.cell)
        notes = list(st.notes)
        if st.space_group is None:
            notes.append("space group not recognised: |F| uses the CIF's symmetry operations, "
                         "absences are not checked")
        if notes:
            self.crystal_msg.value = _note(html.escape("CIF: " + "; ".join(notes)))

    def _set_structure(self, st, notes: list | None = None):
        """The CIF's structure: a ``Structure``, its ``to_dict()`` (from a config
        or link) or None. Fills the grazing tab's formula and density from it."""
        if isinstance(st, dict):
            try:
                st = cry.Structure.from_dict(st)
            except Exception as exc:
                st = None
                if notes is not None:
                    notes.append(f"the CIF structure could not be read back ({exc})")
        self.structure = st
        self.cif_clear.layout.display = "none" if st is None else "flex"
        if st is not None:
            with contextlib.suppress(Exception):       # formula and density need xraydb
                self.g_formula.value = st.formula
                self.g_density.value = round(st.density, 4)

    def _clear_cif(self, _b=None):
        self._set_structure(None)
        with contextlib.suppress(Exception):           # so the same file can be loaded again
            self.cif.value = ()
        self.recompute()

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
            self.ctrl["energy_fixed"].value = True
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
        cfg = {"version": CONFIG_VERSION}
        if inc["sample"]:
            cfg["sample"] = {k: self.ctrl[k].value for k in DEFAULT_SAMPLE}
            if self.structure is not None:          # the CIF's atoms, so |F| and its symmetry come back
                cfg["structure"] = self.structure.to_dict()
        if inc["hkl"]:
            cfg["hkl"] = {n: self.ctrl[n].value for n in ("h", "k", "l", "two", "h2", "k2", "l2", "detector")}
        if inc["energy"]:
            cfg["energy"] = self.ctrl["energy"].value
        if inc["angles"]:
            cfg["angles"] = {n: self.ctrl[n].value for n in dif.CIRCLES + dif.VIRTUAL_CONSTRAINTS}
        if inc["fixed"]:
            cfg["fixed"] = {n: self.ctrl[f"{n}_fixed"].value for n in dif.CONSTRAINTS + ("energy",)}
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
        tools = {"detector": self.d_preset.value, "width": self.d_w.value, "height": self.d_h.value,
                 "distance": self.d_dist.value, "tab": self.tool_tabs.selected_index,
                 "overlays": {k: cb.value for k, cb in self.d_over.items()}}
        tools.update({name: getattr(self, name).value for name in TOOL_SETTINGS + ("s_name",)})
        state["tools"] = tools
        state["selected"] = self.selected
        return state

    def link_state(self) -> dict:
        """What a link carries: everything autosave_state() has; the 3D page's
        display settings are added when the page reports them (make_link)."""
        return self.autosave_state()

    def make_link(self, _b=None, display: dict | None = None) -> dict:
        """Show the links to the current state; returns {name: url}.

        Python cannot read the 3D page's display settings, so the click shows
        the links at once without them and asks the page to keep the state as
        a "link" snapshot: the page stores it with its display and reports it
        back (_apply_memory_report), and the links are rebuilt with it."""
        state = self.link_state()
        if display is None and self._restore_done:
            self._link_pending = json.loads(json.dumps(state))   # as it comes back from the page
            self._memory_seq += 1
            self._memory_ops.append({"op": "add", "kind": "link", "seq": self._memory_seq,
                                     "name": "link", "state": state})
            self.push_view()
        return self._show_links(dict(state, display=display) if display else state, display is not None)

    def _show_links(self, state: dict, with_display: bool) -> dict:
        links = state_links(encode_link_state(state), self.page_address)
        size = max(len(u) for u in links.values())
        warn = (_note(f" these are long ({size} characters): some servers refuse such addresses; "
                      "a config file always works", "#a60") if size > LINK_LONG else "")
        self.link_html.value = (
            '<div style="font-size:12px">open the app in this state: '
            + " &middot; ".join(f'<a href="{html.escape(u)}" target="_blank">{html.escape(k)}</a>'
                                for k, u in links.items())
            + ' <span style="color:#666">(right-click to copy; the browser one below to paste)</span>'
            + (' <span style="color:#282">with the 3D view\'s settings</span>' if with_display else
               ' <span style="color:#888">(the 3D view\'s settings follow ...)</span>' if self._link_pending
               else "")
            + warn + "</div>")
        self.link_text.value = links["in the browser"]
        self.link_text.layout.display = "flex"
        return links

    def _link_from_address(self, address: str | None) -> tuple[bool, str]:
        """Apply the state a link carries in ``address`` (or, under voila, in the
        kernel's QUERY_STRING): (applied, why not if there was one that failed)."""
        param = state_param(address)
        if param is None and os.environ.get("QUERY_STRING"):
            param = state_param("?" + os.environ["QUERY_STRING"])
        if param is None:
            return False, ""
        try:
            state = decode_link_state(param)
        except Exception as exc:
            return False, f"the link's state could not be read ({exc})"
        self.restore_state(state, what="the link")
        return True, ""

    def _state_from_view(self, change):
        try:
            report = json.loads(change["new"] or "{}")
        except ValueError:
            return
        if report.get("kind") == "memory":
            self._apply_memory_report(report)
            return
        if self._restore_done:                 # the offer comes once per session
            return
        self._restore_done = True
        self._apply_memory_report(report)
        self.page_address = report.get("address")
        link_error = ""
        try:
            linked, link_error = self._link_from_address(self.page_address)
            if not linked and report.get("state"):   # a link wins over what this browser remembers
                self.restore_state(report["state"], report.get("display"), report.get("savedText", ""))
        except Exception as exc:
            self.restore_note.value = _err(f"could not restore the remembered state: {exc}")
            self.restore_box.layout.display = "flex"
        if link_error:
            self.restore_note.value = (_err(link_error) + "<br>" + self.restore_note.value
                                       if self.restore_box.layout.display == "flex" else _err(link_error))
            self.restore_box.layout.display = "flex"
        self.push_view()

    def restore_state(self, state: dict, display: dict | None = None, saved_text: str = "",
                      what: str = "your last state in this browser"):
        """Apply a state from autosave_state() (and the 3D page's display settings)."""
        cfg = dict(state)
        if display:
            cfg["display"] = display
        self._suspend = True
        try:
            self.apply_config(cfg)
        finally:
            self._suspend = False
        self.recompute()
        # the scan's angle list exists only now (held constraints), and choosing an
        # angle resets the range to its default: the angle first, then the range
        tools = state.get("tools") if isinstance(state.get("tools"), dict) else {}
        for name in ("s_name", "s_from", "s_to"):
            if name in tools:
                _set_if_valid(getattr(self, name), tools[name])
        sel = state.get("selected")
        if sel in (SEL_CONFIG, SEL_ZERO) or (isinstance(sel, int) and 0 <= sel < len(self.cols)):
            self.select(sel)
        when = f" (saved {html.escape(saved_text)})" if saved_text else ""
        extra = "; ".join(self._config_notes)
        self.restore_note.value = _note(f"continued from {html.escape(what)}{when}"
                                        f"{'; ' + html.escape(extra) if extra else ''}; earlier states: "
                                        "<i>config</i> tab, <i>in this browser</i>", "#282")
        self.restore_box.layout.display = "flex"

    def _start_fresh(self, _b=None):
        self._forget = True
        self.restore_note.value = _note("starting fresh (the current state is kept as a snapshot), "
                                        "the page reloads ...", "#a60")
        self.restore_box.layout.display = "flex"
        self.push_view()

    # ---------------------------------------------------------- snapshots
    # Requests go to the 3D page as numbered ops in every push until its next
    # report acknowledges them ("ack"); the report carries the whole list.

    def _set_memory_enabled(self, on: bool):
        for wdg in (self.remember, self.snap_name, self.snap_btn, self.snap_list, self.snap_restore,
                    self.snap_delete, self.fresh_btn2, self.forget_btn):
            wdg.disabled = not on

    def _memory_op(self, **op):
        self._memory_seq += 1
        self._memory_ops.append(dict(op, seq=self._memory_seq))
        self.push_view()

    def _apply_memory_report(self, report: dict):
        ack = report.get("ack", 0)
        self._memory_ops = [op for op in self._memory_ops if op["seq"] > ack]
        snaps = report.get("snapshots")
        if isinstance(snaps, list):
            self.snapshots = [s for s in snaps if isinstance(s, dict) and s.get("id")]
        self._memory_loading = True
        try:
            if isinstance(report.get("remember"), bool):
                self.remember.value = report["remember"]
            keep = self.snap_list.value
            options = [(self._snapshot_label(s), s["id"]) for s in reversed(self.snapshots)]
            self.snap_list.options = options or [("(no snapshots yet)", "")]
            ids = [v for _, v in options]
            self.snap_list.value = keep if keep in ids else (ids[0] if ids else "")
        finally:
            self._memory_loading = False
        if self._link_pending is not None:    # the link's snapshot is back, with the page's display
            snap = next((x for x in reversed(self.snapshots) if x.get("kind") == "link"
                         and x.get("state") == self._link_pending and isinstance(x.get("display"), dict)), None)
            if snap is not None:
                self._link_pending = None
                self._show_links(dict(snap["state"], display=snap["display"]), True)
        if self.snap_btn.disabled:            # the first report: the controls work now
            self._set_memory_enabled(True)
            self.memory_msg.value = ""
        self._show_snapshot_link()

    @staticmethod
    def _snapshot_label(snap: dict) -> str:
        name = str(snap.get("name") or "snapshot")
        if snap.get("auto"):
            return "auto: " + name                     # these names carry their date already
        when = snap.get("savedText")                   # the browser's local time
        if not when and isinstance(snap.get("saved"), (int, float)):
            when = datetime.fromtimestamp(snap["saved"] / 1000).strftime("%Y-%m-%d %H:%M")
        prefix = {"config": "config: ", "link": "link: "}.get(snap.get("kind"), "★ ")
        return prefix + name + (f" · {when}" if when else "")

    def _snapshot(self, sid=None) -> dict | None:
        sid = self.snap_list.value if sid is None else sid
        return next((s for s in self.snapshots if s.get("id") == sid), None)

    def _show_snapshot_link(self, _change=None):
        snap = self._snapshot()
        if snap is None or not isinstance(snap.get("state"), dict):
            self.snap_download.value = ""
            return
        cfg = dict(snap["state"])
        if snap.get("display"):
            cfg["display"] = snap["display"]
        blob = base64.b64encode(json.dumps(cfg, indent=2).encode()).decode()
        stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(snap.get("name", ""))).strip("._")
        name = f"bernina_diffraction_{stem or 'snapshot'}.json"
        self.snap_download.value = (f'<a download="{html.escape(name)}" href="data:application/json;base64,{blob}" '
                                    f'style="font-size:12px">&#11015; download this snapshot as a config file</a>')

    def _on_remember(self, change):
        if self._memory_loading or not self._restore_done:
            return
        self.memory_msg.value = (_note("the state is remembered again", "#282") if change["new"] else
                                 _note("nothing is remembered any more and the stored state is deleted; "
                                       "snapshots stay until you delete them"))
        self.push_view()

    def _take_snapshot(self, _b=None):
        name = self.snap_name.value.strip()        # none: the page names it by its own clock
        self._memory_op(op="add", name=name, state=self.autosave_state())
        self.snap_name.value = ""
        self.memory_msg.value = _note(f"snapshot {('<b>' + html.escape(name) + '</b> ') if name else ''}"
                                      "stored in this browser", "#282")

    def _restore_snapshot(self, _b=None):
        snap = self._snapshot()
        if snap is None or not isinstance(snap.get("state"), dict):
            return
        self._memory_seq += 1                 # the state being left, as an automatic snapshot
        self._memory_ops.append({"op": "add", "auto": True, "seq": self._memory_seq,
                                 "name": f"before restoring {snap.get('name', 'a snapshot')}",
                                 "state": self.autosave_state()})
        try:
            self.restore_state(snap["state"], snap.get("display"), what=f"the snapshot {snap.get('name', '')}")
            self.memory_msg.value = _note(f"restored <b>{html.escape(str(snap.get('name', '')))}</b>; the state "
                                          "before is an automatic snapshot", "#282")
        except Exception as exc:
            self.memory_msg.value = _err(f"could not restore this snapshot: {exc}")
        self.push_view()

    def _delete_snapshot(self, _b=None):
        snap = self._snapshot()
        if snap is not None:
            self._memory_op(op="delete", id=snap["id"])
            self.memory_msg.value = _note(f"deleted {html.escape(str(snap.get('name', '')))}", "#a60")

    def _forget_everything(self, _b=None):
        if self.forget_btn.description != "really forget?":
            self.forget_btn.description = "really forget?"
            self.forget_btn.button_style = "danger"
            self.memory_msg.value = _note("click again to delete the remembered state and all snapshots")
            return
        self.forget_btn.description, self.forget_btn.button_style = "forget everything", ""
        self._memory_op(op="clear")
        self.memory_msg.value = _note("everything this app kept in this browser is deleted", "#a60")

    def _on_make(self, _b=None):
        """Write the config file and download it at once: the link carries the
        app's part, the 3D page adds its display block (unless left out) and
        clicks it (``download`` in push_view). The same state goes into this
        browser's history as a "config" snapshot with its date."""
        cfg = self.current_config()
        with_display = self.include["display"].value
        blob = base64.b64encode(json.dumps(cfg, indent=2).encode()).decode()
        stem = self.file_name.value.strip()
        stem = stem[:-5] if stem.lower().endswith(".json") else stem
        stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem).strip("._")
        name = (stem or "bernina_diffraction_config") + ".json"
        self._downloads += 1
        self.download_link.value = (
            f'<a download="{name}" id="{VIEW_ID}-config-download" data-cfg="{blob}" '
            f'data-display="{1 if with_display else 0}" data-n="{self._downloads}" '
            f'href="data:application/json;base64,{blob}" style="font-size:12px">'
            f'&#11015; {name}</a> <span style="font-size:11px;color:#666">downloading; if nothing '
            'arrived, click the name</span>')
        self._memory_seq += 1
        self._memory_ops.append({"op": "add", "kind": "config", "seq": self._memory_seq,
                                 "name": name, "state": self.autosave_state()})
        self.push_view()

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
        if "sample" in cfg:                       # a sample without "structure" had no CIF
            self._set_structure(cfg.get("structure"), notes)
            applied.append("structure")
        if "energy" in cfg:
            self.ctrl["energy"].value = cfg["energy"]
            applied.append("energy")
        for name, value in (cfg.get("angles") or {}).items():
            name = legacy(name)
            if name in self.ctrl and name not in dif.BOOL_CONSTRAINTS and not isinstance(value, bool):
                self.ctrl[name].value = float(value)
                applied.append(name)
        if isinstance(cfg.get("fixed"), dict) and "energy" not in cfg["fixed"]:
            self.ctrl["energy_fixed"].value = True     # files from before held it, always
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
        tools = cfg.get("tools")
        if isinstance(tools, dict):
            if tools.get("detector") in dif.DETECTOR_PRESETS:
                self.d_preset.value = tools["detector"]
            for key, wid in (("width", self.d_w), ("height", self.d_h), ("distance", self.d_dist)):
                if isinstance(tools.get(key), (int, float)):
                    wid.value = tools[key]
            for name in TOOL_SETTINGS:
                if name in tools:
                    _set_if_valid(getattr(self, name), tools[name])
            for k, on in (tools.get("overlays") or {}).items():
                if k in self.d_over and isinstance(on, bool):
                    self.d_over[k].value = on
            if isinstance(tools.get("tab"), int) and 0 <= tools["tab"] < len(self.tool_tabs.children):
                self.tool_tabs.selected_index = tools["tab"]
            applied.append("tools")
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
        with _working(self.t_out, self.t_btn, self.t_progress, "solving the reflections ...") as step:
            self._run_table(step)

    def _run_table(self, step):
        self.t_out.clear_output()
        if not self._need_sample(self.t_out):
            return
        try:
            sg = self.space_group()
            mode = self.t_group.value
            if self.t_source.value == "all":
                # grouped: every member of a family, else one per family
                lst = cry.reflection_list(self.cell(), self.energy(), sg, self.structure, d_min=self.t_dmin.value,
                                          max_rows=self.t_max.value, unique=not mode, allowed_only=True)
                hkls = [tuple(r) for r in lst[["h", "k", "l"]].itertuples(index=False)] if len(lst) else []
            else:
                hkls = [dif.parse_vector(ln) for ln in self.t_hkls.value.splitlines() if ln.strip()]
            if mode == "sym" and sg is None and self.structure is None:
                raise ValueError("symmetry-equivalent needs a space group (sample tab)")
            groups = self._table_groups(hkls, mode, sg)
        except Exception as exc:
            self.results["table"] = exc
            with self.t_out:
                print(f"error: {exc}")
            return
        df = dif.reflection_table(self.sample, self.constraints(), hkls, self.energy(), self.ranges(),
                                  self.t_sol.value, self.ctrl["detector"].value, progress=step)
        extra = cry.crystal_columns(hkls, sg, self.structure, self.energy())
        parts = ([groups] if groups is not None else []) + [df.reset_index(drop=True), extra]
        table = pd.concat(parts, axis=1)
        table.index = df.index
        if groups is not None:
            table = table.reset_index().sort_values("group", kind="stable")
            if self.t_one.value:
                angle = next((c for c in table.columns if str(c).startswith("Δ")), None)
                table = cry.collapse_groups(table, angle=angle)
            table = table.set_index(["h", "k", "l"])
        self.results["table"] = table
        _show_df(self.t_out, table)

    def _table_groups(self, hkls, mode, sg):
        """The grouping columns of the angles table, or None."""
        if not mode or not hkls:
            return None
        if mode == "sym":
            g = cry.equivalence_groups(hkls, sg or self.structure.space_group)
            return g[["group", "family"]]
        ub = self.sample.UB
        if mode == "phi":
            axis = np.asarray(dif.GEOMETRY.circle("phi").axis, float)
            label = "Δphi [deg]"
        elif mode == "surface":
            if self.sample.n_surface is None:
                raise ValueError("grouping about the surface normal needs one (sample tab)")
            axis, label = np.asarray(self.sample.n_surface, float), "Δ about n [deg]"
        else:
            axis, label = self._direction(self.t_dir.value), f"Δ about {self.t_dir.value.strip()} [deg]"
        g = cry.rotation_groups(hkls, ub, axis)
        return g[["group", "rotation", "Q_par", "Q_perp"]].rename(
            columns={"rotation": label, "Q_par": "Q∥ [1/Å]", "Q_perp": "Q⊥ [1/Å]"})

    def _direction(self, text):
        """'[u v w]' (real space) or '(h k l)' (plane normal) -> vector in the UB frame."""
        t = text.strip()
        if not t or t[0] not in "[(" or t[-1] not in "])":
            raise ValueError(f"write the direction as [u v w] (real space) or (h k l) (plane normal), not {text!r}")
        v = dif.parse_vector(t[1:-1])
        if t[0] == "[":
            return cry.real_direction_in_reciprocal_frame(self.sample.UB, v)
        return self.sample.UB @ np.asarray(v, float)

    def run_reachable(self, _b=None):
        with _working(self.r_out, self.r_btn, self.r_progress, "solving the reflections ...") as step:
            self._run_reachable(step)

    def _run_reachable(self, step):
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
                                           aperture, self.ctrl["detector"].value, progress=step)
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
        with _working(self.d_out, self.d_btn):
            self._run_detector()

    def _run_detector(self):
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
        with _working(self.c_out, self.c_btn):
            self._run_crystal()

    def _run_crystal(self):
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
        with _working(self.g_out, self.g_btn):
            self._run_grazing()

    def _run_grazing(self):
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
        self.g_out.figure(fig)
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

    # ================================================================ scan (angles tab)

    SCAN_RANGES = {"alpha": (-10.0, 90.0), "beta": (-10.0, 90.0), "betain": (-5.0, 45.0),
                   "betaout": (-5.0, 90.0)}

    def _refresh_scan_names(self):
        """The scan dropdown lists the held numeric constraints."""
        held = [n for n in dif.CIRCLES + dif.VIRTUAL_CONSTRAINTS if self.ctrl[f"{n}_fixed"].value]
        options = [(n, n) for n in held] or [("(hold a value first)", "")]
        if list(self.s_name.options) != options:
            keep = self.s_name.value
            self.s_name.options = options
            self.s_name.value = keep if keep in held else options[0][1]

    def _scan_defaults(self, change):
        lo, hi = self.SCAN_RANGES.get(change["new"], (-180.0, 180.0))
        self.s_from.value, self.s_to.value = lo, hi

    def run_scan(self, _b=None):
        """Solve with the chosen held value stepped through the range; plot the
        number of solutions passing the min/max (hklforge Diffractometer.scan)."""
        name = self.s_name.value
        self.s_msg.value = ""
        if not name:
            self.s_msg.value = _note("hold an angle (fix) to scan it")
            return
        if self.sample is None:
            self.s_msg.value = _err("no valid sample yet (sample tab)")
            return
        s = self.sample
        try:
            mode = dif.make_mode(self.constraints())
            if any(k in mode for k in dif.SURFACE_CONSTRAINTS) and s.surface is None:
                raise ValueError("betain, betaout and bin_eq_bout need a surface normal (sample tab)")
            detector = self.ctrl["detector"].value if "h2" in mode else "h1"
            d = Diffractometer(dif.GEOMETRY, s.UB, energy=self.energy(), surface=s.n_surface,
                               reference=s.n_reference)
            ranges = self.ranges()

            def keep(pos):
                if not ranges:
                    return True
                return dif.passes({**pos, **virtual_angles(dif.GEOMETRY, pos, s.n_surface, s.n_reference)},
                                  ranges)

            values = np.linspace(float(self.s_from.value), float(self.s_to.value), int(self.s_steps.value))
            self.s_btn.disabled = True
            try:
                res = d.scan(tuple(self.hkl()), mode, name, values, detector, keep=keep)
            finally:
                self.s_btn.disabled = False
        except Exception as exc:
            self.s_msg.value = _err(exc)
            self.results["scan"] = exc
            return
        counts = np.array([len(r["solutions"]) for r in res])
        errors = sum(r["error"] is not None for r in res)
        self._scan = (name, values, counts)
        self.results["scan"] = pd.DataFrame({name: values, "solutions": counts})
        self.draw_scan()
        self.s_msg.value = _note(f"{int((counts > 0).sum())} of {len(values)} values have solutions"
                                 + (f"; {errors} with a continuum or no valid mode" if errors else "")
                                 + "; click in the plot to set " + html.escape(name), "#555")

    def draw_scan(self):
        if self._scan is None:
            return
        name, values, counts = self._scan
        fig = self.s_fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.step(values, counts, where="mid", color="#2166ac")
        ax.fill_between(values, counts, step="mid", alpha=0.15, color="#2166ac")
        ax.axvline(float(self.ctrl[name].value), color="#d62728", lw=1)
        ax.set_xlabel(f"{name} [deg]")
        ax.set_ylabel("solutions")
        ax.set_ylim(0, max(1, int(counts.max())) + 0.5)
        ax.set_yticks(range(0, max(1, int(counts.max())) + 1))
        ax.grid(alpha=0.3)
        self.s_figure.redraw()

    def _scan_click(self, event):
        if self._scan is None or event.inaxes is None or event.xdata is None:
            return
        name = self._scan[0]
        if not self.ctrl[f"{name}_fixed"].value:
            return
        self.ctrl[name].value = round(float(event.xdata), 2)     # recompute follows
        self.draw_scan()
