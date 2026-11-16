# bernina-scripts

Interactive prep tools for the Bernina endstation (SwissFEL), published on
mybinder.org so users need no local install.

`apps/` holds the thin notebooks: an introduction and a cell that shows an app.
Everything else lives in `bernina_prep/`, where it can be imported and tested:
calculations (`diffraction`, `crystal`, `grazing`, `timing`, `beamsize`,
`thc_angles`) and the widget apps built on them (`reflections_app`, and the
app classes in `thc_angles`/`beamsize`). Work in progress can go to
`apps/dev/` (same rules, not linked from the README).

## Repo layout and environments

- `binder/environment.yml` + `binder/postBuild` — **the only** Binder build.
  repo2docker prefers `binder/` and ignores everything at the repo root, so a
  second spec at the top level is a silently dead file. Do not add one.
- `pixi.toml` — local dev convenience, mirrors the Binder spec. Not used by Binder.
- `bernina_prep` is installed editable **by `binder/postBuild`**, not by a
  `-e .` line in the env file: pip resolves such a path against the working
  directory, which repo2docker does not document. postBuild derives the repo
  root from its own location instead.

### Binder dependency rules

repo2docker installs `binder/environment.yml` **into the same conda env that
runs Binder's Jupyter server** (any python >= 3.7). So anything that makes
conda downgrade the server stack silently breaks widget rendering — output
appears as an empty area, with no error anywhere.

- Never add `jupyter_contrib_nbextensions`: it requires `notebook <7` and drags
  notebook 7.4 → 6.5, ipython 9 → 8.2, traitlets back with it.
- Keep `ipywidgets>=8.1,<9`. Binder ships `jupyterlab_widgets` 3.x /
  `widgetsnbextension` 4.x, which only speak the ipywidgets 8 model protocol;
  an unpinned solve can land on 7.x and nothing renders.
- Do not list `jupyter`/`jupyterlab`/`notebook`/`nodejs` — Binder provides them.
- **Do not pin `voila<0.5.8` and do not remove jupyterlab from the image.**
  That pairing was tried, and the jupyterlab removal was only needed *because*
  of the old voila pin. voila 0.5.12 alongside jupyterlab 4.4.9 is verified
  working on mybinder.

Binder's stack for reference: python 3.12, ipywidgets 8.1.7,
jupyterlab_widgets 3.0.15, widgetsnbextension 4.0.14, jupyterlab 4.4.9,
notebook 7.4.7, voila 0.5.12.

## Writing a panel that survives voila

A panel can render perfectly in JupyterLab and show **nothing** under voila.
voila renders through its own bundled widget manager
(`@voila-dashboards/widgets-manager8`) and hides errors by default. Rules,
each learned from an actual failure here:

- **No percentage sizes on containers.** `Layout(max_width='50%')` resolves
  against a container that is not always definite outside a JupyterLab cell and
  can collapse a whole column to zero width. Use fixed `px`.
- **Core widgets only.** `TagsInput` is not rendered by every widget manager,
  and when one child view fails the entire enclosing `VBox` comes up blank
  while a sibling `Output` still renders — which looks exactly like a broken
  frontend. A `Textarea` did the same job.
- **Give an `Output` an explicit flex basis** to place it beside something. Its
  content is `width:100%`, so with `flex-basis:auto` it claims the whole row and
  wraps underneath. Use `flex='1 1 380px'`, not `'1 1 auto'`.
- **Build controls explicitly** in a dict fed to `interactive_output`, rather
  than slicing `widgets.interactive(...).children[12:38]`.
- **No LaTeX in `description=`** — needs MathJax; plain unicode (α, β, γ) is one
  less dependency.
- `Layout(positioning=...)` is not a real trait; it is silently swallowed.
- **Ship notebooks with outputs cleared.** A saved
  `application/vnd.jupyter.widget-view+json` output with no matching
  `metadata.widgets` state renders as a blank area for anyone who opens the
  notebook without running it.
- Set a float format when displaying a solutions table
  (`pd.option_context('display.float_format', ...)`): mixing ~1e2 angles with
  ~1e-15 numerical zeros makes pandas switch the column to scientific notation.

`binder/postBuild` turns on `VoilaConfiguration.show_tracebacks` (off by
default) so a failing panel shows its error instead of a blank page. Leave it.

## Driving an embedded page from python

The 3D page is hklforge's `hklforge.view` (`view/viewer.html`), a
self-contained three.js page built from any `Geometry`; the app embeds it with
`bernina_view_page()` (Bernina geometry, detector presets, storage prefix
`bernina-reflections` so browsers keep what they remembered) via
`<iframe srcdoc>`. Messages come from `hklforge.view.message` in the lab
(SwissFEL) frame - the page turns vectors into its own frame, so there is no
`to_you()` any more. escape (>= 0.2.21) offers the same as
`diffraction.viewer_page()` / `view_message()`, which eco uses since
2026-11-16; escape's `diffractometer_3d.html` (`VIEWER_HTML`) and eco's copy
are the earlier You-only page, kept only for older eco/escape combinations. A `srcdoc` iframe inherits the parent document's origin, so
the page can simply **read a hidden element the notebook writes into**
(`#diff3d-payload`, polled every 150 ms) — no custom widget, no `postMessage`
plumbing, and the page stays fully usable standalone because the read just
fails silently. Every push includes a changing counter so repeats are ignored.

The reverse direction works the same way: python cannot read the page, so to
save the page's own display settings the page **writes into the parent DOM**.
The config download link carries python's part of the config
(`#diff3d-config-download`, base64 JSON in `data-cfg`, `data-display="0"`
to leave the display out) and the page keeps its `href` rewritten with the
current `display` block; "download" pushes a new `download` number and the
page clicks the link (not on the first payload a reloaded page reads). The
same state goes to the browser history as a snapshot of kind "config". On load, `display` rides on
exactly one push so later pushes do not reset the user's toggles.

Clicking a pandas DataFrame's rows is browser text selection and **cannot**
reach python. To make solutions selectable, the table is rendered as HTML with
a row of `Button`s acting as column headers.

## Where the code lives

The calculations are not in this repo any more (2026-11):

- **hklforge** (`~/mypy/hklforge`, PyPI `hklforge`): the angle solver, and
  `hklforge.crystal` (cells, all space groups with absences and reasons,
  CIF, structure factors, reflection lists and groups), `hklforge.orient`
  (U from orientations), `Geometry.detector_towards`. Tests there compare
  with gemmi and diffcalc.
- **escape** (`~/mypy/escape-fel`, PyPI `escape-fel`): `escape.bernina` -
  the Bernina geometry (`geometry`), `diffraction` (Sample, tables, two
  reflections, detector maps, the 3D page's message), `recspace` (eco's
  RecSpace form), `detectors`, `detector_view`, `otti` (geometry only) and
  the 3D page `diffractometer_3d.html` (`escape.bernina.VIEWER_HTML`).
  Tests in `tests/bernina/` there, including the diffcalc battery
  (`check_engine.py`). `import escape` is lazy, so `escape.bernina` needs no
  dask: Binder (postBuild) and the browser build install escape-fel with
  `--no-deps`.
- here: the apps and widgets, `crystal.py` (names over `hklforge.crystal`),
  `grazing.py`, `timing.py`, `beamsize.py`, `thc_angles.py`, the Otti panel,
  `presets.json`.
- eco uses the same `escape.bernina.recspace` (`eco/utilities/recspace_calc.py`
  re-exports it, with a temporary local fallback until escape is deployed).

A change to a calculation or to the 3D page goes to hklforge/escape; for a
local test install them editable into the pixi envs
(`.pixi/envs/<env>/bin/python -m pip install --no-deps -e ../hklforge -e ../escape-fel`)
and build the browser version with `HKLFORGE=... ESCAPE=... lite/build.sh`.
Binder and the Pages build take the released versions (hklforge >= 0.2.0,
escape-fel >= 0.2.19, the first with escape.bernina).

## Diffraction conventions (escape.bernina.diffraction)

- **One lab frame**, SwissFEL/Bernina: z+ beam, y+ up, x+ = y × z (left,
  looking along the beam). The diffcalc version of the app relabelled to
  diffcalc's axes internally; there is no such second frame any more.
  `GEOMETRY` is hklforge's builtin `psic` turned into this frame.
- The out-of-plane detector circle is **gamma** (diffcalc: `nu`).
  `LEGACY_NAMES` maps `gam`/`nu` when reading old configs; `mu_is_gam` is
  not supported by hklforge and is dropped with a note.
- **Surface normal and reference are separate** (h,k,l) fields: the surface
  normal drives betain/betaout/bin_eq_bout, the reference (default (1 0 0),
  as eco and diffcalc) alpha/beta/psi/naz. The diffcalc version set both
  from the surface-normal box, so `apply_config` copies the normal into the
  reference when it loads a version-1 file. The surface normal may be
  empty; then betain etc. raise a clear error.
- **UB** in the 2π convention (`|UB h| = 2π/d`, `|k| = 2π/λ`), from the
  first two orientations (Busing & Levy, the first exact) or one (smallest
  rotation from U = 1). Orientations are at all angles 0; lab directions
  may be words (`beam`, `up`, `x+`).
- The 3D page (hklforge.view) builds its circles from `dif.GEOMETRY` and
  takes UB, surface normal (`nPhi`) and reference (`refPhi`) in the lab
  frame. Virtual angles, Brillouin-zone special points (`hklforge.crystal.bz`)
  and circle axes are drawn by the page; the geometry tab (tools) is
  hklforge's `GeometryExplorer`, built when first opened (second page).
- **Degenerate** modes (a continuum, e.g. Q along the phi axis with phi
  free) come back from hklforge as `Degenerate`; `Sample.solve` turns that
  into a ValueError the app shows. diffcalc returns arbitrary members of
  the continuum there, so those cases are excluded from the comparison.
- **Filters**: a circle's min/max with both bounds may match at ±360 (as
  eco's solution ranges); one-sided bounds and virtual angles compare as is.
- **THC apertures**: `thc_angles.aperture_check` uses the envelopes'
  (gamma, delta) as the diffractometer's own gamma/delta (the THC app says
  so), with (gamma, delta) ≡ (gamma + 180, 180 − delta). Otti is left out:
  its precomputed envelope is a single curve without a stated inside.
- **xraydb is optional**: without it there are no structure factors and
  grazing incidence needs δ and β typed in.
- **Browser memory**: the 3D page keeps the app's state, snapshots and the
  remember switch in localStorage (`bernina-reflections-state`,
  `-snapshots`, `-remember`); the app talks to it through hidden text
  widgets and numbered `msg.memory.ops` (see the comments in the page and
  in `reflections_app.py`).

## Testing

Unit tests (no browser) — the explicit interpreter and PYTHONPATH matter at
PSI, where `bernina_prep` is not installed:

```bash
PYTHONPATH=$PWD /mnt/app/python/.pixi/envs/bpy312/bin/python -m pytest tests -q
```

Simpler: the pixi test env (`.pixi/envs/test/bin/python -m pytest tests`),
with hklforge and escape installed as above.

Under voila, Lumino tab bars ignore synthetic `.click()`; switch tabs with
real mouse events (CDP `Input.dispatchMouseEvent`). A button inside a hidden
tab still fires, but its output is not in `innerText` until the tab shows.


Build the Binder environment the way repo2docker does — the two-step is what
catches a dependency that would downgrade the server stack:

```bash
curl -sO https://raw.githubusercontent.com/jupyterhub/repo2docker/main/repo2docker/buildpacks/conda/environment.py-3.12-linux-64.lock
micromamba create -n binderlike -y --file environment.py-3.12-linux-64.lock
micromamba install -n binderlike -y -f binder/environment.yml   # expect NO downgrades
micromamba run -n binderlike pip install --no-deps -e .
```

Then serve and check **both** views — lab working proves nothing about voila:

```bash
micromamba run -n binderlike jupyter lab --no-browser --port=8899
# lab:   http://127.0.0.1:8899/lab
# voila: http://127.0.0.1:8899/voila/render/apps/<notebook>.ipynb
```

What is and is not trustworthy:

- `curl` on a voila URL tells you **nothing** — the page is a shell and widgets
  arrive later over a websocket. Same for `chrome --headless --dump-dom` with
  `--virtual-time-budget`: virtual time does not wait for websocket traffic, so
  it reports an empty page even when the panel is fine.
- What works: drive Chrome over the DevTools protocol (`--remote-debugging-port`),
  navigate, wait in **real** time, then evaluate JS against the live DOM. That
  can also click widget buttons and read the iframe's state, so interactions are
  verifiable end to end. The 3D scene exposes `window.__diffApp.debugCounts()`
  for exactly this.
- A local voila render is **not** proof. The original blank-widget failure never
  reproduced locally even in a package-identical environment. The rules above
  are more reliable than any local test; mybinder itself is the final check.
- Heavy background processes (headless Chrome + voila) can make an
  `nbconvert --execute` run appear to hang. Stop them before concluding a
  notebook is broken.

### Notes from the 3D view work

- **Measured hkl** is `UB⁻¹ · R_sample⁻¹ · Q_lab`, computed in the browser from
  the live circle positions, so it updates while dragging. Sanity check: at a
  solved reflection it returns exactly the requested indices; off the Bragg
  condition it goes non-integer. Rotating a circle whose axis is parallel to
  the reflection (e.g. phi when UB·(111) lies along z) correctly leaves it
  unchanged — that is not a bug.
- **Translucent shells** (the Ewald sphere) read as a flat wash with a
  double-sided `MeshBasicMaterial`. Rendering only `THREE.BackSide` with a lit
  material, plus a faint `WireframeGeometry` sphere, is what made it read as a
  balloon. Clip it with `renderer.localClippingEnabled` + six `THREE.Plane`s so
  a sphere far larger than the scene does not take over.
- **Label sprites must be sized to their text.** The original helper drew onto
  a fixed 160x80 canvas, which silently clipped anything longer than a couple
  of characters ("Z (beam)", any Q label carrying hkl). Measure the string,
  size the canvas to it, and scale the sprite by the same aspect so glyph
  height stays constant. Non-power-of-two canvases need
  `minFilter = LinearFilter` and `generateMipmaps = false`.
- **Detector face axes**: at zero angles width is along SwissFEL x and height
  along SwissFEL y. In the scene's local frame that is local Z and local X
  respectively (local Y is the arm), so the box is
  `BoxGeometry(height, thickness, width)` - easy to get backwards.
- **Beamline setups** live in `bernina_prep/presets.json`, in exactly the
  names and format of eco's `CONSTRAINT_PRESETS` (`set_constraints`): held
  constraints plus solution ranges. Keep the two in step. Loading one
  applies only the constraints and min/max, deliberately leaving the sample
  and hkl alone. Applying a whole config writes many widgets, each of which would
  fire its own solve - suspend solving during the batch and run exactly one
  afterwards (`_suspend` in the app).
- `FileUpload` and `data:` URI download links both work under voila, so config
  save/restore needs no server round trip.
- Flexbox in the embedded page: use `width:100%` not `100vw` (inside an iframe
  `100vw` includes the scrollbar), and a column flex item needs `min-height:0`
  or it refuses to shrink below its content and pushes siblings off-screen.
