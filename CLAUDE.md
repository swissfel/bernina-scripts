# bernina-scripts

Interactive prep tools for the Bernina endstation (SwissFEL), published on
mybinder.org so users need no local install.

`apps/` holds the thin widget notebooks (calculations live in `bernina_prep/`).
`apps/dev/` holds work in progress — same rules apply, it is just not linked
from the README yet.

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

`apps/dev/diffractometer_3d.html` is a self-contained three.js page embedded via
`<iframe srcdoc>`. A `srcdoc` iframe inherits the parent document's origin, so
the page can simply **read a hidden element the notebook writes into**
(`#diff3d-payload`, polled every 150 ms) — no custom widget, no `postMessage`
plumbing, and the page stays fully usable standalone because the read just
fails silently. Every push includes a changing counter so repeats are ignored.

Clicking a pandas DataFrame's rows is browser text selection and **cannot**
reach python. To make solutions selectable, the table is rendered as HTML with
a row of `Button`s acting as column headers.

## Diffraction conventions (bernina_prep.diffraction)

- The beamline calls the out-of-plane detector circle **gamma**; diffcalc calls
  it `nu`. The library translates at its boundary — callers only see `gam`/`gamma`.
- **Lab axes: the apps speak SwissFEL, diffcalc speaks its own.** Same
  right-handed frame, cyclically relabelled:

  | | x | y | z |
  |---|---|---|---|
  | diffcalc | +x | +y | +z |
  | SwissFEL | +y | +z | +x |

  So diffcalc x is SwissFEL y (vertical), diffcalc y is SwissFEL z (beam), and
  diffcalc z is SwissFEL x (horizontal transverse). `swissfel_to_diffcalc()` /
  `diffcalc_to_swissfel()` in `bernina_prep.diffraction` do the mapping; it is a
  pure rotation (determinant +1), so nothing about handedness changes.

  Convert **lab-frame directions only**. An `(h,k,l)` is a reciprocal-lattice
  index and carries no lab frame, so it is never converted — which is why the
  surface normal, entered as `(h,k,l)`, is frame-independent while the xyz half
  of an orientation line is not.

  `build_ubcalc(..., frame=...)` takes `"swissfel"` or `"diffcalc"`. The default
  is **`"diffcalc"`** so old callers keep their exact meaning; `apps/dev`
  passes `frame="swissfel"` and the stable `apps/diffraction_reflections.ipynb`
  still uses diffcalc axes. Whenever an app is converted, translate its default
  orientations too rather than leaving the numbers and silently changing what
  they mean — the check is that UB comes out identical.

  The 3D scene is built on diffcalc axes with only the on-screen axis arrows
  and labels relabelled to SwissFEL, which is why it already looked right
  before the inputs were converted.
- **diffcalc-core's `UB` uses the 2π convention**: `|a*| = 2π/a` and
  `|UB @ (h,k,l)| = 2π/d`. Anything drawn alongside it (k, k′, Q) must use
  `|k| = 2π/λ` or Q will not land on a reciprocal lattice node. `k_magnitude()`
  exists so this lives in one place.
- `en2lam` is inlined (same expression *and operation order* as
  xrayutilities', so results are bit-identical) to keep xrayutilities out of
  the Binder image. Operation order matters: the naive reordering differs by
  1 ULP at 20 keV.
- The 3D scene's rotation chains were checked numerically against diffcalc, not
  assumed: sample `Rx(mu)·Rz(-eta)·Ry(chi)·Rz(-phi)` equals `MU@ETA@CHI@PHI`,
  and detector gamma-outer/delta-inner equals `NU@DELTA`, over random
  orientations and every solved branch. Re-run those checks if the scene's
  nesting or `ANGLE_SIGN` is touched.

## Testing

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
- `FileUpload` and `data:` URI download links both work under voila, so config
  save/restore needs no server round trip.
- Flexbox in the embedded page: use `width:100%` not `100vw` (inside an iframe
  `100vw` includes the scrollbar), and a column flex item needs `min-height:0`
  or it refuses to shrink below its content and pushes siblings off-screen.
