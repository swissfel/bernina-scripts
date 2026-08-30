# bernina-scripts

Notebooks and small ipywidgets panels for the Bernina beamline, published on
mybinder.org so people can use them without installing anything.

## Two environments, on purpose

- `binder/environment.yml` - what mybinder builds. Deliberately minimal.
- `environment.yml` (repo root) - the full beamline/analysis environment (PSI
  channels, `jungfrau_utils`, `bsread`, `escape-fel`, the `hdf5` pin). **Binder
  never uses this**: repo2docker prefers config in `binder/` over the repo root.

Keep them separate. The root file does not currently solve on its own
(`hdf5=1.10.4` + `python=3.12` + `h5py` is unsatisfiable), which is survivable
only because Binder ignores it.

### Rules for `binder/environment.yml`

repo2docker installs it **into the same conda env that runs Binder's Jupyter
server** (true for any python >= 3.7). So anything here that makes conda
downgrade the server stack silently breaks the widget frontend, and widget
output then renders as an empty area with no error.

- Do **not** add `jupyter`, `jupyterlab`, `notebook`, or `nodejs` - Binder
  provides them.
- Do **not** ever add `jupyter_contrib_nbextensions`. It requires `notebook <7`
  and drags notebook 7.4 -> 6.5, ipython 9 -> 8.2, traitlets back with it.
- Keep `ipywidgets>=8.1,<9` pinned. Binder's frontend ships
  `jupyterlab_widgets` 3.x / `widgetsnbextension` 4.x, which only speak the
  ipywidgets 8 model protocol; an unpinned solve can land on 7.x and then
  nothing renders.

Binder's own stack, for reference: ipywidgets 8.1.7, jupyterlab_widgets 3.0.15,
widgetsnbextension 4.0.14, jupyterlab 4.4.9, notebook 7.4.7, python 3.12.

## Writing a panel that survives voila

A panel can render perfectly in JupyterLab and show **nothing** under voila.
voila renders through its own bundled widget manager
(`@voila-dashboards/widgets-manager8`), not the lab one, and it hides errors by
default. Rules learned the hard way on `applet_sim_diff_bernina.ipynb`:

- **No percentage sizes on containers.** `Layout(max_width='50%')` resolves
  against a container that is not always definite outside a JupyterLab cell; it
  can collapse the whole column to zero width. Use fixed `px`.
- **Stick to core widgets.** `TagsInput` is ipywidgets-8-only and is not
  rendered by every widget manager. If one child view fails to instantiate, the
  entire enclosing `VBox` comes up blank while a sibling `Output` still renders
  - which looks exactly like a broken frontend. A `Textarea` did the same job.
- **Give an `Output` an explicit flex basis** if you want it beside something.
  Its content is `width: 100%`, so with the default `flex-basis: auto` it claims
  the whole row and wraps underneath. Use `flex='1 1 380px'`, not `'1 1 auto'`.
- **Build controls explicitly**, in a dict fed to `widgets.interactive_output`,
  rather than slicing `widgets.interactive(...).children[12:38]`. The slicing
  was correct but silently depends on argument order.
- **Avoid LaTeX in `description=`.** It needs MathJax; plain unicode (α, β, γ)
  is one less dependency.
- `Layout(positioning=...)` is not a real trait. It is silently swallowed and
  does nothing.

Also: ship notebooks **with outputs cleared**. A saved
`application/vnd.jupyter.widget-view+json` output without matching
`metadata.widgets` state renders as a blank area for anyone who opens the
notebook without running it.

`binder/postBuild` turns on `VoilaConfiguration.show_tracebacks` (off by
default) so a failing panel shows its traceback in the app instead of a blank
page. Leave that in place.

## Testing locally

Build the Binder environment exactly, from repo2docker's own lockfile plus our
file - the two-step is what repo2docker does, and it is what catches a
dependency that would downgrade the server stack:

```bash
curl -sO https://raw.githubusercontent.com/jupyterhub/repo2docker/main/repo2docker/buildpacks/conda/environment.py-3.12-linux-64.lock
micromamba create -n binderlike -y --file environment.py-3.12-linux-64.lock
micromamba install -n binderlike -y -f binder/environment.yml   # expect NO downgrades
```

Then serve the repo and check **both** views - lab working proves nothing about
voila:

```bash
micromamba run -n binderlike jupyter lab --no-browser --port=8899
# lab:   http://127.0.0.1:8899/lab
# voila: http://127.0.0.1:8899/voila/render/prep_notebooks/<notebook>.ipynb
```

Caveats worth knowing before you trust a test:

- `curl` on a voila URL tells you nothing. The page is a shell; widgets arrive
  later over a websocket. Same for `google-chrome --headless --dump-dom`
  with `--virtual-time-budget` - virtual time does not wait for websocket
  traffic, so it reports an empty page even when the panel is fine. Use a real
  browser, or drive Chrome over the DevTools protocol and wait in real time.
- A local voila render is **not** proof. The failure that motivated all of the
  above did not reproduce locally even in a package-identical environment. The
  rules above are more reliable than any local test; mybinder itself is the
  final check.
