# bernina-scripts

Notebooks and scripts to help users prepare for experiments at the Bernina
endstation (SwissFEL).

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/swissfel/bernina-scripts/HEAD?urlpath=voila/tree/apps)

The badge launches Voila's tree view of `apps/` as a start page — pick a tool
and it runs live, no local install needed. Individual apps can also be opened
directly:

| app | JupyterLab | app view (voila) |
| --- | --- | --- |
| diffraction reflections | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/swissfel/bernina-scripts/HEAD?labpath=apps%2Fdiffraction_reflections.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/swissfel/bernina-scripts/HEAD?urlpath=voila%2Frender%2Fapps%2Fdiffraction_reflections.ipynb) |
| 3D diffractometer | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/swissfel/bernina-scripts/HEAD?labpath=apps%2Fdiffractometer_3d.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/swissfel/bernina-scripts/HEAD?urlpath=voila%2Frender%2Fapps%2Fdiffractometer_3d.ipynb) |
| beam size calculator | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/swissfel/bernina-scripts/HEAD?labpath=apps%2Fbeam_size_calculator.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/swissfel/bernina-scripts/HEAD?urlpath=voila%2Frender%2Fapps%2Fbeam_size_calculator.ipynb) |
| pump-probe time resolution | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/swissfel/bernina-scripts/HEAD?labpath=apps%2Ftime_resolution.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/swissfel/bernina-scripts/HEAD?urlpath=voila%2Frender%2Fapps%2Ftime_resolution.ipynb) |

Notebooks are shipped without stored outputs, so in JupyterLab use
*Run > Restart Kernel and Run All Cells* to build the input fields. The voila
links run the notebook for you and show only the widgets.

## Layout

- **`apps/`** — standalone interactive tools, meant to run on Binder or locally
  with `voila`/Jupyter. Kept thin: each notebook is just widgets + display
  calls, importing its actual calculations from `bernina_prep/`, so they build
  fast and stay easy to diff.
  - `diffraction_reflections.ipynb` — diffractometer angles for a reflection
    (or a whole table of reflections) using diffcalc's `you` UB geometry,
    with a live 3D view of the selected solution (k, k′, Q, reciprocal cell,
    Ewald sphere, detector cut), beamline presets, solution filters and
    config files to store/recall a setup. Lab directions are entered in
    **SwissFEL axes** (x horizontal transverse, y up, z along the beam).
    Note: this beamline calls the out-of-plane detector circle **gamma**;
    diffcalc itself calls it `nu` — the apps/library translate automatically.
  - `beam_size_calculator.ipynb` — infers focus waist/Rayleigh range from a
    measured FWHM spot size and plots the beam envelope against named
    beamline components (`data/beamline.csv`).
  - `diffractometer_3d.ipynb` — animated 3D model of the six-circle `you`
    geometry (mu/eta/chi/phi/delta/gamma), cross-checked against diffcalc's
    own reference diagram; paste a solved reflection's angles to see the
    mechanism move, orbit/zoom with the mouse. A self-contained three.js
    page (`diffractometer_3d.html`, offline-capable, no CDN) embedded via
    iframe so it runs the same under Voila as standalone.
  - `time_resolution.ipynb` — pump-probe absorbed fluence and the geometric
    (footprint-driven) timing smear at grazing incidence, ported from
    [`calx.pumpprobe`](https://github.com/htlemke/calx) (pure numpy/scipy,
    reimplemented directly in `bernina_prep.timing` rather than depending
    on the full `calx` package).
- **`bernina_prep/`** — the shared, non-widget calculation code behind the
  apps above (`diffraction.py`, `beamsize.py`, `timing.py`). Installed in
  editable mode by `binder/postBuild`, so it's importable from any notebook
  in the repo.
- **`prep_notebooks/`** — richer, narrative walkthroughs meant for local use
  (they may depend on packages, data, or facility access not available on
  Binder).
- **`data/`** — small reference datasets used by the apps (e.g. beamline
  component positions).
- **`binder/`** — the Binder build. See below.

## Running locally

At PSI, use the `bpy312` pixi environment in `/mnt/app/python`
(`pixi run -e bpy312 jupyter lab`). Elsewhere, `pixi.toml` in this repo mirrors
the Binder environment for fast local iteration:

```
pixi run voila          # serve apps/ the way Binder does
pixi run lab            # JupyterLab, for editing
```

Or build the Binder environment itself with conda/mamba:

```
mamba env create -n bernina-prep -f binder/environment.yml
conda activate bernina-prep
pip install --no-deps -e .
voila apps
```

## Running on Binder

Binder builds **only** from [`binder/`](binder/) — repo2docker prefers that
directory and ignores any config at the repo root, so `pixi.toml` here is a
local convenience and is not what Binder uses.

Read the comments in [`binder/environment.yml`](binder/environment.yml) and
[`CLAUDE.md`](CLAUDE.md) before changing dependencies or writing a new panel.
Several constructs render fine in JupyterLab and show up completely blank under
voila, and the widget stack is deliberately constrained.
