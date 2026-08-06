# bernina-scripts

Notebooks and scripts to help users prepare for experiments at the Bernina
endstation (SwissFEL).

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/swissfel/bernina-scripts/HEAD?urlpath=voila/tree/apps)

## Layout

- **`apps/`** — standalone interactive tools, meant to run on Binder (via the
  badge above) or locally with `voila`/Jupyter. Kept thin: each notebook is
  just widgets + display calls, importing its actual calculations from
  `bernina_prep/`, so they build fast and stay easy to diff.
  - `diffraction_reflections.ipynb` — diffractometer angles for a reflection
    (or a whole table of reflections) using diffcalc's `you` UB geometry.
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
  editable mode via `environment.yml`, so it's importable from any notebook
  in the repo.
- **`prep_notebooks/`** — richer, narrative walkthroughs meant for local use
  (they may depend on packages, data, or facility access not available on
  Binder).
- **`data/`** — small reference datasets used by the apps (e.g. beamline
  component positions).

## Running locally

At PSI, use the `bpy312` pixi environment in `/mnt/app/python`
(`pixi run -e bpy312 jupyter lab`), which has diffcalc-core, xrayutilities,
ipywidgets and voila. Elsewhere, `environment.yml` at the repo root builds an
equivalent conda/mamba environment:

```
mamba env create -f environment.yml
conda activate bernina312_prep
jupyter lab   # or: voila apps
```

## Running on Binder

The badge above launches straight into Voila's tree view of `apps/`, which
serves as the start page — pick a tool and it runs live, no local install
needed. `environment.yml` is kept deliberately minimal (just what `apps/`
needs) so Binder builds stay fast; `prep_notebooks/` may need extra
dependencies installed locally.
