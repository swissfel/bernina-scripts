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
| THC accessible angles | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/swissfel/bernina-scripts/HEAD?labpath=apps%2Fthc_angles.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/swissfel/bernina-scripts/HEAD?urlpath=voila%2Frender%2Fapps%2Fthc_angles.ipynb) |
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
    on Bernina's six-circle (You) diffractometer, solved with
    [hklforge](https://github.com/swissfel/hklforge) exactly as eco does on
    the beamline. Live 3D view of the selected solution (k, k′, Q,
    reciprocal cell, Ewald sphere, detector cut, the cells and Brillouin
    zones around the reflections and on the detector), Bernina setups (the same
    as eco's `set_constraints`), solution min/max filters, a second
    reflection held in Bragg condition, any set of angles as a "config" pose
    with the hkl it puts in Bragg condition, and config files. Tools: a
    table of reflections; which allowed reflections a setup reaches,
    optionally through a THC aperture; a detector view (the face coloured
    by h, k, l, |Q|, 2θ, Brillouin zone, distance from the nearest lattice
    point, a diffuse-scattering estimate, exit angle and Q along/across the
    surface, polarization or solid angle, with overlays for integer hkl,
    cell and Brillouin-zone boundaries, Bragg spots, powder rings, the
    direct beam, sample horizon and Yoneda line, and truncation-rod
    crossings; hover or click for the values at a point; Bernina's
    JF01T03 and JF07T32 with their module gaps); the allowed
    reflections of a space group or CIF with structure factors; and
    grazing incidence (critical angle, penetration depth, footprint). It
    warns when a hexagonal cell has gamma = 90 or the target reflection is
    forbidden. Lab directions are SwissFEL axes (x horizontal transverse,
    y up, z along the beam); the out-of-plane detector circle is gamma.
  - `thc_angles.ipynb` — which detector angles (gamma, delta) the THC
    chamber's in-vacuum detector, Be window and Otti arm can reach; for
    Otti also where its JUNGFRAU module looks at given otti_nu / otti_del.
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
- **`bernina_prep/`** — everything behind the apps: calculations
  (`diffraction.py` angles and detector, `detector_view.py` the detector
  view, `otti.py` the Otti arm's detector (eco's `calc_otti` geometry,
  spin models, sky map), `crystal.py` crystal systems,
  space groups, CIF and structure factors, `grazing.py` penetration depth,
  `thc_angles.py`, `beamsize.py`, `timing.py`), the reflections app itself
  (`reflections_app.py`), the 3D page (`diffractometer_3d.html`) and the
  Bernina setups (`presets.json`). Installed in editable mode by
  `binder/postBuild`, so it's importable from any notebook in the repo.
  gemmi and xraydb are optional: without them the app knows only
  lattice-centring absences and needs the optical constants typed in.
- **`tests/`** — `pytest tests`; `test_diffraction.py` compares the angles
  with diffcalc.
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

## Browser version (draft)

`lite/build.sh` builds the apps as a static site that runs entirely in the
browser (Voici: voila on JupyterLite, Python as WebAssembly), plus
`bernina-apps.zip` with a small launcher per operating system
(`lite/server`, Go). No installation; ~100 MB zip, ~60 MB loaded by the
browser on first start. gemmi has no WebAssembly build, so the browser
version knows only lattice-centring absences. `.github/workflows/lite.yml`
would publish it to GitHub Pages (needs Pages enabled with source "GitHub
Actions").
