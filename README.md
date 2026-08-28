# bernina-scripts
Notebooks and scripts for bernina

## Run on mybinder

| notebook | JupyterLab | app (voila) |
| --- | --- | --- |
| diffraction angle calculator | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/swissfel/bernina-scripts/HEAD?labpath=prep_notebooks%2Fapplet_sim_diff_bernina.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/swissfel/bernina-scripts/HEAD?urlpath=voila%2Frender%2Fprep_notebooks%2Fapplet_sim_diff_bernina.ipynb) |
| pump-probe time resolution | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/swissfel/bernina-scripts/HEAD?labpath=prep_notebooks%2Ftime_resolution.ipynb) | |

The notebooks are shipped without stored outputs, so in JupyterLab use
*Run > Restart Kernel and Run All Cells* to build the input fields. The voila
link runs the notebook for you and shows only the widgets.

Binder builds from [`binder/environment.yml`](binder/environment.yml), not from the
`environment.yml` in the repo root - see the comments in that file before adding
packages to it, the widget stack there is deliberately constrained.

## Local environment

`environment.yml` in the repo root is the full beamline/analysis environment
(PSI channels, `jungfrau_utils`, `bsread`, `escape-fel`, ...) and is not used by
Binder.
