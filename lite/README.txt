Bernina apps - browser version
==============================

The reflections calculator, beam size, THC angles, time resolution and the
3D diffractometer, running entirely in your web browser: Python runs inside
the browser (WebAssembly), nothing is installed and nothing is sent anywhere.

Start
-----
Double-click the program for your system. It opens the apps in your default
browser; keep its window open while you use them.

  Windows                  bernina-apps-windows-amd64.exe
  Mac (Apple silicon)      bernina-apps-macos-arm64
  Mac (Intel)              bernina-apps-macos-amd64
  Linux                    bernina-apps-linux-amd64

Mac: the first time, right-click the program and choose "Open" (it is not
signed by Apple). Windows may ask once whether to allow it; it only listens
on this computer (127.0.0.1).

Or, with Python installed:   python -m http.server -d site
and open http://localhost:8000/voici/render/diffraction_reflections.html

The first start takes about half a minute (the browser loads Python and the
scientific packages, ~60 MB); after that they come from the browser cache.

Differences from the Binder/beamline version
--------------------------------------------
No gemmi in the browser: space groups give only lattice-centring absences
(no glide/screw absences), CIF files cannot be read and there are no
structure factors. Everything else is the same.
