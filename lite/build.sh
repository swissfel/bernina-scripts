#!/bin/bash
# Build the browser version of the apps (Voici: voila on JupyterLite) and the
# downloadable zip with a small launcher per operating system.
#
#   lite/build.sh            -> lite/_output/site/          (the static site)
#                               lite/_output/bernina-apps.zip
#   LITE_OUT=/elsewhere lite/build.sh   builds there instead (~2 GB while building)
#
# Needs python >= 3.10 (python or python3, or $PYTHON) and, for the launchers, go (skipped if missing).
# The site can be put on any static web host (GitHub Pages: see
# .github/workflows/lite.yml); locally: python -m http.server -d lite/_output/site
set -euo pipefail
cd "$(dirname "$0")"
out=$(realpath -m "${LITE_OUT:-_output}")
rm -rf "$out"
export PIP_CACHE_DIR="$out/pip-cache" GOCACHE="$out/go-cache"
mkdir -p "$out/wheels" "$out/content"

"${PYTHON:-$(command -v python || command -v python3)}" -m venv "$out/venv"
"$out/venv/bin/pip" install -q voici ipywidgets ipympl
"$out/venv/bin/pip" wheel -q --no-deps -w "$out/wheels" ..
rm -rf ../build                       # setuptools' scratch folder
wheel=$(ls "$out"/wheels/bernina_prep-*.whl)
sed "s#@WHEEL@#$(realpath "$wheel")#" environment.yml.in > environment.yml

cp ../apps/*.ipynb "$out/content/"    # not apps/dev
"$out/venv/bin/voici" build --contents "$out/content" --output-dir "$out/site"
rm environment.yml

# the zip: site + launchers + read-me
pkg="$out/bernina-apps"
mkdir -p "$pkg"
cp -r "$out/site" "$pkg/site"
cp README.txt "$pkg/"
if command -v go >/dev/null; then
    (cd server
     for target in windows/amd64 darwin/arm64 darwin/amd64 linux/amd64; do
         os=${target%/*}; arch=${target#*/}
         name=bernina-apps-$os-$arch; [ "$os" = windows ] && name=$name.exe
         [ "$os" = darwin ] && name=bernina-apps-macos-$arch
         GOOS=$os GOARCH=$arch CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o "$pkg/$name" .
     done)
else
    echo "go not found: zip without launchers"
fi
(cd "$out" && zip -qr bernina-apps.zip bernina-apps)
du -sh "$out/site" "$out/bernina-apps.zip"
