"""The reflections app builds, solves, loads setups and configs, and runs its tools."""
import json
from pathlib import Path

import pytest

pytest.importorskip("ipywidgets")
import matplotlib

matplotlib.use("Agg")

from bernina_prep import crystal as cry
from bernina_prep.reflections_app import ReflectionsApp, load_presets

DATA = Path(__file__).parent / "data"


@pytest.fixture
def app():
    return ReflectionsApp()


def test_default_solves_and_selects(app):
    assert app.sample is not None and len(app.cols) >= 1
    app.select(0)
    msg = json.loads(app.payload_holder.value.split(">", 1)[1].rsplit("<", 1)[0].replace("&quot;", '"'))
    assert msg["label"] == "sol. 0" and len(msg["ub"]) == 3
    assert "Bragg condition" in app.config_hkl.value


def test_presets_are_ecos(app):
    p = load_presets()
    assert set(p) >= {"sd", "sd_horizontal", "thc", "gic"}
    app.load_preset("thc")
    assert app.ctrl["phi_fixed"].value and app.ctrl["chi"].value == 90 and not app.ctrl["eta_fixed"].value
    assert app.filters["delta"][0].value == "45" and app.filters["gamma"][0].value == ""


def test_filters_hide_solutions(app):
    app.load_preset("sd")
    for n, v in zip("hkl", (2, 2, 0)):
        app.ctrl[n].value = v
    n = len(app.df.columns)
    lo, hi = app.filters["delta"]
    lo.value, hi.value = "1000", "1001"
    assert app.cols == [] and "hidden" in app.solve_msg.value and n > 0


def test_forbidden_target_warns(app):
    app.ctrl["space_group"].value = "Fd-3m"
    for n, v in zip("hkl", (1, 0, 0)):          # F centring: known with or without gemmi
        app.ctrl[n].value = v
    assert "forbidden" in app.target_msg.value


def test_hexagonal_with_gamma_90_warns(app):
    app.ctrl["space_group"].value = "R-3m"
    for k, v in dict(uc_a=4.143, uc_b=4.143, uc_c=28.636, uc_gamma=90).items():
        app.ctrl[k].value = v
    assert "gamma = 120" in app.crystal_msg.value
    app.system.value = "trigonal (hexagonal axes)"
    app._apply_system()
    assert app.ctrl["uc_gamma"].value == 120


def test_config_roundtrip_and_legacy(app):
    app.load_preset("thc")
    cfg = app.current_config()
    other = ReflectionsApp()
    other.apply_config(cfg)
    other.recompute()
    assert other.constraints() == app.constraints()
    legacy = {"sample": {"uc_a": 5.43, "surface_normal": "(0,0,1)", "orientations": "(0,0,1) > (1,0,0)"},
              "angles": {"gam": 0.0, "mu": 0.0, "chi": 90.0, "mu_is_gam": False},
              "fixed": {"gam": True, "mu": True, "chi": True, "eta": False, "phi": False, "mu_is_gam": False},
              "filters": {"gam": ["-10", "10"]}}
    other.apply_config(legacy)
    assert other.ctrl["gamma_fixed"].value and other.filters["gamma"][0].value == "-10"
    assert other.ctrl["reference"].value == "(0,0,1)"        # the old app used the normal as reference


def test_tools_run(app):
    app.select(0)
    for run, key in ((app.run_table, "table"), (app.run_reachable, "reachable"),
                     (app.run_detector, "detector"), (app.run_crystal, "crystal")):
        run()
        assert not isinstance(app.results.get(key), Exception), (key, app.results.get(key))
    assert len(app.results["table"]) == 4 and app.results["reachable"]["reachable"].any()
    m, spots = app.results["detector"]
    assert m["h"].shape[0] > 10
    for cb in app.d_over.values():
        cb.value = True
    for key in ("zone", "tds", "alpha_f"):
        app.d_qty.value = key                      # redraws from the cached map
        assert isinstance(app.results["detector notes"], list)
    crystal = app.results["crystal"]
    allowed = crystal[crystal["allowed"]]
    families = {tuple(sorted(map(abs, r))) for r in zip(allowed.h, allowed.k, allowed.l)}
    assert {(1, 1, 1), (0, 2, 2), (1, 1, 3)} <= families and (0, 0, 1) not in families
    assert ((0, 0, 2) in families) is (cry.gemmi is None)     # (2 0 0): removed by the d glide
    app.g_delta.value, app.g_beta.value = "4.9e-6", "7.5e-8"         # Si at 9 keV, no xraydb needed
    app.run_grazing()
    assert app.results["grazing"]["critical angle"] == pytest.approx(0.179, abs=1e-3)


def test_cif_upload_fills_sample(app):
    pytest.importorskip("gemmi")
    content = (DATA / "Bi2Se3.cif").read_bytes()
    app._on_cif({"new": ({"name": "Bi2Se3.cif", "content": memoryview(content)},)})
    assert app.ctrl["uc_gamma"].value == 120 and "R -3 m" in app.ctrl["space_group"].value
    assert "Bi2Se3" in app.g_formula.value and 7.5 < app.g_density.value < 7.8


def _payload(app):
    import html
    return json.loads(html.unescape(app.payload_holder.value.split(">", 1)[1].rsplit("<", 1)[0]))


def test_detector_in_step_with_3d_view(app):
    import re
    from bernina_prep import diffraction as dif
    from bernina_prep.reflections_app import VIEWER_HTML
    # the page's own copy of the preset list (used standalone, e.g. in eco)
    page = re.search(r"const DEFAULT_DET_PRESETS = (\[.*?\]);\n", VIEWER_HTML.read_text()).group(1)
    assert json.loads(page) == [[k, w, h] for k, (w, h) in dif.DETECTOR_PRESETS.items()]
    # chosen in the tool: the next push carries it to the page
    jf01 = next(k for k in dif.DETECTOR_PRESETS if k.startswith("JF01") and "turned" in k)
    app.d_preset.value = jf01
    msg = _payload(app)
    assert msg["detector"] == {"w": 121.05, "h": 77.25} and [jf01, 121.05, 77.25] in msg["detectors"]
    # chosen in the page: the tool follows, with the module gaps of the preset
    jf07 = next(k for k in dif.DETECTOR_PRESETS if k.startswith("JF07") and "turned" not in k)
    app.det_from_view.value = json.dumps({"name": jf07, "w": 316.125, "h": 332.4})
    assert app.d_preset.value == jf07 and (app.d_w.value, app.d_h.value) == (316.125, 332.4)
    assert len(app._detector_modules()) == 32
    app.det_from_view.value = json.dumps({"name": "", "w": 50, "h": 60})
    assert (app.d_w.value, app.d_h.value) == (50, 60) and app._detector_modules() is None
    app.det_from_view.value = "not json"          # ignored
    assert _payload(app)["detector"] == {"w": 50, "h": 60}


def test_state_remembered_in_browser(app):
    from bernina_prep import diffraction as dif
    # nothing is saved before the page has offered what it remembers
    app.push_view()
    assert "autosave" not in _payload(app)
    app.state_from_view.value = "{}"                 # nothing remembered: start saving
    assert _payload(app)["autosave"]["energy"] == app.ctrl["energy"].value
    # change things, then carry the state over into a new session
    app.ctrl["h"].value, app.ctrl["energy"].value = 2.0, 8500.0
    app.filters["delta"][1].value = "170"
    jf01 = next(k for k in dif.DETECTOR_PRESETS if k.startswith("JF01") and "turned" not in k)
    app.d_preset.value, app.d_dist.value = jf01, 320.0
    assert len(app.cols) >= 2
    app.select(1)
    state = _payload(app)["autosave"]
    new = ReflectionsApp()
    new.state_from_view.value = json.dumps({"state": state, "display": {"showCell": True}, "savedText": "today"})
    assert new.ctrl["h"].value == 2.0 and new.ctrl["energy"].value == 8500.0
    assert new.filters["delta"][1].value == "170"
    assert new.d_preset.value == jf01 and new.d_dist.value == 320.0 and new.selected == app.selected
    assert new.restore_box.layout.display == "flex" and "today" in new.restore_note.value
    assert _payload(new)["autosave"]["hkl"]["h"] == 2.0
    new.state_from_view.value = json.dumps({"state": {"energy": 7000.0}})   # only once per session
    assert new.ctrl["energy"].value == 8500.0
    # start fresh: the page is told to forget and reload; no more autosaves
    new.fresh_btn.click()
    msg = _payload(new)
    assert msg.get("forgetState") is True and "autosave" not in msg
