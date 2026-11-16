"""The reflections app builds, solves, loads setups and configs, and runs its tools."""
import base64
import json
from pathlib import Path

import numpy as np
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
    app.t_source.value = "list"
    for run, key in ((app.run_table, "table"), (app.run_reachable, "reachable"),
                     (app.run_detector, "detector"), (app.run_crystal, "crystal")):
        run()
        assert not isinstance(app.results.get(key), Exception), (key, app.results.get(key))
    assert len(app.results["table"]) == 4 and app.results["reachable"]["reachable"].any()
    # results are written into the tools' HTML areas (not Output widgets), busy line gone, buttons back
    for area, btn in ((app.t_out, app.t_btn), (app.r_out, app.r_btn), (app.c_out, app.c_btn)):
        assert "<table" in area.widget.value and "&#8987;" not in area.widget.value and not btn.disabled
    assert app.t_progress.layout.display == "none"
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
    assert (0, 0, 2) not in families                          # (2 0 0): removed by the d glide
    app.g_delta.value, app.g_beta.value = "4.9e-6", "7.5e-8"         # Si at 9 keV, no xraydb needed
    app.run_grazing()
    assert app.results["grazing"]["critical angle"] == pytest.approx(0.179, abs=1e-3)


def test_cif_upload_fills_sample(app):
    content = (DATA / "Bi2Se3.cif").read_bytes()
    app._on_cif({"new": ({"name": "Bi2Se3.cif", "content": memoryview(content)},)})
    assert app.ctrl["uc_gamma"].value == 120 and "R -3 m" in app.ctrl["space_group"].value
    assert "Bi2Se3" in app.g_formula.value and 7.5 < app.g_density.value < 7.8


def _payload(app):
    import html
    return json.loads(html.unescape(app.payload_holder.value.split(">", 1)[1].rsplit("<", 1)[0]))


def test_detector_in_step_with_3d_view(app):
    import re
    from escape.bernina import diffraction as dif
    from bernina_prep.reflections_app import bernina_view_page
    # the preset list built into the page (used standalone, apps/diffractometer_3d.ipynb)
    cfg = json.loads(re.search(r"<script>window\.HKLFORGE_VIEW = (\{.*?\});</script>", bernina_view_page()).group(1))
    assert cfg["detectors"] == [[k, w, h] for k, (w, h) in dif.DETECTOR_PRESETS.items()]
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
    from escape.bernina import diffraction as dif
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



def test_snapshots_in_browser(app):
    # the controls wait for the 3D page's first report
    assert app.snap_btn.disabled
    app.state_from_view.value = json.dumps({"kind": "offer", "remember": False, "ack": 0, "snapshots": []})
    assert not app.snap_btn.disabled and app.remember.value is False
    assert _payload(app)["memory"] == {"remember": False, "ops": []}
    app.remember.value = True
    assert _payload(app)["memory"]["remember"] is True
    # take one: an op is sent until the page acknowledges it
    app.ctrl["energy"].value = 8123.0
    app.snap_name.value = "before beamtime"
    app.snap_btn.click()
    ops = _payload(app)["memory"]["ops"]
    assert len(ops) == 1 and ops[0]["op"] == "add" and ops[0]["name"] == "before beamtime"
    assert ops[0]["state"]["energy"] == 8123.0 and app.snap_name.value == ""
    snap = {"id": "a1", "name": "before beamtime", "saved": 0, "auto": False,
            "state": ops[0]["state"], "display": {"showCell": True}}
    auto = {"id": "a0", "name": "last session", "saved": 0, "auto": True, "state": {"energy": 7000.0}}
    app.state_from_view.value = json.dumps({"kind": "memory", "remember": True, "ack": ops[0]["seq"],
                                            "n": 2, "snapshots": [auto, snap]})
    assert app._memory_ops == []                       # acknowledged: not sent again
    assert [v for _, v in app.snap_list.options] == ["a1", "a0"] and app.snap_list.value == "a1"
    assert app.snap_list.options[0][0].startswith("★ before beamtime · ") and app.snap_list.options[1][0].startswith("auto:")
    assert "download" in app.snap_download.value
    # restore: the state being left becomes an automatic snapshot
    app.ctrl["energy"].value = 9999.0
    app.snap_restore.click()
    assert app.ctrl["energy"].value == 8123.0 and "before beamtime" in app.restore_note.value
    ops = _payload(app)["memory"]["ops"]
    assert ops[-1]["auto"] and ops[-1]["state"]["energy"] == 9999.0
    # delete, and forget everything only on the second click
    app.snap_delete.click()
    assert _payload(app)["memory"]["ops"][-1] == {"op": "delete", "id": "a1", "seq": ops[-1]["seq"] + 1}
    app.forget_btn.click()
    assert _payload(app)["memory"]["ops"][-1]["op"] == "delete"
    app.forget_btn.click()
    assert _payload(app)["memory"]["ops"][-1]["op"] == "clear"
    # a downloaded snapshot loads with its detector tool settings
    cfg = json.loads(base64.b64decode(app.snap_download.value.split("base64,")[1].split('"')[0]))
    assert cfg["energy"] == 8123.0 and "tools" in cfg and cfg["display"] == {"showCell": True}

def test_two_reflections_rows_and_second_detector(app):
    from escape.bernina import diffraction as dif
    for n in dif.CONSTRAINTS:                      # hold mu = gamma = 0 and the second reflection
        app.ctrl[f"{n}_fixed"].value = n in ("mu", "gamma")
    app.ctrl["mu"].value = app.ctrl["gamma"].value = 0.0
    app.ctrl["two"].value = True
    app.ctrl["h2"].value, app.ctrl["k2"].value, app.ctrl["l2"].value = 1.0, 1.0, 3.0
    assert len(app.cols) >= 1, app.solve_msg.value
    app.select(0)
    rows = list(app.df.index)
    assert {"∠(Q1,Q2)", "∠(k′1,k′2)", "gamma @2", "delta @2", "ttheta @2"} <= set(rows)
    assert not any(r.endswith(" @1") for r in rows)          # detector on reflection 1: no block for it
    assert "detector on reflection 2 (1 1 3)" in app.table_html.value
    ghosts = _payload(app)["ghosts"]
    assert len(ghosts) == 1 and ghosts[0]["which"] == 2 and ghosts[0]["label"] == "(1 1 3)"
    col = app.df[app.cols[0]]
    assert (ghosts[0]["gamma"], ghosts[0]["delta"]) == pytest.approx((col["gamma @2"], col["delta @2"]))
    # detector between both (mu and chi held): a block for each reflection, two detectors in the view
    app.ctrl["gamma_fixed"].value = False
    app.ctrl["chi_fixed"].value, app.ctrl["chi"].value = True, 90.0
    app.ctrl["detector"].value = "between"
    assert len(app.cols) >= 1, app.solve_msg.value
    app.select(0)
    assert {"gamma @1", "gamma @2"} <= set(app.df.index)
    assert sorted(g["which"] for g in _payload(app)["ghosts"]) == [1, 2]
    # one reflection again: no per-reflection rows, no second detector
    app.ctrl["two"].value = False
    assert app.df is None or not any("@" in r for r in app.df.index)
    assert _payload(app)["ghosts"] == []


def test_table_groups_and_crystal_columns(app):
    from escape.bernina import diffraction as dif
    content = (DATA / "Bi2Se3.cif").read_bytes()
    app._on_cif({"new": ({"name": "Bi2Se3.cif", "content": memoryview(content)},)})
    app.ctrl["orientations"].value = "(0,0,1) > (0,1,0)\n(1,0,0) > (1,0,0)"     # c along the beam-up plane
    app.t_source.value, app.t_dmin.value, app.t_max.value = "all", 2.0, 60
    app.run_table()
    t = app.results["table"]
    assert {"|F|", "|F|²", "I rel", "allowed"} <= set(t.columns) and t["allowed"].all()
    # symmetry-equivalent: families
    app.t_group.value = "sym"
    app.run_table()
    t = app.results["table"]
    assert "family" in t.columns and t.groupby("group")["family"].nunique().max() == 1
    # about the phi axis: the change of phi brings each member where the first one was
    app.t_group.value = "phi"
    app.run_table()
    t = app.results["table"].reset_index()
    assert "Δphi [deg]" in t.columns and t.groupby("group").size().max() > 1
    checked = 0
    for _, grp in t.groupby("group"):
        first = grp.iloc[0]
        if len(grp) < 2 or first["error"] != "" or not np.isfinite(grp["Δphi [deg]"]).all():
            continue
        a = {n: float(first[n]) for n in dif.CIRCLES}
        for _, m in grp.iloc[1:].iterrows():
            moved = dict(a, phi=a["phi"] + m["Δphi [deg]"])
            np.testing.assert_allclose(app.sample.hkl_at(moved, app.energy()), (m.h, m.k, m.l), atol=1e-6)
            checked += 1
    assert checked >= 5
    # one row per group, a direction, and a bad direction
    app.t_one.value = True
    app.run_table()
    assert "members" in app.results["table"].columns
    app.t_group.value, app.t_dir.value = "dir", "[0 0 1]"
    app.run_table()
    assert any(c.startswith("Δ about [0 0 1]") for c in app.results["table"].columns)
    app.t_dir.value = "0 0 1"
    app.run_table()
    assert isinstance(app.results["table"], Exception)


def test_position_buttons(app):
    names = [b.description for b in app.position_buttons.children]
    assert names == ["constraint configuration", "zero position"]
    app.position_buttons.children[1].click()
    assert app.selected == "zero" and _payload(app)["label"] == "zero position"
    assert app.position_buttons.children[1].button_style == "info"
    assert all(b.button_style == "" for b in app.solution_buttons.children[1:])
    app.position_buttons.children[0].click()
    assert app.selected == "config" and _payload(app)["label"] == "constraint configuration"


def test_view_is_hklforge_with_bernina_geometry(app):
    """The 3D page is hklforge's with the Bernina geometry; vectors go in the
    lab (SwissFEL) frame, the browser storage keeps the earlier keys."""
    import re
    import numpy as np
    from escape.bernina import diffraction as dif
    from hklforge.view import geometry_message
    from bernina_prep.reflections_app import bernina_view_page
    cfg = json.loads(re.search(r"<script>window\.HKLFORGE_VIEW = (\{.*?\});</script>", bernina_view_page()).group(1))
    assert cfg["geometry"] == json.loads(json.dumps(geometry_message(dif.GEOMETRY)))
    assert cfg["storage"] == "bernina-reflections" and cfg["id"] == "diff3d"
    msg = _payload(app)
    assert np.allclose(msg["ub"], app.sample.UB)                       # lab frame, no to_you any more
    assert np.allclose(msg["nPhi"], app.sample.n_surface) and np.allclose(msg["refPhi"], app.sample.n_reference)
    assert msg["bz"]["variant"] == "FCC" and "L" in msg["bz"]["points"]   # Si, Fd-3m
    app.ctrl["surface_normal"].value = ""
    assert _payload(app)["nPhi"] is None                                 # cleared in the view too


def test_store_downloads_and_keeps_history(app):
    app.file_name.value = "my run"
    app.include["display"].value = False
    app._on_make()
    link = app.download_link.value
    assert 'id="diff3d-config-download"' in link and 'download="my_run.json"' in link
    assert 'data-display="0"' in link
    msg = _payload(app)
    assert msg["download"] == 1                                         # the page clicks the link once
    op = app._memory_ops[-1]
    assert op["op"] == "add" and op["kind"] == "config" and op["name"] == "my_run.json"
    assert op["state"]["sample"]["space_group"] == "Fd-3m"
    app._on_make()
    assert _payload(app)["download"] == 2
    assert app._snapshot_label({"name": "my_run.json", "kind": "config", "savedText": "1.1.2026"}) == \
        "config: my_run.json · 1.1.2026"


def test_scan_counts_what_solve_shows(app):
    import numpy as np
    from escape.bernina import diffraction as dif
    assert [v for _, v in app.s_name.options] == ["eta", "chi", "phi"]  # the held ones
    app.s_name.value = "chi"
    app.filters["delta"][0].value = "0"                                  # a min/max is honoured
    app.s_from.value, app.s_to.value, app.s_steps.value = -90, 90, 19
    app.run_scan()
    df = app.results["scan"]
    assert len(df) == 19
    for v, n in zip(df["chi"], df["solutions"]):
        cons = {**app.constraints(), "chi": float(v)}
        sol = app.sample.solve(app.hkl(), app.energy(), cons)
        want = sum(dif.passes(sol[c], app.ranges()) for c in sol.columns) if len(sol.columns) else 0
        assert n == want, (v, n, want)
    # a click in the plot sets the scanned angle
    class Ev:
        inaxes, xdata = True, 12.345
    app._scan_click(Ev())
    assert app.ctrl["chi"].value == 12.35


def test_geometry_tab_built_when_opened(app):
    assert app.explorer is None
    app.tool_tabs.selected_index = 5
    e = app.explorer
    assert e is not None and e.pick.value.startswith("bernina")
    assert e.geometry.names == ["mu", "eta", "chi", "phi", "gamma", "delta"]
    assert 'class="geo3d-frame"' in e.view.value
