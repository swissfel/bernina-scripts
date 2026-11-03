"""bernina_prep.diffraction (hklforge) against diffcalc, which the app used before."""
import itertools

import numpy as np
import pytest

from bernina_prep import diffraction as dif

CELL = dict(a=5.431, b=5.431, c=5.431, alpha=90, beta=90, gamma=90)
ORIENT = "(1,1,1) > (1,0,0)\n(1,1,-1) > (0,1,0)"
ENERGY = 9000.0


def _sample(**kw):
    return dif.Sample.from_text(CELL, ORIENT, surface=kw.get("surface", "(1,1,1)"),
                                reference=kw.get("reference", "(1,1,1)"))


def _diffcalc_ub(surface=(1, 1, 1), reference=(1, 1, 1)):
    ub_mod = pytest.importorskip("diffcalc.ub.calc")
    ubcalc = ub_mod.UBCalculation("you")
    ubcalc.set_lattice("si", **CELL)
    for hkl, xyz in dif.parse_orientations(ORIENT.splitlines()):
        ubcalc.add_orientation(hkl, tuple(dif.to_you(xyz)), None, None)   # diffcalc's own axes
    ubcalc.n_hkl, ubcalc.surf_nhkl = reference, surface
    ubcalc.calc_ub()
    return ubcalc


def test_ub_equals_diffcalc():
    ubcalc = _diffcalc_ub()
    assert np.allclose(dif.to_you(_sample().UB), ubcalc.UB, atol=1e-12)


@pytest.mark.parametrize("cons", [
    {"eta": 0, "chi": 0, "phi": 0},
    {"mu": 0, "chi": 90, "phi": 0},
    {"mu": 0, "gamma": 0, "chi": 90},
    {"delta": 30, "eta": 0, "chi": 0},
    {"alpha": 2, "mu": 0, "chi": 90},
    {"qaz": 90, "alpha": 1, "phi": 0},
    {"a_eq_b": True, "mu": 0, "eta": 0},
])
@pytest.mark.parametrize("hkl", [(1, 1, 1), (2, 2, 0), (1, 1, 3)])
def test_solutions_equal_diffcalc(cons, hkl):
    calc = pytest.importorskip("diffcalc.hkl.calc")
    from diffcalc.hkl.constraints import Constraints

    ubcalc = _diffcalc_ub()
    dc_cons = {("nu" if k == "gamma" else k): v for k, v in cons.items()}
    try:
        res = calc.HklCalculation(ubcalc, Constraints(dc_cons)).get_position(*hkl, dif.en2lam(ENERGY))
    except Exception:
        res = []
    theirs = sorted(tuple(round(((getattr(p, "nu" if n == "gamma" else n) + 180) % 360) - 180, 4)
                          for n in dif.CIRCLES) for p, _ in res)
    if hkl == (1, 1, 1) and ("gamma" in cons or "a_eq_b" in cons):
        # Q along the phi axis / parallel to the (1 1 1) reference: a continuum,
        # where diffcalc returns arbitrary members
        with pytest.raises(ValueError, match="continuum"):
            _sample().solve(hkl, ENERGY, cons)
        return
    df = _sample().solve(hkl, ENERGY, cons)
    ours = sorted(tuple(round(((df.loc[n, c] + 180) % 360) - 180, 4) for n in dif.CIRCLES) for c in df.columns)
    assert ours == theirs


def test_virtual_angles_use_reference_and_surface_separately():
    s = dif.Sample.from_text(CELL, ORIENT, surface="(1,1,1)", reference="(1,0,0)")
    df = s.solve((1, 1, 3), ENERGY, {"mu": 0, "chi": 90, "betain": 2})
    assert len(df.columns) and np.allclose(df.loc["betain"], 2)
    assert not np.allclose(df.loc["alpha"], df.loc["betain"])     # alpha: the (1 0 0) reference
    with pytest.raises(ValueError, match="surface normal"):
        dif.Sample.from_text(CELL, ORIENT, surface="").solve((1, 1, 1), ENERGY, {"mu": 0, "chi": 90, "betain": 2})


def test_hkl_at_solution_and_legacy_names():
    s = _sample()
    df = s.solve((2, 2, 0), ENERGY, {"gam": 0, "mu": 0, "chi": 90})    # diffcalc-era name
    for c in df.columns:
        assert np.allclose(s.hkl_at(dif.solution_angles(df, c), ENERGY), (2, 2, 0), atol=1e-9)
    with pytest.raises(ValueError, match="mu_is_gam"):
        s.solve((1, 1, 1), ENERGY, {"mu_is_gam": True, "chi": 90, "phi": 0})


def test_passes_wraps_circles_only_with_both_bounds():
    assert dif.passes({"phi": 190.0}, {"phi": (-180.0, -160.0)})
    assert not dif.passes({"phi": 190.0}, {"phi": (None, 180.0)})
    assert dif.passes({"gamma": -5.0}, {"gam": (-10.0, 10.0)})


def test_detector_map_centre_is_the_reflection():
    s = _sample()
    df = s.solve((2, 2, 0), ENERGY, {"mu": 0, "chi": 90, "phi": 0})
    ang = dif.solution_angles(df, 0)
    m = dif.detector_map(s, ang, ENERGY, 500, 77, 77, n=41)
    c = (len(m["v"]) // 2, len(m["u"]) // 2)
    assert np.allclose([m["h"][c], m["k"][c], m["l"][c]], (2, 2, 0), atol=1e-9)
    on = dif.reflections_on_detector(s, ang, ENERGY, 500, 77, 77, tolerance=1e-6)
    assert (2, 2, 0) in set(zip(on["h"], on["k"], on["l"]))
    assert abs(on.loc[0, "u [mm]"]) < 1e-6 and abs(on.loc[0, "v [mm]"]) < 1e-6


def test_reachable_reflections_say_why_not():
    s = _sample()
    t = dif.reachable_reflections(s, ENERGY, {"mu": 0, "chi": 90, "phi": 0},
                                  [(1, 1, 1), (2, 2, 0), (9, 9, 9)], ranges={"delta": (-10.0, 120.0)})
    assert t["reachable"].tolist()[:2] == [True, True]
    assert "energy too low" in t.loc[2, "why not"]
    t = dif.reachable_reflections(s, ENERGY, {"mu": 0, "chi": 90, "phi": 0}, [(2, 2, 0)],
                                  aperture=lambda g, d: False)
    assert "aperture" in t.loc[0, "why not"]
