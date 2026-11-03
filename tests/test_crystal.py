"""Crystal systems, space groups, CIF and structure factors, grazing incidence."""
from pathlib import Path

import numpy as np
import pytest

from bernina_prep import crystal as cry
from bernina_prep import grazing as gra

DATA = Path(__file__).parent / "data"


def test_cell_for_system():
    assert cry.cell_for_system("hexagonal", 4.143, c=28.636)["gamma"] == 120
    c = cry.cell_for_system("trigonal (rhombohedral axes)", 5.0, alpha=60)
    assert c["a"] == c["b"] == c["c"] and c["alpha"] == c["beta"] == c["gamma"] == 60
    assert cry.cell_for_system("monoclinic", 7.27, 5.0, 5.55, beta=96.8)["beta"] == 96.8


def test_check_cell():
    tetragonal_looking = dict(a=4.143, b=4.143, c=28.636, alpha=90, beta=90, gamma=90)
    assert any("gamma = 120" in w for w in cry.check_cell(tetragonal_looking))
    pytest.importorskip("gemmi")
    assert any("trigonal" in w for w in cry.check_cell(tetragonal_looking, "R-3m"))
    assert cry.check_cell(dict(tetragonal_looking, gamma=120), "R-3m") == []


@pytest.mark.parametrize("use_gemmi", [True, False])
def test_absences(monkeypatch, use_gemmi):
    if use_gemmi:
        pytest.importorskip("gemmi")
    else:
        monkeypatch.setattr(cry, "gemmi", None)
    r = cry.SpaceGroup("R-3m")
    assert not r.is_allowed((0, 0, 1)) and r.is_allowed((0, 0, 3)) and not r.is_allowed((1, 0, 0))
    assert "R centring" in r.why_forbidden((0, 0, 1))
    f = cry.SpaceGroup("Fd-3m")
    assert not f.is_allowed((1, 0, 0)) and f.is_allowed((1, 1, 1))
    # (2 0 0) of diamond: forbidden by the d glide, which only gemmi knows
    assert f.is_allowed((2, 0, 0)) is (not use_gemmi)
    assert f.full is use_gemmi


def test_cif_structure_factors():
    pytest.importorskip("gemmi")
    st = cry.read_cif(DATA / "Bi2Se3.cif")
    assert st.formula == "Bi2Se3" and abs(st.density - 7.66) < 0.05 and st.z_electrons == 804
    assert st.structure_factor((0, 0, 1)) < 1e-6 and st.structure_factor((0, 0, 3)) > 50
    assert abs(st.structure_factor((0, 0, 0)) - 804) < 2          # F(000) = electrons per cell
    df = cry.reflection_list(st.cell, 9000, structure=st, d_min=2.0)
    assert {(0, 0, 3), (0, 0, 6), (1, 0, -5)} <= set(zip(df.h, df.k, df.l))   # (1 0 -5) ~ (0 1 5)
    assert not df[(df.h == 0) & (df.k == 0) & (df.l == 1)]["allowed"].any() if len(df) else True
    assert cry.target_check((0, 0, 1), st.cell, structure=st)


def test_grazing_limits():
    d, b, e = 4.9e-6, 7.5e-8, 9000.0
    ac = gra.critical_angle(d)
    assert abs(ac - np.degrees(np.sqrt(2 * d))) < 1e-12
    lam = 12398.42 / e
    mu = 4 * np.pi * b / lam
    assert abs(gra.penetration_depth(30, d, b, e) / (np.sin(np.radians(30)) / mu) - 1) < 1e-3
    assert gra.penetration_depth(0.3 * ac, d, b, e) < 100            # evanescent: a few nm
    assert gra.probed_depth(1, 1, d, b, e) == pytest.approx(gra.penetration_depth(1, d, b, e) / 2)
