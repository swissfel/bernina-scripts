"""bernina_prep.otti against eco's calc_otti and the stored CAD envelope."""
import numpy as np
import pytest

from bernina_prep import otti
from bernina_prep.thc_angles import load_otti_envelope

DATA = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "otti.npy"


def _eco_calc_otti(ottinu, ottidel, ottidet):
    """eco's calc_otti (bernina_sample_environments.py), minus its plotting."""
    from scipy.spatial.transform import Rotation
    x = np.arange(1024) * 75e-3
    z = np.arange(512) * 75e-3
    X, Y, Z = np.meshgrid(x - x.mean(), [179.287], z - z.mean())
    pix = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    rot = lambda v: Rotation.from_rotvec(v).as_matrix()          # noqa: E731
    p = rot([0, np.radians(ottinu), 0]) @ rot([np.radians(ottidel), 0, 0]) @ rot([0, np.radians(ottidet), 0]) @ pix
    p = p + np.array([46, 77.436, 0])[:, None]
    rxz = p[0] ** 2 + p[2] ** 2
    nu = np.arctan2(p[0], p[2]).reshape(X.shape[1:])
    delta = np.arctan2(p[1], np.sqrt(rxz)).reshape(X.shape[1:])
    return np.degrees(nu), np.degrees(delta), np.sqrt(rxz + p[1] ** 2).reshape(X.shape[1:])


@pytest.mark.parametrize("nu, de, det", [(0, 0, 0), (30, 20, -30), (-60, 35, 75), (120, -30, 10)])
def test_matches_eco(nu, de, det):
    az, el, _, r = otti.angles(otti.pixel_positions(nu, de, det))
    eaz, eel, er = _eco_calc_otti(nu, de, det)
    np.testing.assert_allclose(az, eaz, atol=1e-9)
    np.testing.assert_allclose(el, eel, atol=1e-9)
    np.testing.assert_allclose(r, er, atol=1e-9)


def test_reach_matches_cad_envelope():
    x, _y1, _y2, y1s, _ = np.load(DATA)
    az_env, el_env = np.degrees(x), np.degrees(y1s)
    centres, low = otti.reach(otti.DEL_MAX, nu_step=2, bins=36)
    env = np.array([el_env[np.abs(otti._wrap(az_env - c)) < 5].min() for c in centres])
    assert np.sqrt(np.mean((low - env) ** 2)) < 1.0
    g, d = otti.to_gamma_delta(az_env, el_env)
    g1, d1, g2, d2 = load_otti_envelope(DATA)
    assert set(np.round(g, 6)) == set(np.round(np.concatenate([g1, g2]), 6))


def test_spin_models():
    assert otti.spin(30, 0, "twist-free", offset=5) == pytest.approx(-25)
    # a gimbal at zero tilt, or tilted along one of its axes, does not twist
    for nu in (-90, 0, 45, 90):
        assert otti.spin(nu, 0, "cardan") == pytest.approx(otti._wrap(-nu), abs=1e-9)
    assert otti.spin(0, 30, "cardan", psi=0) == pytest.approx(0, abs=1e-9)
    # off-axis it does: about del^2/2 sin 2(nu - psi)
    twist = otti._wrap(otti.spin(45, 30, "cardan", psi=0) - otti.spin(45, 30, "twist-free"))
    assert abs(twist) == pytest.approx(4.11, abs=0.02)
    # the normal is the same in both models
    for model in ("twist-free", "cardan"):
        R = otti.orientation(45, 30, otti.spin(45, 30, model))
        np.testing.assert_allclose(R @ [0, 1, 0], otti.orientation(45, 30, 0) @ [0, 1, 0], atol=1e-12)


def test_fit_cardan_recovers_parameters():
    rng = np.random.default_rng(3)
    nu = rng.uniform(-180, 180, 40)
    de = rng.uniform(0, 44, 40)
    meas = np.array([otti.spin(a, b, "cardan", psi=30.0, offset=12.5) for a, b in zip(nu, de)])
    psi, offset, rms = otti.fit_cardan(nu, de, meas + rng.normal(0, 0.05, 40))
    assert psi == pytest.approx(30.0, abs=0.5) and offset == pytest.approx(12.5, abs=0.1) and rms < 0.1
