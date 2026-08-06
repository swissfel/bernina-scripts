"""Pump-probe fluence and grazing-incidence timing calculations.

Ported directly from calx.pumpprobe (github.com/htlemke/calx) -- pure
numpy/scipy physics with no facility-specific dependencies, so it's
reimplemented here rather than pulling in the full calx package (which adds
a heavier, git-installed dependency not needed for just these two
functions).
"""
from __future__ import annotations

import numpy as np
from scipy import constants


def fluence(energy_j: float, area_m2: float, inc_angle_rad: float = np.pi / 2, reflectivity: float = 0.0) -> float:
    """Absorbed fluence at a surface, accounting for the geometric spreading
    of the beam footprint at oblique incidence.

    Parameters
    ----------
    energy_j : pulse energy [J]
    area_m2 : beam cross-sectional area, normal to the wavevector [m^2]
    inc_angle_rad : angle between the surface plane and the incoming beam
        [rad] (pi/2 = normal incidence; smaller = more grazing, spreading
        the same energy over a larger illuminated footprint)
    reflectivity : fraction of the pulse reflected, not absorbed (0-1)

    Returns
    -------
    Absorbed fluence in mJ/cm^2.
    """
    return energy_j * 1e3 / area_m2 / 1e4 * (1 - reflectivity) * np.sin(inc_angle_rad)


def grazing_incidence_time_resolution(probe_size_m: float, probe_angle_rad: float, pump_angle_rad: float) -> float:
    """Geometric (footprint-driven) contribution to the pump-probe time
    resolution at grazing incidence.

    At grazing incidence, the illuminated footprint is elongated along the
    beam direction; because the pump and probe beams cross the surface at
    different angles, the pump wavefront doesn't reach every point of that
    footprint simultaneously. This geometric path-length mismatch adds a
    time smearing on top of (and independent of) the pulse durations
    themselves -- purely a consequence of geometry, not pulse length.

    Parameters
    ----------
    probe_size_m : probe beam size [m]
    probe_angle_rad, pump_angle_rad : each beam's own grazing angle to the
        sample surface [rad] (typically very different: e.g. a very
        grazing X-ray probe together with a steeper optical pump)

    Returns
    -------
    Geometric time-smearing in seconds.
    """
    footprint = probe_size_m / np.sin(probe_angle_rad)
    len_probe = probe_size_m / np.tan(probe_angle_rad)
    len_pump = np.cos(pump_angle_rad) * footprint
    return (len_pump - len_probe) / constants.c
