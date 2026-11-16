"""Crystal setup help: crystal systems, space groups, allowed reflections, CIF.

The mistakes this exists to catch, all seen in practice: a hexagonal or
trigonal cell entered with gamma = 90; aligning on a reflection the space
group forbids (e.g. (0 0 1) of R-3m Bi2Se3); a cell that does not match its
space group.

The calculations are hklforge's (``hklforge.crystal``, no gemmi, also in
the browser): all 230 space groups in their settings with every absence
(centring, glides, screws), equivalent reflections, CIF reading and X-ray
structure factors with anomalous terms. This module keeps the names the
apps use.
"""
from __future__ import annotations

import io

from hklforge.crystal.cell import (AVOGADRO, SYSTEMS, check_cell, cell_for_system,  # noqa: F401
                                   real_direction_in_reciprocal_frame, system_of_number)
from hklforge.crystal.reflections import (collapse_groups, crystal_columns, equivalence_groups,  # noqa: F401
                                          reflection_list, rotation_groups, target_check)
from hklforge.crystal.spacegroups import SpaceGroup  # noqa: F401
from hklforge.crystal.structure import Structure, read_cif  # noqa: F401


def allowed_function(space_group=None, structure: Structure | None = None):
    """``hkl -> bool`` for filtering, or None if nothing is known."""
    sg = structure.space_group if structure is not None and space_group is None else space_group
    if sg is None:
        return None
    sg = sg if isinstance(sg, SpaceGroup) else SpaceGroup(sg)
    return sg.is_allowed


def to_cif_text(upload_bytes) -> str:
    """Bytes of an uploaded CIF -> text (UTF-8, latin-1 as a fallback)."""
    raw = bytes(upload_bytes)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return io.TextIOWrapper(io.BytesIO(raw), encoding="latin-1").read()
