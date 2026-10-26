"""Measured geometry shared by Steve's known-carton deployment tools.

All values are expressed in ``ur5ebase_link`` metres.  Keep these constants in
one module so scene registration and execution preflight cannot silently use
different carton or table dimensions.
"""

from __future__ import annotations

import math


# The friend pipeline's textured carton mesh, sorted as thin/mid/wide.  The
# carton is placed flat with its wide axis vertical, so the ordering does not
# matter for the enclosing-sphere calculation below.
KNOWN_CARTON_DIMS_M = (0.031, 0.090, 0.127)

# Measured Steve top-deck proxy: 0.5 m square, with the arm base 0.11 m from
# its rear edge.  Its top is flush with z=0 in ur5ebase_link.
SUPPORT_CENTER_M = (0.0, 0.14, -0.5)
SUPPORT_DIMS_M = (0.5, 0.5, 1.0)

# Fixed obstacle in front of the arm from the hardware-tested friend pipeline,
# including its 1 cm tape-measure padding on every side.
FRONT_BOX_CENTER_M = (0.0, 0.34, 0.04)
FRONT_BOX_DIMS_M = (0.10, 0.22, 0.10)


def known_carton_keepout_radius(placement_uncertainty_m: float) -> float:
    """Return a sphere enclosing the whole carton plus placement uncertainty.

    Dimensions describe full side lengths.  The farthest point from the box
    centre is therefore half the 3-D diagonal, not the full diagonal.
    """
    if not math.isfinite(placement_uncertainty_m) or placement_uncertainty_m < 0.0:
        raise ValueError("placement uncertainty must be finite and nonnegative")
    half_diagonal = 0.5 * math.sqrt(sum(value * value for value in KNOWN_CARTON_DIMS_M))
    return half_diagonal + placement_uncertainty_m
