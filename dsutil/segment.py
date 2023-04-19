"""Functions related to line segments.

For the usage messages, please refer to the `test_segment.py`.
"""
from math import dist
from typing import NamedTuple

import numpy as np
import numpy.typing as npt


class Segment(NamedTuple):
    """A line segment with positions of two ends."""

    begin: npt.ArrayLike
    end: npt.ArrayLike


def projection_factor(
    point: npt.ArrayLike, segment: Segment, fixed_range: bool = False,
) -> float:
    """Compute the factor transforming the segment to the scalar projection.

    The factor `f` is the quantity, ranged in [0, 1] if `fixed_range` is True, 
    where `segment.begin + f * vec{segment}` is the projection point.
    """
    # unpack first to allow the tuple to be passed in
    begin, end = segment

    # calculate the length and the unit vector of the segment
    length = dist(begin, end)
    unit = (end - begin) / length

    # calculate the scalar projection
    scalar = np.dot(point-begin, unit)
    factor = scalar / length

    # fix the projection on the edge if `on_edge` is True
    if fixed_range:
        if factor < 0:
            factor = 0
        elif factor > 1:
            factor = 1

    return factor


def projection(
    point: npt.ArrayLike, segment: Segment, on_segment: bool = False
) -> npt.ArrayLike:
    """Compute the projection of a point onto a line segment.

    The projection would be fixed on the edge if `on_segment` is True, 
    otherwise, maybe outside of or over-cover the edge.
    """
    factor = projection_factor(point, segment, fixed_range=on_segment)

    return segment[0] + factor * (segment[1] - segment[0])


def distance(point: npt.ArrayLike, segment: Segment) -> float:
    """Compute the distance between a point and a line segment."""
    # nearest point on the segment
    nearest = projection(point, segment, on_segment=True)
    return np.linalg.norm(point - nearest)
