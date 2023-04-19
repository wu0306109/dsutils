import numpy as np
from numpy.testing import assert_array_equal

from dsutil import segment


def test_projection_factor():
    # test the projection inside the segment
    assert segment.projection_factor(
        np.array([-1, 4]), (np.array([-2, 0]), np.array([8, 0])),
    ) == 0.1
    # reverse the order of the segment
    assert segment.projection_factor(
        np.array([-1, 4]), (np.array([8, 0]), np.array([-2, 0]))
    ) == 0.9
    # increase the dimension
    assert segment.projection_factor(
        np.array([-1, 5, 4, -3]),
        (np.array([8, 5, 0, -3]), np.array([-2, 5, 0, -3])),
    ) == 0.9

    # test the projection outside the segment and without fixing range
    assert segment.projection_factor(
        np.array([-5, 4]), (np.array([-2, 0]), np.array([8, 0])),
        fixed_range=False,
    ) == -0.3
    # project to another side
    assert segment.projection_factor(
        np.array([10, 4]), (np.array([-2, 0]), np.array([8, 0])),
        fixed_range=False,
    ) == 1.2

    # test the projection outside the segment and with fixing range
    assert segment.projection_factor(
        np.array([-5, 4]), (np.array([-2, 0]), np.array([8, 0])),
        fixed_range=True,
    ) == 0
    # project to another side
    assert segment.projection_factor(
        np.array([10, 4]), (np.array([-2, 0]), np.array([8, 0])),
        fixed_range=True,
    ) == 1


def test_point_projection():
    # test the projection inside the segment
    projection = segment.projection(
        np.array([-1, 2]), (np.array([-2, 0]), np.array([4, 0])))
    assert_array_equal(projection, np.array([-1, 0]))
    # reverse the order of the segment
    projection = segment.projection(
        np.array([-1, 2]), (np.array([4, 0]), np.array([-2, 0])))
    assert_array_equal(projection, np.array([-1, 0]))
    # increase the dimension of the segment
    projection = segment.projection(
        np.array([-1, 10, 2, -3]),
        (np.array([4, 10, 0, -3]), np.array([-2, 10, 0, -3])))
    assert_array_equal(projection, np.array([-1, 10, 0, -3]))

    # test the projection outside the segment if on_segment is False
    projection = segment.projection(
        np.array([-6, 3]), (np.array([-2, 0]), np.array([4, 0])),
        on_segment=False,
    )
    assert_array_equal(projection, np.array([-6, 0]))
    # project to another side
    projection = segment.projection(
        np.array([10, 3]), (np.array([-2, 0]), np.array([4, 0])),
        on_segment=False,
    )
    assert_array_equal(projection, np.array([10, 0]))

    # test the projection outside the segment if on_segment is True
    projection = segment.projection(
        np.array([-6, 3]), (np.array([-2, 0]), np.array([4, 0])),
        on_segment=True,
    )
    assert_array_equal(projection, np.array([-2, 0]))
    # project to another side
    projection = segment.projection(
        np.array([10, 3]), (np.array([-2, 0]), np.array([4, 0])),
        on_segment=True,
    )
    assert_array_equal(projection, np.array([4, 0]))


def test_segment_distance():
    # test the shortest distance is perpendicular to the segment
    assert segment.distance(
        np.array([-1, 4]), (np.array([-2, 0]), np.array([8, 0])),
    ) == 4

    # test the shortest distance is not perpendicular to the segment
    assert segment.distance(
        np.array([-5, 4]), (np.array([-2, 0]), np.array([8, 0])),
    ) == 5
