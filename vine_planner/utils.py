import numpy as np
import shapely

from shapely import Polygon, LineString
from scipy.spatial import ConvexHull, QhullError
from .constants import (
    YIELD_PRESSURE_PSI,
    LENGTH_FRICTION_PSI_PER_FT,
    TAIL_TENSION_PSI,
    CURVATURE_FRICTION_COEFFICIENT,
    _LENGTH_TO_FEET
)

def vine_robot_pressure(
    length: float,
    angles: float | list[float] | np.ndarray,
    *,
    length_units: str = "ft",
    angle_units: str = "degrees",
) -> float:
    """
    Compute the vine-robot driving pressure in psi.

    Parameters
    ----------
    length:
        Vine-robot path length in ``length_units``.
    angles:
        One turning angle or a collection of turning angles.
    length_units:
        Units used for ``length``. Supported values are ``"ft"``,
        ``"in"``, ``"m"``, and ``"cm"``.
    angle_units:
        Units used for ``angles``: ``"degrees"`` or ``"radians"``.

    Returns
    -------
    float
        Required pressure in psi.
    """
    if length < 0:
        raise ValueError("length must be nonnegative.")

    normalized_length_units = length_units.lower()
    if normalized_length_units not in _LENGTH_TO_FEET:
        raise ValueError(
            "length_units must be one of: 'ft', 'in', 'm', or 'cm'."
        )

    angle_array = np.asarray(angles, dtype=float)
    normalized_angle_units = angle_units.lower()

    if normalized_angle_units in {"degree", "degrees", "deg"}:
        angles_radians = np.deg2rad(angle_array)
    elif normalized_angle_units in {"radian", "radians", "rad"}:
        angles_radians = angle_array
    else:
        raise ValueError(
            "angle_units must be 'degrees' or 'radians'."
        )

    length_ft = float(length) * _LENGTH_TO_FEET[normalized_length_units]
    cumulative_angle_radians = float(np.sum(angles_radians))

    return float(
        YIELD_PRESSURE_PSI
        + (
            LENGTH_FRICTION_PSI_PER_FT * length_ft
            + TAIL_TENSION_PSI
        )
        * np.exp(
            CURVATURE_FRICTION_COEFFICIENT
            * cumulative_angle_radians
        )
    )

def path_metrics(path: LineString) -> tuple[float, float]:
    """Return path length and cumulative unsigned turning angle."""
    coords = np.asarray(path.coords, dtype=float)

    if len(coords) < 2:
        return 0.0, 0.0

    # Remove consecutive duplicate coordinates before computing turns.
    keep = np.concatenate((
        [True],
        np.any(np.diff(coords, axis=0) != 0.0, axis=1),
    ))
    coords = coords[keep]

    if len(coords) < 2:
        return 0.0, 0.0

    vectors = np.diff(coords, axis=0)
    path_length = float(np.linalg.norm(vectors, axis=1).sum())

    if len(vectors) < 2:
        return path_length, 0.0

    incoming = vectors[:-1]
    outgoing = vectors[1:]

    denominators = (
        np.linalg.norm(incoming, axis=1)
        * np.linalg.norm(outgoing, axis=1)
    )

    cosine_angles = np.einsum(
        "ij,ij->i",
        incoming,
        outgoing,
    ) / denominators

    turning_angles = np.arccos(
        np.clip(cosine_angles, -1.0, 1.0)
    )

    return path_length, float(turning_angles.sum())

def pressure_profile(
    path: LineString,
    *,
    length_units: str = "ft",
    points_per_segment: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative path length and required pressure samples.

    Pressure increases linearly with deployed length along each straight
    segment. At every interior path vertex, the accumulated turning angle
    changes instantaneously, so the returned profile contains two pressure
    values at that same cumulative length to display the resulting jump.
    """
    normalized_length_units = length_units.lower()
    if normalized_length_units not in _LENGTH_TO_FEET:
        raise ValueError(
            "length_units must be one of: 'ft', 'in', 'm', or 'cm'."
        )

    if not isinstance(points_per_segment, int) or points_per_segment < 1:
        raise ValueError("points_per_segment must be a positive integer.")

    coords = np.asarray(path.coords, dtype=float)

    # Remove consecutive duplicates so every segment has positive length.
    if len(coords) > 1:
        keep = np.concatenate((
            [True],
            np.any(np.diff(coords, axis=0) != 0.0, axis=1),
        ))
        coords = coords[keep]

    if len(coords) < 2:
        raise ValueError(
            "A pressure profile requires a path with at least two "
            "distinct points."
        )

    segment_vectors = np.diff(coords, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)

    if np.any(segment_lengths == 0.0):
        raise ValueError(
            "A pressure profile cannot contain zero-length segments."
        )

    if len(segment_vectors) > 1:
        incoming = segment_vectors[:-1]
        outgoing = segment_vectors[1:]

        cosine_angles = np.einsum(
            "ij,ij->i",
            incoming,
            outgoing,
        ) / (
            segment_lengths[:-1] * segment_lengths[1:]
        )

        turning_angles = np.arccos(
            np.clip(cosine_angles, -1.0, 1.0)
        )
    else:
        turning_angles = np.empty(0, dtype=float)

    length_to_feet = _LENGTH_TO_FEET[normalized_length_units]
    cumulative_length = 0.0
    cumulative_angle = 0.0

    profile_lengths = []
    profile_pressures = []

    for segment_index, segment_length in enumerate(segment_lengths):
        if segment_index > 0:
            # The previous segment already contributed the pressure just
            # before this turn. Add the post-turn pressure at the same
            # cumulative length to create the vertical jump in the plot.
            cumulative_angle += turning_angles[segment_index - 1]
            post_turn_pressure = (
                YIELD_PRESSURE_PSI
                + (
                    LENGTH_FRICTION_PSI_PER_FT
                    * cumulative_length
                    * length_to_feet
                    + TAIL_TENSION_PSI
                )
                * np.exp(
                    CURVATURE_FRICTION_COEFFICIENT
                    * cumulative_angle
                )
            )
            profile_lengths.append(cumulative_length)
            profile_pressures.append(post_turn_pressure)

        local_lengths = np.linspace(
            0.0,
            segment_length,
            points_per_segment + 1,
        )

        # The segment start is already present after an interior turn.
        if segment_index > 0:
            local_lengths = local_lengths[1:]

        sampled_lengths = cumulative_length + local_lengths
        sampled_pressures = (
            YIELD_PRESSURE_PSI
            + (
                LENGTH_FRICTION_PSI_PER_FT
                * sampled_lengths
                * length_to_feet
                + TAIL_TENSION_PSI
            )
            * np.exp(
                CURVATURE_FRICTION_COEFFICIENT
                * cumulative_angle
            )
        )

        profile_lengths.extend(sampled_lengths.tolist())
        profile_pressures.extend(sampled_pressures.tolist())
        cumulative_length += segment_length

    return (
        np.asarray(profile_lengths, dtype=float),
        np.asarray(profile_pressures, dtype=float),
    )

# ------------------------------------------------------------------
# Sampling helpers
# ------------------------------------------------------------------

def sample_box(
    low_x: float,
    low_y: float,
    high_x: float,
    high_y: float,
    count: int,
    rng: np.random.Generator,
    epsilon: float
) -> np.ndarray:
    return np.column_stack((
        rng.uniform(low_x + epsilon, high_x - epsilon, count),
        rng.uniform(low_y + epsilon, high_y - epsilon, count),
    ))

def sample_uniform_points(
    region,
    number_of_points: int,
    rng
) -> np.ndarray:
    triangulation = shapely.constrained_delaunay_triangles(
        region
    )

    triangles = [
        triangle
        for triangle in shapely.get_parts(triangulation)
        if (
            triangle.geom_type == "Polygon"
            and triangle.area > 0
        )
    ]

    if not triangles:
        raise RuntimeError(
            "The sampling region could not be triangulated."
        )

    areas = np.array(
        [triangle.area for triangle in triangles],
        dtype=float,
    )

    # Selecting triangles in proportion to their areas makes the
    # final points uniform over the complete sampling region.
    selected_indices = rng.choice(
        len(triangles),
        size=number_of_points,
        replace=True,
        p=areas / areas.sum(),
    )

    selected_triangles = np.stack(
        [
            np.asarray(
                triangles[index].exterior.coords,
                dtype=float,
            )[:3, :2]
            for index in selected_indices
        ]
    )

    point_a = selected_triangles[:, 0]
    point_b = selected_triangles[:, 1]
    point_c = selected_triangles[:, 2]

    # Uniform barycentric coordinates within each triangle.
    random_1 = np.sqrt(rng.random(number_of_points))
    random_2 = rng.random(number_of_points)

    return (
        (1.0 - random_1)[:, None] * point_a
        + (
            random_1 * (1.0 - random_2)
        )[:, None] * point_b
        + (
            random_1 * random_2
        )[:, None] * point_c
    )

def sample_outside_hull(quadrant_polygon, hull_points, rng):
    """One point of the quadrant lying outside the current hull.

    Cutting the hull out of the quadrant first is what makes the
    new point a vertex of the grown hull rather than something
    that might land inside it.
    """
    region = quadrant_polygon.difference(Polygon(hull_points))

    if region.is_empty or region.area <= 0.0:
        return None

    try:
        return sample_uniform_points(region, 1, rng)
    except RuntimeError:
        return None

# ------------------------------------------------------------------
# Corner barriers
# ------------------------------------------------------------------

def grow_barrier(quadrant, first_strip, second_strip, rng, min_vertices, max_vertices, epsilon) -> np.ndarray:
    vertex_target = int(
        rng.integers(min_vertices, max_vertices + 1)
    )

    quadrant_polygon = shapely.box(*quadrant)

    required = np.vstack((
        sample_box(*first_strip, 1, rng, epsilon),
        sample_box(*second_strip, 1, rng, epsilon),
    ))

    # A hull needs three points to start from. The retry is only
    # for a genuinely degenerate draw, three near collinear
    # points, not for where the points landed.
    for _ in range(20):
        points = np.vstack((
            required,
            sample_box(*quadrant, 1, rng, epsilon),
        ))

        try:
            hull = ConvexHull(points)
        except QhullError:
            continue

        break
    else:
        raise RuntimeError(
            "Could not build a starting triangle for the barrier."
        )

    hull_points = points[hull.vertices]

    for _ in range(vertex_target - 3):
        extra = sample_outside_hull(
            quadrant_polygon,
            hull_points,
            rng
        )

        if extra is None:
            break

        candidate = np.vstack((points, extra))

        try:
            hull = ConvexHull(candidate)
        except QhullError:
            break

        points = candidate
        hull_points = points[hull.vertices]

    # ConvexHull lists vertices counterclockwise in two
    # dimensions, so this is already a valid ring.
    return hull_points
