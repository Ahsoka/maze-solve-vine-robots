import numpy as np
import itertools
import shapely

from shapely import Polygon, LineString, Point
from scipy.spatial import ConvexHull, QhullError
from .constants import (
    YIELD_PRESSURE_PSI,
    LENGTH_FRICTION_PSI_PER_FT,
    TAIL_TENSION_PSI,
    CURVATURE_FRICTION_COEFFICIENT,
    _LENGTH_TO_FEET
)

# ------------------------------------------------------------------
# Pressure model
# ------------------------------------------------------------------
#
# Recursive (nested capstan) growth pressure. For a path of n segments
# with lengths l_1 ... l_n and n - 1 interior turning angles t_1 ... t_{n-1}:
#
#     tau_0 = T
#     tau_i = (tau_{i-1} + f * l_i) * exp(mu_c * |t_i|)      i = 1 ... n-1
#     P     = Y + tau_{n-1} + f * l_n
#
# Each segment's drag is amplified only by the bends *distal* to it, i.e.
# the bends its accumulated tension must still be dragged through on the
# way to the tip. The terminal segment has nothing distal to it and is
# therefore charged at face value.
#
# This replaces the summed form P = Y + (f * L + T) * exp(mu_c * sum|t_i|),
# which charges every segment for every bend and is a strict upper bound.
# ------------------------------------------------------------------

_DEGREE_ALIASES = {"degree", "degrees", "deg"}
_RADIAN_ALIASES = {"radian", "radians", "rad"}


def _to_radians(angles, angle_units: str) -> np.ndarray:
    angle_array = np.atleast_1d(
        np.asarray(angles, dtype=float)
    ).ravel()

    normalized = angle_units.lower()

    if normalized in _DEGREE_ALIASES:
        return np.deg2rad(angle_array)

    if normalized in _RADIAN_ALIASES:
        return angle_array

    raise ValueError("angle_units must be 'degrees' or 'radians'.")


def _length_scale(length_units: str) -> float:
    normalized = length_units.lower()

    if normalized not in _LENGTH_TO_FEET:
        raise ValueError(
            "length_units must be one of: 'ft', 'in', 'm', or 'cm'."
        )

    return _LENGTH_TO_FEET[normalized]


def segment_geometry(path) -> tuple[np.ndarray, np.ndarray]:
    """Return per-segment lengths and per-vertex turning angles.

    Accepts a ``LineString`` or any ``(n + 1, d)`` array of vertices, in
    two or three dimensions. Consecutive duplicate vertices are dropped.
    Turning angles are unsigned and reported in radians; there is one
    per interior vertex, so ``len(angles) == len(lengths) - 1``.
    """
    coords = np.asarray(
        path.coords if hasattr(path, "coords") else path,
        dtype=float,
    )

    if coords.ndim != 2 or len(coords) < 2:
        return np.zeros(0), np.zeros(0)

    keep = np.concatenate((
        [True],
        np.any(np.diff(coords, axis=0) != 0.0, axis=1),
    ))
    coords = coords[keep]

    if len(coords) < 2:
        return np.zeros(0), np.zeros(0)

    vectors = np.diff(coords, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)

    if len(vectors) < 2:
        return lengths, np.zeros(0)

    incoming = vectors[:-1]
    outgoing = vectors[1:]

    cosine_angles = np.einsum(
        "ij,ij->i",
        incoming,
        outgoing,
    ) / (
        lengths[:-1] * lengths[1:]
    )

    angles = np.arccos(
        np.clip(cosine_angles, -1.0, 1.0)
    )

    return lengths, angles


def vine_robot_pressure(
    segment_lengths,
    angles,
    *,
    length_units: str = "ft",
    angle_units: str = "degrees",
    yield_pressure: float = YIELD_PRESSURE_PSI,
    length_friction: float = LENGTH_FRICTION_PSI_PER_FT,
    tail_tension: float = TAIL_TENSION_PSI,
    curvature_coefficient: float = CURVATURE_FRICTION_COEFFICIENT,
) -> float:
    """
    Compute the vine-robot driving pressure in psi (recursive capstan model).

    Parameters
    ----------
    segment_lengths:
        Per-segment lengths in ``length_units``, ordered base to tip. A
        bare float is accepted for a single straight segment.
    angles:
        Turning angles at the interior vertices, one per vertex, so
        ``len(angles) == len(segment_lengths) - 1``. Signs are ignored.
    length_units:
        Units used for ``segment_lengths``: ``"ft"``, ``"in"``, ``"m"``,
        or ``"cm"``.
    angle_units:
        Units used for ``angles``: ``"degrees"`` or ``"radians"``.
    yield_pressure, length_friction, tail_tension, curvature_coefficient:
        Model parameters, defaulting to the fitted values in
        ``constants``. ``length_friction`` is psi per foot regardless of
        ``length_units``.

    Returns
    -------
    float
        Required pressure in psi.

    Notes
    -----
    This function needs the *interleaving* of lengths and bends, not just
    the totals, so it deliberately rejects the old ``(total_length,
    total_angle)`` call signature rather than silently reinterpreting it.
    """
    lengths = np.atleast_1d(
        np.asarray(segment_lengths, dtype=float)
    ).ravel()

    turning_angles = np.abs(
        _to_radians(angles, angle_units)
    )

    if lengths.size == 0:
        raise ValueError(
            "segment_lengths must contain at least one segment."
        )

    if np.any(lengths < 0.0):
        raise ValueError("Segment lengths must be nonnegative.")

    if turning_angles.size != lengths.size - 1:
        raise ValueError(
            "Expected one turning angle per interior vertex: "
            f"len(angles) == len(segment_lengths) - 1 == "
            f"{lengths.size - 1}, got {turning_angles.size}. The "
            "recursive pressure model needs per-segment lengths, not a "
            "single total length."
        )

    if curvature_coefficient < 0.0:
        raise ValueError("curvature_coefficient must be nonnegative.")

    lengths_ft = lengths * _length_scale(length_units)
    drag = length_friction * lengths_ft
    amplification = np.exp(curvature_coefficient * turning_angles)

    # tau_i = (tau_{i-1} + f * l_i) * exp(mu_c * |t_i|), then the
    # terminal segment is added without amplification.
    tension = float(tail_tension)

    for segment_drag, factor in zip(drag[:-1], amplification):
        tension = (tension + segment_drag) * factor

    tension += drag[-1]

    return float(yield_pressure + tension)


def path_pressure(
    path,
    *,
    length_units: str = "ft",
    **model_parameters,
) -> float:
    """Growth pressure of a ``LineString`` or vertex array, in psi."""
    lengths, angles = segment_geometry(path)

    if lengths.size == 0:
        raise ValueError(
            "A pressure requires a path with at least two distinct points."
        )

    return vine_robot_pressure(
        lengths,
        angles,
        length_units=length_units,
        angle_units="radians",
        **model_parameters,
    )


def path_metrics(path: LineString) -> tuple[float, float]:
    """Return path length and cumulative unsigned turning angle.

    Retained for reporting. The cumulative angle is no longer sufficient
    to determine pressure under the recursive model, which depends on
    where each bend sits along the path.
    """
    lengths, angles = segment_geometry(path)

    return float(lengths.sum()), float(angles.sum())


def pressure_profile(
    path: LineString,
    *,
    length_units: str = "ft",
    points_per_segment: int = 50,
    yield_pressure: float = YIELD_PRESSURE_PSI,
    length_friction: float = LENGTH_FRICTION_PSI_PER_FT,
    tail_tension: float = TAIL_TENSION_PSI,
    curvature_coefficient: float = CURVATURE_FRICTION_COEFFICIENT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative path length and required pressure samples.

    The profile is the pressure history of the deployment: the value at
    arclength ``s`` is the pressure required when the tip has reached
    ``s``, which is the recursion applied to the path truncated there.

    Under the recursive model the tip is always distal to every bend, so
    freshly deployed body adds drag at the constant rate ``f`` and the
    profile rises linearly within each segment. Rounding a vertex
    multiplies the accumulated tension by ``exp(mu_c * |t|)``, so the
    profile contains two pressure values at that same cumulative length
    to display the resulting jump. (Under the old summed model the slope
    itself grew after every turn; here it does not.)
    """
    length_to_feet = _length_scale(length_units)

    if not isinstance(points_per_segment, int) or points_per_segment < 1:
        raise ValueError("points_per_segment must be a positive integer.")

    coords = np.asarray(path.coords, dtype=float)

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

    segment_lengths, turning_angles = segment_geometry(coords)

    if np.any(segment_lengths == 0.0):
        raise ValueError(
            "A pressure profile cannot contain zero-length segments."
        )

    cumulative_length = 0.0
    tension = float(tail_tension)

    profile_lengths = []
    profile_pressures = []

    for segment_index, segment_length in enumerate(segment_lengths):
        if segment_index > 0:
            # Everything deployed so far is now proximal to this bend, so
            # the whole accumulated tension is amplified at once. The
            # previous segment already contributed the pre-turn pressure
            # at this cumulative length.
            tension *= np.exp(
                curvature_coefficient
                * turning_angles[segment_index - 1]
            )
            profile_lengths.append(cumulative_length)
            profile_pressures.append(yield_pressure + tension)

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
            yield_pressure
            + tension
            + length_friction * local_lengths * length_to_feet
        )

        profile_lengths.extend(sampled_lengths.tolist())
        profile_pressures.extend(sampled_pressures.tolist())

        tension += length_friction * segment_length * length_to_feet
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

# ------------------------------------------------------------------
# Standard obstacles
# ------------------------------------------------------------------

def sample_angles(count: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted vertex angles whose largest gap stays below pi.

    Independent uniform angles are the standard construction, but
    sorting them is not on its own enough to keep the polygon
    simple: once the largest gap between consecutive angles
    reaches pi, the chord that closes it passes on the far side of
    the center and can cut across the other edges. Holding every
    gap below pi confines each edge to its own angular wedge, and
    wedges cannot cross, so the polygon is simple by construction.
    """
    for _ in range(100):
        angles = np.sort(
            rng.uniform(0.0, 2.0 * np.pi, count)
        )

        gaps = np.diff(
            np.concatenate((angles, angles[:1] + 2.0 * np.pi))
        )

        if gaps.max() < np.pi:
            return angles

    # Unreachable in practice. Evenly spaced angles at a random
    # phase leave every gap at 2 pi / count, which is below pi for
    # any polygon with three or more vertices.
    return (
        2.0 * np.pi * np.arange(count) / count
        + rng.uniform(0.0, 2.0 * np.pi / count)
    )

def annulus_obstacle(
    center: np.ndarray,
    rng: np.random.Generator,
    min_vertices: int,
    max_vertices: int,
    min_radius: float,
    max_radius: float,
    polygons: list[Polygon],
    course_region: Polygon,
    *,
    forbidden_points=(),
    max_attempts: int = 5000,
    raise_on_failure: bool = True,

):
    """Sample one annulus obstacle entirely inside current free space.

    Angles and radii retain the original annulus construction. A draw
    is accepted only when every sampled vertex lies in the Shapely
    free-space set and the complete candidate polygon is covered by
    that same set. The candidate must also contain its nominal center,
    preserving the meaning of the supplied line-placement point as the
    obstacle center.

    ``forbidden_points`` are future line centers. Rejecting candidates
    that cover any of them prevents an earlier obstacle from consuming
    the center needed by a later obstacle.
    """
    center = np.asarray(center, dtype=float)
    free_space = shapely.difference(course_region, shapely.union_all(polygons))
    center_point = Point(center)
    forbidden_geometry = (
        shapely.points(np.asarray(forbidden_points, dtype=float))
        if len(forbidden_points)
        else None
    )

    if not shapely.covers(free_space, center_point):
        if raise_on_failure:
            raise RuntimeError(
                "The requested obstacle center is not in the current "
                "Shapely free-space region."
            )

        return None

    for _ in range(max_attempts):
        number_of_vertices = int(
            rng.integers(min_vertices, max_vertices + 1)
        )

        angles = sample_angles(number_of_vertices, rng)
        radii = rng.uniform(
            min_radius,
            max_radius,
            number_of_vertices,
        )
        vertices = (
            center
            + radii[:, None] * np.column_stack((
                np.cos(angles),
                np.sin(angles),
            ))
        )

        # Enforce the vertex-level requirement directly using the
        # Shapely set difference defining the current free space.
        vertex_points = shapely.points(vertices)
        if not np.all(shapely.covers(free_space, vertex_points)):
            continue

        candidate = Polygon(vertices)

        if (
            candidate.is_empty
            or not candidate.is_valid
            or candidate.area <= 0.0
            or not shapely.contains(candidate, center_point)
        ):
            continue

        # Vertices being free is not sufficient by itself: an edge can
        # cross an occupied polygon while both endpoints remain free.
        # Requiring the whole candidate to be covered by free space
        # guarantees zero positive-area overlap. Boundary contact is
        # allowed, but obstacle interiors remain disjoint.
        if not shapely.covers(free_space, candidate):
            continue

        if forbidden_geometry is not None and np.any(
            shapely.intersects(candidate, forbidden_geometry)
        ):
            continue

        return candidate

    if raise_on_failure:
        raise RuntimeError(
            "Could not sample a non-overlapping obstacle around the "
            "requested line center. Reduce num_obstacles or radius, "
            "or increase the course dimensions."
        )

    return None

# ------------------------------------------------------------------
# Clusters in the two corners the barriers do not reach
# ------------------------------------------------------------------

def corner_interval(
    target: float,
    low: float,
    high: float,
    span: float,
) -> tuple[float, float]:
    if target <= low:
        return low, min(low + span, high)

    return max(high - span, low), high

def clip_segment(
    start: np.ndarray,
    end: np.ndarray,
    inset_low_x: float,
    inset_low_y: float,
    inset_high_x: float,
    inset_high_y: float
):
    """Parameter range of ``start -> end`` inside the inset box.

    Liang-Barsky: walk the four slabs, tightening the parameter
    interval, and bail out as soon as it becomes empty. The shifted
    bridge segments begin near barrier vertices, so they generally
    need trimming to keep complete obstacles inside the course.
    """
    direction = end - start

    low = 0.0
    high = 1.0

    for numerator, denominator in (
        (start[0] - inset_low_x, -direction[0]),
        (inset_high_x - start[0], direction[0]),
        (start[1] - inset_low_y, -direction[1]),
        (inset_high_y - start[1], direction[1]),
    ):
        if denominator == 0.0:
            if numerator < 0.0:
                return None

            continue

        ratio = numerator / denominator

        if denominator < 0.0:
            if ratio > high:
                return None

            low = max(low, ratio)
        else:
            if ratio < low:
                return None

            high = min(high, ratio)

    if high <= low:
        return None

    return low, high

def free_parameter_intervals(
    start: np.ndarray,
    end: np.ndarray,
    span: tuple[float, float],
    polygons: list[Polygon]
) -> list[tuple[float, float]]:
    """Intervals whose shifted-line centers lie in current free space.

    The wall-clipped line is differenced from the current occupied
    polygon union with Shapely. Each remaining LineString component is
    then converted back to the original segment parameter.
    """
    low, high = span
    direction = end - start
    squared_length = float(np.dot(direction, direction))

    clipped_start = start + low * direction
    clipped_end = start + high * direction
    clipped_line = LineString((clipped_start, clipped_end))
    occupied_region = shapely.union_all(polygons)
    free_line = shapely.difference(clipped_line, occupied_region)
    intervals = []

    for part in shapely.get_parts(free_line):
        if part.geom_type != "LineString" or part.length <= 0.0:
            continue

        coordinates = np.asarray(part.coords, dtype=float)
        parameters = (
            (coordinates - start) @ direction / squared_length
        )
        interval_low = max(low, float(parameters.min()))
        interval_high = min(high, float(parameters.max()))

        if interval_high > interval_low:
            intervals.append((interval_low, interval_high))

    intervals.sort()
    return intervals

def intersect_intervals(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Pairwise intersections of two sorted interval collections."""
    intersections = []
    first_index = 0
    second_index = 0

    while first_index < len(first) and second_index < len(second):
        low = max(
            first[first_index][0],
            second[second_index][0],
        )
        high = min(
            first[first_index][1],
            second[second_index][1],
        )

        if high > low:
            intersections.append((low, high))

        if first[first_index][1] < second[second_index][1]:
            first_index += 1
        else:
            second_index += 1

    return intersections

def bridge_segment(path_vertices, obstacle_a_vertices, obstacle_b_vertices):
    """Return the path's unique direct edge from A to B.

    The returned edge is always oriented from obstacle A toward
    obstacle B. Paths without exactly one such edge are ignored.
    """
    crossings = []

    for first, second in itertools.pairwise(path_vertices):
        first_on_a = first in obstacle_a_vertices
        first_on_b = first in obstacle_b_vertices
        second_on_a = second in obstacle_a_vertices
        second_on_b = second in obstacle_b_vertices

        if first_on_a and second_on_b:
            crossings.append((first, second))
        elif first_on_b and second_on_a:
            crossings.append((second, first))

    if len(crossings) != 1:
        return None

    return crossings[0]
