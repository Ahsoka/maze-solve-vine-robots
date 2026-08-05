import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import itertools
import shapely

from .utils import grow_barrier
from .constants import _LENGTH_TO_FEET
from shapely.plotting import plot_polygon
from attrs import define, field, validators
from shapely import Polygon, Point, LineString


@define
class ObstacleCourse:
    width: float = field(default=25.0, converter=float, validator=[validators.instance_of(float), validators.gt(0)])
    height: float = field(default=25.0, converter=float, validator=[validators.instance_of(float), validators.gt(0)])
    obstacles: list[Polygon] = field(factory=list)

    obstacles_region = field(init=False)

    def __attrs_post_init__(self):
        self.obstacles_region = shapely.union_all(self.obstacles)
        shapely.prepare(self.obstacles_region)

    @classmethod
    def random(
        cls,
        width: float = 25,
        height: float = 25,
        num_obstacles: int = None,
        seed: int = None,
        min_obstacles: int = 15,
        max_obstacles: int = 20,
        radius: float = None,
        annulus_fraction: float = 0.005,
        wall_fraction: float = 0.1,
        corner_obstacles: int = 1,
        corner_span_fraction: float = 0.15,
        min_vertices: int = 3,
        max_vertices: int = 6,
    ):
        """Generate a random course anchored on two opposite corner barriers.

        Two barriers are built, one across the bottom-left quadrant and a
        mirrored one across the top-right quadrant. Splitting a quadrant in
        four again, its barrier is forced to reach into two opposite
        sub-quadrants, and in each of those the point is pushed hard
        against the outer wall: within ``wall_fraction`` of the course
        dimension. So the bottom-left barrier runs from the left wall to
        the floor, sealing off the origin, and the top-right barrier
        mirrors it.

        Every further barrier vertex is drawn from the quadrant with the
        current hull cut out of it, which puts the new point outside the
        hull by construction and so makes it a vertex of the grown hull.

        The remaining obstacles have vertices sampled the standard way, at
        a random angle and a random radius from a thin annulus. Their
        centers come from three places: near ``(0, height)``, near
        ``(width, 0)``, and along four lines derived from paths through the
        two corner barriers. A temporary two-obstacle course is solved from
        ``(0, 0)`` to ``(width, height)`` with ``PathPlanner``. The shortest
        path is retained first. The next retained path is the shortest path
        whose direct A-to-B segment shares neither its obstacle-A endpoint
        nor its obstacle-B endpoint with the first path's segment. Each
        direct segment
        is translated vertically by ``-radius / 2`` and ``+radius / 2``,
        producing four placement lines. The remaining obstacle count is
        divided between the two original direct segments in proportion to
        their usable segment lengths. For each original direct segment, the
        combined centers assigned to its two translated lines are then spaced
        evenly along the original segment. Those centerline points
        are then alternately projected vertically onto the lower and upper
        translated lines. This staggers neighboring obstacles instead of
        placing opposing obstacles at nearly the same longitudinal position.
        Together with the two barriers and the two corner clusters, all four
        corners are always occupied, so the course always holds at least
        four obstacles.

        Every standard obstacle is sampled against the current Shapely
        free-space geometry. Before an obstacle is accepted, the union of all
        previously generated polygons is subtracted from the complete course
        region with ``shapely.difference``. All sampled vertices and the full
        candidate polygon must be covered by that free-space region, so
        obstacle interiors cannot overlap.

        Parameters
        ----------
        radius:
            Inner sampling radius, ``min_radius``. Defaults to four
            percent of the smaller course dimension.
        annulus_fraction:
            ``max_radius`` is ``radius`` plus this fraction of
            ``min(width, height)``.
        wall_fraction:
            How close to the wall the two required barrier points are
            pushed, as a fraction of the course dimension.
        corner_obstacles:
            Obstacles placed at each of ``(0, height)`` and
            ``(width, 0)``.
        corner_span_fraction:
            Size of the box those centers are drawn from, as a fraction of
            the course dimensions.
        min_vertices, max_vertices:
            Bounds on the vertex count, inclusive.
        """

        rng = np.random.default_rng(seed)

        # ------------------------------------------------------------------
        # Validate arguments
        # ------------------------------------------------------------------

        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive.")

        if corner_obstacles < 0:
            raise ValueError("corner_obstacles must be nonnegative.")

        # Two barriers, plus a cluster at each of the two free corners.
        reserved_obstacles = 2 + 2 * corner_obstacles

        if num_obstacles is None:
            if min_obstacles < reserved_obstacles:
                raise ValueError(
                    f"min_obstacles must be at least {reserved_obstacles}: "
                    "two barriers plus the corner clusters."
                )

            if max_obstacles < min_obstacles:
                raise ValueError(
                    "max_obstacles must be greater than or equal to "
                    "min_obstacles."
                )

            num_obstacles = int(
                rng.integers(min_obstacles, max_obstacles + 1)
            )

        if num_obstacles < reserved_obstacles:
            raise ValueError(
                f"num_obstacles must be at least {reserved_obstacles}: "
                "two barriers plus the corner clusters."
            )

        if min_vertices < 3:
            raise ValueError("min_vertices must be at least 3.")

        if max_vertices < min_vertices:
            raise ValueError(
                "max_vertices must be greater than or equal to "
                "min_vertices."
            )

        if not 0.0 < wall_fraction < 0.25:
            raise ValueError(
                "wall_fraction must lie in (0, 0.25) so the strip stays "
                "inside its sub-quadrant."
            )

        if not 0.0 < corner_span_fraction <= 0.5:
            raise ValueError(
                "corner_span_fraction must lie in (0, 0.5]."
            )

        smaller_dimension = min(width, height)

        if radius is None:
            radius = 0.04 * smaller_dimension

        if radius <= 0:
            raise ValueError("radius must be positive.")

        min_radius = radius
        max_radius = min_radius + annulus_fraction * smaller_dimension

        # Nudges every sampled coordinate strictly inside the course, so
        # that 0 < x < width and 0 < y < height rather than 0 <= x <= width.
        epsilon = 1e-9 * smaller_dimension

        # Keeping every center at least max_radius from each wall puts the
        # whole sampling disk inside the course, which is what guarantees
        # that every vertex satisfies 0 < x < width and 0 < y < height.
        inset_low_x = max_radius + epsilon
        inset_low_y = max_radius + epsilon
        inset_high_x = width - max_radius - epsilon
        inset_high_y = height - max_radius - epsilon

        if inset_high_x <= inset_low_x or inset_high_y <= inset_low_y:
            raise ValueError(
                "radius is too large for the course: no obstacle center "
                "can sit far enough from the walls to keep its vertices "
                "inside."
            )

        half_width = 0.5 * width
        half_height = 0.5 * height
        wall_x = wall_fraction * width
        wall_y = wall_fraction * height

        # Bottom-left: reaches the left wall in the top-left sub-quadrant
        # and the floor in the bottom-right sub-quadrant.
        origin_barrier = grow_barrier(
            (0.0, 0.0, half_width, half_height),
            (0.0, 0.25 * height, wall_x, half_height),
            (0.25 * width, 0.0, half_width, wall_y),
            rng, min_vertices, max_vertices, epsilon
        )

        # Top-right: the same construction rotated half a turn, so it
        # reaches the right wall and the ceiling.
        far_barrier = grow_barrier(
            (half_width, half_height, width, height),
            (width - wall_x, half_height, width, 0.75 * height),
            (half_width, height - wall_y, 0.75 * width, height),
            rng, min_vertices, max_vertices, epsilon
        )

        polygons = [
            Polygon(origin_barrier),
            Polygon(far_barrier),
        ]

        # ------------------------------------------------------------------
        # Standard obstacles
        # ------------------------------------------------------------------

        def sample_angles(count: int) -> np.ndarray:
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

        course_region = shapely.box(0.0, 0.0, width, height)

        def current_free_space():
            """Course region minus every polygon generated so far.

            The subtraction intentionally uses the vectorized Shapely
            functions rather than manual coordinate tests. Recomputing this
            region before each new obstacle makes the acceptance test account
            for every polygon already appended to ``polygons``.
            """
            occupied_region = shapely.union_all(polygons)
            return shapely.difference(course_region, occupied_region)

        def annulus_obstacle(
            center: np.ndarray,
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
            free_space = current_free_space()
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

                angles = sample_angles(number_of_vertices)
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

        span_x = corner_span_fraction * width
        span_y = corner_span_fraction * height

        for corner_x, corner_y in ((0.0, height), (width, 0.0)):
            low_x, high_x = corner_interval(
                corner_x,
                inset_low_x,
                inset_high_x,
                span_x,
            )
            low_y, high_y = corner_interval(
                corner_y,
                inset_low_y,
                inset_high_y,
                span_y,
            )

            for _ in range(corner_obstacles):
                # Keep the same corner box and annulus construction, but retry
                # the random center when a draw cannot fit inside the current
                # Shapely free-space set. This matters when more than one
                # obstacle is requested in a corner cluster.
                for _center_attempt in range(2000):
                    center = np.array([
                        rng.uniform(low_x, high_x),
                        rng.uniform(low_y, high_y),
                    ])
                    obstacle = annulus_obstacle(
                        center,
                        raise_on_failure=False,
                    )

                    if obstacle is not None:
                        polygons.append(obstacle)
                        break
                else:
                    raise RuntimeError(
                        "Could not place a non-overlapping obstacle in a "
                        "corner cluster. Reduce corner_obstacles or radius, "
                        "or increase corner_span_fraction."
                    )

        # ------------------------------------------------------------------
        # The rest: centers along offsets of two shortest-path bridge edges
        # ------------------------------------------------------------------

        def clip_segment(start: np.ndarray, end: np.ndarray):
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

        remaining = num_obstacles - reserved_obstacles

        if remaining > 0:
            from .path_planner import PathPlanner
            # Solve the simplified problem using only obstacle A (the
            # bottom-left barrier) and obstacle B (the top-right barrier).
            simplified_course = cls(
                width=width,
                height=height,
                obstacles=polygons[:2].copy(),
            )
            simplified_planner = PathPlanner(
                obstacle_course=simplified_course,
                start=np.array([0.0, 0.0]),
                end=np.array([width, height]),
            )

            start_node = tuple(
                np.asarray(simplified_planner.start, dtype=float)
            )
            end_node = tuple(
                np.asarray(simplified_planner.end, dtype=float)
            )

            obstacle_a_vertices = set(
                simplified_course.obstacles[0].exterior.coords[:-1]
            )
            obstacle_b_vertices = set(
                simplified_course.obstacles[1].exterior.coords[:-1]
            )

            def bridge_segment(path_vertices):
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

            # nx.shortest_simple_paths yields paths in nondecreasing total
            # visibility-graph length. The first usable path is therefore the
            # shortest path. After fixing its bridge (a_i, b_j), continue in
            # the same ordered stream and retain the first path whose bridge
            # (a_k, b_l) satisfies both a_k != a_i and b_l != b_j. Thus the
            # second selected path is the shortest one obeying the endpoint-
            # disjointness requirement, rather than merely using a different
            # A-to-B edge.
            chosen_bridges = []
            first_a_vertex = None
            first_b_vertex = None

            try:
                candidate_paths = nx.shortest_simple_paths(
                    simplified_planner.graph,
                    source=start_node,
                    target=end_node,
                    weight="weight",
                )

                for path_vertices in candidate_paths:
                    bridge = bridge_segment(path_vertices)

                    if bridge is None:
                        continue

                    a_vertex, b_vertex = bridge

                    if not chosen_bridges:
                        chosen_bridges.append(bridge)
                        first_a_vertex = a_vertex
                        first_b_vertex = b_vertex
                        continue

                    if (
                        a_vertex == first_a_vertex
                        or b_vertex == first_b_vertex
                    ):
                        continue

                    chosen_bridges.append(bridge)
                    break

            except nx.NetworkXNoPath as error:
                raise RuntimeError(
                    "The simplified two-obstacle course has no path from "
                    "(0, 0) to (width, height)."
                ) from error

            if len(chosen_bridges) < 2:
                raise RuntimeError(
                    "Could not find a second path whose A-to-B segment uses "
                    "a different obstacle-A vertex and a different "
                    "obstacle-B vertex from the shortest path."
                )

            # Translating y by +/- radius/2 implements
            # y = m*x + b +/- radius/2 exactly as requested.
            vertical_offsets = (
                np.array([0.0, -1 * radius]),
                np.array([0.0, 1 * radius]),
            )
            placement_pairs = []

            for bridge_start, bridge_end in chosen_bridges:
                bridge_start = np.asarray(bridge_start, dtype=float)
                bridge_end = np.asarray(bridge_end, dtype=float)

                shifted_segments = []

                for vertical_offset in vertical_offsets:
                    shifted_start = bridge_start + vertical_offset
                    shifted_end = bridge_end + vertical_offset
                    span = clip_segment(shifted_start, shifted_end)

                    if span is None:
                        raise RuntimeError(
                            "A shifted shortest-path line does not leave "
                            "enough room for an obstacle center. Reduce "
                            "radius."
                        )

                    free_intervals = free_parameter_intervals(
                        shifted_start,
                        shifted_end,
                        span,
                    )

                    if not free_intervals:
                        raise RuntimeError(
                            "A shifted shortest-path line has no center points "
                            "in the current Shapely free-space region."
                        )

                    shifted_segments.append((
                        shifted_start,
                        shifted_end,
                        span,
                        free_intervals,
                    ))

                # Use a common interval on which either vertical projection is
                # a valid center. Shapely line-minus-polygon differences remove
                # portions lying inside the barriers or corner obstacles. When
                # several components remain, retain the longest common one so
                # the centers still form one evenly spaced sequence.
                common_intervals = intersect_intervals(
                    shifted_segments[0][3],
                    shifted_segments[1][3],
                )

                if not common_intervals:
                    raise RuntimeError(
                        "The two shifted shortest-path lines have no common "
                        "free-space interval for obstacle centers."
                    )

                common_low, common_high = max(
                    common_intervals,
                    key=lambda interval: interval[1] - interval[0],
                )

                placement_pairs.append((
                    bridge_start,
                    bridge_end,
                    shifted_segments,
                    (common_low, common_high),
                ))

            # Allocate obstacles between the two original A-to-B segments
            # in proportion to their usable lengths. The usable length is the
            # portion common to both vertically shifted copies, because every
            # center in the staggered sequence must be projectable to either
            # line. Largest-remainder rounding preserves the requested total.
            usable_pair_lengths = np.asarray(
                [
                    float(np.linalg.norm(end - start)) * (high - low)
                    for start, end, _, (low, high) in placement_pairs
                ],
                dtype=float,
            )

            if (
                np.any(usable_pair_lengths <= 0.0)
                or usable_pair_lengths.sum() <= 0.0
            ):
                raise RuntimeError(
                    "The selected shortest-path segments have no usable "
                    "length for obstacle placement."
                )

            exact_pair_counts = (
                remaining
                * usable_pair_lengths
                / usable_pair_lengths.sum()
            )
            pair_counts = np.floor(exact_pair_counts).astype(int)
            leftover = remaining - int(pair_counts.sum())

            if leftover:
                remainders = exact_pair_counts - pair_counts
                # The random secondary key makes exact ties unbiased while
                # keeping larger fractional remainders ahead of smaller ones.
                order = np.lexsort((rng.random(len(remainders)), -remainders))
                pair_counts[order[:leftover]] += 1

            placement_groups = []

            for pair_index, (
                bridge_start,
                bridge_end,
                shifted_segments,
                (low, high),
            ) in enumerate(placement_pairs):
                pair_count = int(pair_counts[pair_index])

                # Split this segment's proportional count as evenly as
                # possible between its lower and upper shifted lines.
                line_counts = np.full(2, pair_count // 2, dtype=int)
                if pair_count % 2:
                    line_counts[int(rng.integers(0, 2))] += 1

                if pair_count == 0:
                    continue

                placement_groups.append((
                    bridge_start,
                    bridge_end,
                    low,
                    high,
                    line_counts,
                    pair_count,
                ))

            # A locally valid early shape can still consume space needed to
            # form a later polygon. Use bounded group-level backtracking. Each
            # attempt preserves the proportional counts and exact equal spacing
            # along every original segment, but may change the common phase of
            # that arithmetic sequence within its usable interval. This avoids
            # pathological alignments between the two path-derived line pairs.
            fixed_polygon_count = len(polygons)

            for _layout_attempt in range(200):
                del polygons[fixed_polygon_count:]
                line_centers = []

                for (
                    bridge_start,
                    bridge_end,
                    low,
                    high,
                    line_counts,
                    pair_count,
                ) in placement_groups:
                    interval_length = high - low
                    spacing = interval_length / pair_count

                    # Any phase in [0, spacing] keeps all points inside the
                    # interval and preserves exact equal spacing. Stay slightly
                    # away from the interval endpoints to avoid numerical
                    # contact with an occupied boundary.
                    phase = rng.uniform(0.1, 0.9) * spacing
                    fractions = (
                        low
                        + phase
                        + np.arange(pair_count) * spacing
                    )

                    if line_counts[0] > line_counts[1]:
                        first_line = 0
                    elif line_counts[1] > line_counts[0]:
                        first_line = 1
                    else:
                        first_line = int(rng.integers(0, 2))

                    line_sequence = np.array(
                        [
                            first_line if index % 2 == 0 else 1 - first_line
                            for index in range(pair_count)
                        ],
                        dtype=int,
                    )

                    actual_counts = np.bincount(
                        line_sequence,
                        minlength=2,
                    )
                    if not np.array_equal(actual_counts, line_counts):
                        raise RuntimeError(
                            "Internal obstacle-line allocation error."
                        )

                    bridge_direction = bridge_end - bridge_start

                    for fraction, line_index in zip(
                        fractions,
                        line_sequence,
                        strict=True,
                    ):
                        original_center = (
                            bridge_start + fraction * bridge_direction
                        )
                        line_centers.append(
                            original_center + vertical_offsets[line_index]
                        )

                layout_succeeded = True

                for center_index, center in enumerate(line_centers):
                    obstacle = annulus_obstacle(
                        center,
                        forbidden_points=line_centers[center_index + 1:],
                        max_attempts=800,
                        raise_on_failure=False,
                    )

                    if obstacle is None:
                        layout_succeeded = False
                        break

                    polygons.append(obstacle)

                if layout_succeeded:
                    break
            else:
                del polygons[fixed_polygon_count:]
                raise RuntimeError(
                    "Could not generate a complete non-overlapping layout "
                    "after repeated Shapely free-space resampling and evenly "
                    "spaced center-phase adjustments. Reduce num_obstacles or "
                    "radius, or increase the course dimensions."
                )

        return cls(
            width=width,
            height=height,
            obstacles=polygons,
        )

    def plot(
        self,
        ax: plt.Axes = None,
        show_vertices: bool = True,
        show_labels: bool = False,
        title: str = None,
        show: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the obstacle course."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure

        for index, obstacle in enumerate(self.obstacles or [], start=1):
            if obstacle is None or obstacle.is_empty:
                continue

            plot_polygon(
                obstacle,
                ax=ax,
                add_points=show_vertices,
                facecolor="tab:blue",
                edgecolor="black",
                linewidth=1.5,
                alpha=0.4,
            )

            if show_labels:
                # Unlike the centroid, representative_point() is guaranteed
                # to lie within the polygon.
                label_point = obstacle.representative_point()
                ax.text(
                    label_point.x,
                    label_point.y,
                    str(index),
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontweight="bold",
                )

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")#, zorder=1)
        ax.set_ylabel("y")#, zorder=1)
        ax.set_title(title or "Obstacle Course")
        ax.grid(alpha=0.25)

        if show:
            plt.show()

        return fig, ax
