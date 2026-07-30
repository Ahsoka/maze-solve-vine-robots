import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import itertools
import shapely

from attrs import define, field, validators
from shapely.plotting import plot_polygon
from shapely import Polygon, Point, LineString


YIELD_PRESSURE_PSI = 0.278
LENGTH_FRICTION_PSI_PER_FT = 0.0341
TAIL_TENSION_PSI = 1.207
CURVATURE_FRICTION_COEFFICIENT = 0.234

_LENGTH_TO_FEET = {
    "ft": 1.0,
    "in": 1.0 / 12.0,
    "m": 3.280839895013123,
    "cm": 0.03280839895013123,
}


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
        width: float = 25.0,
        height: float = 25.0,
        num_obstacles: int = None,
        seed: int = None,
        min_obstacles: int = 3,
        max_obstacles: int = 8,
        min_radius: float = 0.5,
        max_radius: float = None,
        max_vertices: int = 10,
    ):
        rng = np.random.default_rng(seed)

        # ------------------------------------------------------------------
        # Validate arguments
        # ------------------------------------------------------------------

        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive.")

        if max_radius is None:
            max_radius = 0.6 * min(width, height)

        if num_obstacles is None:
            if min_obstacles < 0:
                raise ValueError("min_obstacles must be nonnegative.")

            if max_obstacles < min_obstacles:
                raise ValueError(
                    "max_obstacles must be greater than or equal to "
                    "min_obstacles."
                )

            num_obstacles = int(
                rng.integers(min_obstacles, max_obstacles + 1)
            )

        if num_obstacles < 0:
            raise ValueError("num_obstacles must be nonnegative.")

        if max_vertices < 3:
            raise ValueError("max_vertices must be at least 3.")

        if min_radius <= 0:
            raise ValueError("min_radius must be positive.")

        if max_radius <= min_radius:
            raise ValueError(
                "max_radius must be greater than min_radius."
            )

        # The obstacle center is sampled at least min_radius from every
        # course boundary. This guarantees that the inner disk fits in
        # the course and that the clipped annulus has positive area.
        if 2.0 * min_radius > width:
            raise ValueError(
                "min_radius is too large for the course width."
            )

        if 2.0 * min_radius > height:
            raise ValueError(
                "min_radius is too large for the course height."
            )

        # ------------------------------------------------------------------
        # Uniform point sampling from an arbitrary polygonal region
        # ------------------------------------------------------------------

        def sample_uniform_points(
            region,
            number_of_points: int,
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

        # ------------------------------------------------------------------
        # Generate obstacles
        # ------------------------------------------------------------------

        course_region = shapely.box(
            0.0,
            0.0,
            width,
            height,
        )

        # Make the lower outer-radius bound strictly larger than
        # min_radius.
        minimum_outer_radius = np.nextafter(
            float(min_radius),
            float(max_radius),
        )

        polygons = []

        for obstacle_index in range(num_obstacles):
            center_point = np.array(
                [
                    rng.uniform(
                        min_radius,
                        width - min_radius,
                    ),
                    rng.uniform(
                        min_radius,
                        height - min_radius,
                    ),
                ],
                dtype=float,
            )

            number_of_vertices = int(
                rng.integers(3, max_vertices + 1)
            )

            outer_radius = float(
                rng.uniform(
                    minimum_outer_radius,
                    max_radius,
                )
            )

            center = Point(
                float(center_point[0]),
                float(center_point[1]),
            )

            outer_circle = center.buffer(
                outer_radius,
                quad_segs=32,
            )

            inner_circle = center.buffer(
                min_radius,
                quad_segs=32,
            )

            # Valid vertex region:
            #
            # (outer circle ∩ obstacle course) − inner circle
            sampling_region = (
                outer_circle
                .intersection(course_region)
                .difference(inner_circle)
            )

            # This condition should hold because of the center and radius
            # constraints above. It remains as a defensive sanity check.
            if (
                sampling_region.is_empty
                or sampling_region.area <= 0
            ):
                raise RuntimeError(
                    "The obstacle sampling region was unexpectedly empty."
                )

            coords = sample_uniform_points(
                sampling_region,
                number_of_vertices,
            )

            # Sorting around the sampled points' centroid produces a
            # consistently ordered polygon, including when the outer
            # circle has been clipped by the course boundary.
            sorting_center = coords.mean(axis=0)

            angles = np.arctan2(
                coords[:, 1] - sorting_center[1],
                coords[:, 0] - sorting_center[0],
            )

            coords = coords[np.argsort(angles)]

            polygon = Polygon(coords)

            if not polygon.is_valid:
                raise RuntimeError(
                    f"Generated obstacle {obstacle_index + 1} "
                    "was invalid."
                )

            if polygon.area <= 0:
                raise RuntimeError(
                    f"Generated obstacle {obstacle_index + 1} "
                    "had zero area."
                )

            if not course_region.covers(polygon):
                raise RuntimeError(
                    f"Generated obstacle {obstacle_index + 1} "
                    "extended outside the obstacle course."
                )

            polygons.append(polygon)

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


@define
class PathPlanner:
    obstacle_course: ObstacleCourse

    start: np.ndarray | None = field(default=None)
    end: np.ndarray | None = field(default=None)

    graph: nx.Graph = field(init=False)
    line_graph: nx.Graph = field(init=False)

    angle_path: LineString | None = field(init=False, default=None)
    total_angle: float | None = field(init=False, default=None)

    distance_path: LineString | None = field(init=False, default=None)

    def __attrs_post_init__(self):
        self._goal_generator()
        self.create_graph()
        self.create_line_graph()

    @start.validator
    @end.validator
    def _point_validator(self, attribute, value):
        # None means that the point will be generated automatically.
        if value is None:
            return

        value = np.asarray(value, dtype=float)

        if value.shape != (2,):
            raise ValueError(
                f"{attribute.name} must contain exactly two coordinates."
            )

        point = Point(
            float(value[0]),
            float(value[1]),
        )

        course_region = shapely.box(
            0.0,
            0.0,
            self.obstacle_course.width,
            self.obstacle_course.height,
        )

        # covers() includes the course boundary.
        if not course_region.covers(point):
            raise ValueError(
                f"{attribute.name} is outside the obstacle course."
            )

        # contains() excludes the obstacle boundary, so points exactly
        # on an obstacle boundary remain valid.
        if any(
            obstacle.contains(point)
            for obstacle in self.obstacle_course.obstacles or []
        ):
            raise ValueError(
                f"{attribute.name} is inside an obstacle."
            )

    def _goal_generator(self) -> None:
        """
        Generate missing start/end points with maximum straight-line
        separation inside the valid region.

        If both points are missing, select the farthest pair.

        If only one point is missing, select the valid point farthest
        from the supplied point.
        """
        if self.start is not None and self.end is not None:
            return

        course_region = shapely.box(
            0.0,
            0.0,
            self.obstacle_course.width,
            self.obstacle_course.height,
        )

        obstacles = self.obstacle_course.obstacles or []

        if obstacles:
            blocked_region = self.obstacle_course.obstacles_region

            valid_region = shapely.difference(
                course_region,
                blocked_region,
            )
        else:
            valid_region = course_region

        if valid_region.is_empty or valid_region.area <= 0:
            raise RuntimeError(
                "The obstacle course has no valid area for start "
                "and end points."
            )

        # The farthest pair of points in a polygonal region can be found
        # among the vertices of its convex hull.
        convex_hull = valid_region.convex_hull

        if convex_hull.geom_type != "Polygon":
            raise RuntimeError(
                "The valid region does not contain enough area to "
                "generate distinct start and end points."
            )

        # The final exterior coordinate repeats the first coordinate,
        # so remove it.
        candidates = np.asarray(
            convex_hull.exterior.coords,
            dtype=float,
        )[:-1, :2]

        if len(candidates) < 2:
            raise RuntimeError(
                "Not enough valid candidate points were found."
            )

        if self.start is None and self.end is None:
            # Calculate the squared distance between every pair of hull
            # vertices. Squared distance is sufficient because sqrt()
            # does not change which pair is largest.
            differences = (
                candidates[:, np.newaxis, :]
                - candidates[np.newaxis, :, :]
            )

            squared_distances = np.sum(
                differences**2,
                axis=2,
            )

            start_index, end_index = np.unravel_index(
                np.argmax(squared_distances),
                squared_distances.shape,
            )

            self.start = candidates[start_index].copy()
            self.end = candidates[end_index].copy()

        elif self.start is None:
            fixed_end = np.asarray(
                self.end,
                dtype=float,
            )

            squared_distances = np.sum(
                (candidates - fixed_end) ** 2,
                axis=1,
            )

            self.start = candidates[
                np.argmax(squared_distances)
            ].copy()

        else:
            fixed_start = np.asarray(
                self.start,
                dtype=float,
            )

            squared_distances = np.sum(
                (candidates - fixed_start) ** 2,
                axis=1,
            )

            self.end = candidates[
                np.argmax(squared_distances)
            ].copy()

    def create_graph(self):
        self.graph = nx.Graph()

        # Add all points not in the interior of any polygon
        obstacle_region = self.obstacle_course.obstacles_region
        vertices = [tuple(self.start), tuple(self.end)]
        for point in itertools.chain.from_iterable(
            map(lambda obstacle: obstacle.exterior.coords[:-1], self.obstacle_course.obstacles)
        ):
            if not shapely.contains_xy(obstacle_region, *point):
                vertices.append(point)
                self.graph.add_node(point)

        # Construct all visible edges
        for edge in itertools.combinations(vertices, 2):
            line = shapely.LineString(edge)
            if obstacle_region.disjoint(line) or obstacle_region.touches(line):
                self.graph.add_edge(*edge, weight=line.length)

    def create_line_graph(self):
        self.line_graph = nx.line_graph(self.graph)

        # Each node in the line graph represents an edge in the
        # original visibility graph.
        line_graph_edges = list(self.line_graph.edges)

        if line_graph_edges:
            previous_points = []
            shared_points = []
            next_points = []

            # Extract the three original-graph points associated with
            # every transition between two adjacent edges.
            for original_edge_1, original_edge_2 in line_graph_edges:
                if original_edge_1[0] in original_edge_2:
                    shared = original_edge_1[0]
                    previous = original_edge_1[1]
                elif original_edge_1[1] in original_edge_2:
                    shared = original_edge_1[1]
                    previous = original_edge_1[0]
                else:
                    raise RuntimeError(
                        "Adjacent line-graph nodes do not share an "
                        "original-graph vertex."
                    )

                next_point = (
                    original_edge_2[1]
                    if original_edge_2[0] == shared
                    else original_edge_2[0]
                )

                previous_points.append(previous)
                shared_points.append(shared)
                next_points.append(next_point)

            previous_points = np.asarray(
                previous_points,
                dtype=float,
            )
            shared_points = np.asarray(
                shared_points,
                dtype=float,
            )
            next_points = np.asarray(
                next_points,
                dtype=float,
            )

            # Direction of travel into and out of the shared vertex.
            incoming_vectors = shared_points - previous_points
            outgoing_vectors = next_points - shared_points

            incoming_lengths = np.linalg.norm(
                incoming_vectors,
                axis=1,
            )
            outgoing_lengths = np.linalg.norm(
                outgoing_vectors,
                axis=1,
            )

            if (
                np.any(incoming_lengths == 0)
                or np.any(outgoing_lengths == 0)
            ):
                raise RuntimeError(
                    "Cannot calculate a turning angle for a "
                    "zero-length edge."
                )

            # Vectorized cosine calculation for every turn.
            dot_products = np.einsum(
                "ij,ij->i",
                incoming_vectors,
                outgoing_vectors,
            )

            cosine_angles = dot_products / (
                incoming_lengths * outgoing_lengths
            )

            # Protect arccos from small floating-point errors such as
            # 1.0000000000000002.
            cosine_angles = np.clip(
                cosine_angles,
                -1.0,
                1.0,
            )

            turning_angles = np.arccos(cosine_angles)

            # Assign the precomputed turning costs.
            for line_edge, angle in zip(
                line_graph_edges,
                turning_angles,
                strict=True,
            ):
                self.line_graph.edges[line_edge]["weight"] = float(
                    angle
                )

        # --------------------------------------------------------------
        # Add terminal nodes
        # --------------------------------------------------------------

        start_node = "start"
        end_node = "end"

        original_start = tuple(
            np.asarray(self.start, dtype=float)
        )
        original_end = tuple(
            np.asarray(self.end, dtype=float)
        )

        self.line_graph.add_node(start_node)
        self.line_graph.add_node(end_node)

        # Save this list before adding terminal connections. Every node
        # here represents an original visibility-graph edge.
        original_line_nodes = [
            node
            for node in self.line_graph.nodes
            if node not in {start_node, end_node}
        ]

        for line_node in original_line_nodes:
            # line_node is an original graph edge: (point_a, point_b).
            if original_start in line_node:
                self.line_graph.add_edge(
                    start_node,
                    line_node,
                    weight=0.0,
                )

            if original_end in line_node:
                self.line_graph.add_edge(
                    line_node,
                    end_node,
                    weight=0.0,
                )

        return self.line_graph

    @staticmethod
    def _path_metrics(path: LineString) -> tuple[float, float]:
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

    def compute_distance_path(self) -> LineString:
        """Compute and store the minimum-distance visibility-graph path."""
        start = tuple(np.asarray(self.start, dtype=float))
        end = tuple(np.asarray(self.end, dtype=float))

        try:
            path_vertices = nx.shortest_path(
                self.graph,
                source=start,
                target=end,
                weight="weight",
            )
        except nx.NetworkXNoPath as error:
            raise RuntimeError(
                "No collision-free distance path exists between start and end."
            ) from error

        self.distance_path = LineString(path_vertices)
        return self.distance_path

    def compute_angle_path(self) -> LineString:
        """Compute and store the minimum-cumulative-turn path."""
        try:
            line_path = nx.shortest_path(
                self.line_graph,
                source="start",
                target="end",
                weight="weight",
            )
        except nx.NetworkXNoPath as error:
            raise RuntimeError(
                "No collision-free angle path exists between start and end."
            ) from error

        original_edges = line_path[1:-1]
        start = tuple(np.asarray(self.start, dtype=float))
        end = tuple(np.asarray(self.end, dtype=float))

        path_vertices = [start]
        current = start

        for edge in original_edges:
            point_a, point_b = edge

            if current == point_a:
                current = point_b
            elif current == point_b:
                current = point_a
            else:
                raise RuntimeError(
                    "The line-graph path could not be reconstructed as "
                    "a continuous path in the original graph."
                )

            path_vertices.append(current)

        if current != end:
            raise RuntimeError(
                "The reconstructed angle path does not terminate at the goal."
            )

        self.angle_path = LineString(path_vertices)
        self.total_angle = float(
            nx.path_weight(
                self.line_graph,
                line_path,
                weight="weight",
            )
        )
        return self.angle_path

    @staticmethod
    def _pressure_profile(
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

    def plot_pressure(
        self,
        ax: plt.Axes = None,
        *,
        length_units: str = "ft",
        points_per_segment: int = 50,
        title: str = None,
        show: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot required vine-robot pressure versus deployed path length.

        Any computed distance and angle paths are plotted on the same axes.
        Raises ``RuntimeError`` when neither path has been computed.
        """
        if self.distance_path is None and self.angle_path is None:
            raise RuntimeError(
                "No path has been computed. Call compute_distance_path() "
                "or compute_angle_path() before plot_pressure()."
            )

        normalized_length_units = length_units.lower()
        if normalized_length_units not in _LENGTH_TO_FEET:
            raise ValueError(
                "length_units must be one of: 'ft', 'in', 'm', or 'cm'."
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        if self.distance_path is not None:
            lengths, pressures = self._pressure_profile(
                self.distance_path,
                length_units=normalized_length_units,
                points_per_segment=points_per_segment,
            )
            ax.plot(
                lengths,
                pressures,
                linewidth=2.5,
                color="tab:red",
                label=(
                    "min distance "
                    f"(length={lengths[-1]:.2f} {normalized_length_units}, "
                    f"final pressure={pressures[-1]:.3f} psi)"
                ),
            )

        if self.angle_path is not None:
            lengths, pressures = self._pressure_profile(
                self.angle_path,
                length_units=normalized_length_units,
                points_per_segment=points_per_segment,
            )
            ax.plot(
                lengths,
                pressures,
                linewidth=2.5,
                linestyle="--",
                color="tab:orange",
                label=(
                    "min angle "
                    f"(length={lengths[-1]:.2f} {normalized_length_units}, "
                    f"final pressure={pressures[-1]:.3f} psi)"
                ),
            )

        ax.set_xlabel(f"Path length ({normalized_length_units})")
        ax.set_ylabel("Pressure (psi)")
        ax.set_title(title or "Vine Robot Pressure vs. Path Length")
        ax.grid(alpha=0.25)
        ax.legend()

        if show:
            plt.show()

        return fig, ax

    def plot(
        self,
        ax: plt.Axes = None,
        show_vertices: bool = True,
        show_labels: bool = False,
        title: str = None,
        show: bool = True,
        length_units: str = "ft",
        angle_units: str = "degrees",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the course, goals, and any computed paths.

        Path coordinates are interpreted as ``length_units``. Angles are
        displayed in degrees by default; pass ``angle_units="radians"``
        to report them in radians. Pressure is always reported in psi.
        """
        normalized_length_units = length_units.lower()
        if normalized_length_units not in _LENGTH_TO_FEET:
            raise ValueError(
                "length_units must be one of: 'ft', 'in', 'm', or 'cm'."
            )

        normalized_angle_units = angle_units.lower()
        if normalized_angle_units in {"degree", "degrees", "deg"}:
            angle_scale = 180.0 / np.pi
            angle_label = "deg"
        elif normalized_angle_units in {"radian", "radians", "rad"}:
            angle_scale = 1.0
            angle_label = "rad"
        else:
            raise ValueError(
                "angle_units must be 'degrees' or 'radians'."
            )
        fig, ax = self.obstacle_course.plot(
            ax=ax,
            show_vertices=show_vertices,
            show_labels=show_labels,
            title=title,
            show=False,
        )

        ax.scatter(
            self.start[0],
            self.start[1],
            s=80,
            marker="o",
            color="green",
            label="start",
            clip_on=False,
            zorder=5,
        )
        ax.scatter(
            self.end[0],
            self.end[1],
            s=120,
            marker="*",
            color="purple",
            label="target",
            clip_on=False,
            zorder=5,
        )

        if self.distance_path is not None:
            distance_coords = np.asarray(
                self.distance_path.coords,
                dtype=float,
            )
            distance_length, distance_angle_radians = self._path_metrics(
                self.distance_path
            )
            distance_angle_display = (
                distance_angle_radians * angle_scale
            )
            distance_pressure = vine_robot_pressure(
                distance_length,
                distance_angle_radians,
                length_units=normalized_length_units,
                angle_units="radians",
            )

            ax.plot(
                distance_coords[:, 0],
                distance_coords[:, 1],
                linewidth=2.5,
                color="tab:red",
                label=(
                    "min distance "
                    f"(length={distance_length:.2f} {normalized_length_units}, "
                    f"angle={distance_angle_display:.2f} {angle_label}, "
                    f"pressure={distance_pressure:.3f} psi)"
                ),
                zorder=4,
            )

        if self.angle_path is not None:
            angle_coords = np.asarray(
                self.angle_path.coords,
                dtype=float,
            )
            angle_length, measured_angle_radians = self._path_metrics(
                self.angle_path
            )
            angle_cost_radians = (
                measured_angle_radians
                if self.total_angle is None
                else self.total_angle
            )
            angle_cost_display = angle_cost_radians * angle_scale
            angle_pressure = vine_robot_pressure(
                angle_length,
                angle_cost_radians,
                length_units=normalized_length_units,
                angle_units="radians",
            )

            ax.plot(
                angle_coords[:, 0],
                angle_coords[:, 1],
                linewidth=2.5,
                linestyle="--",
                color="tab:orange",
                label=(
                    "min angle "
                    f"(length={angle_length:.2f} {normalized_length_units}, "
                    f"angle={angle_cost_display:.2f} {angle_label}, "
                    f"pressure={angle_pressure:.3f} psi)"
                ),
                zorder=4,
            )

        ax.legend()

        if show:
            plt.show()

        return fig, ax
