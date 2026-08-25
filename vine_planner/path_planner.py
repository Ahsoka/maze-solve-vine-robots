import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import itertools
import shapely

from typing import overload
from attrs import define, field
from .constants import _LENGTH_TO_FEET
from shapely import LineString
from .obstacle_course import ObstacleCourse, ObstacleCourseVoxels
from .utils import (
    pressure_profile,
    path_metrics
)


@define
class PathPlanner:
    obstacle_course: ObstacleCourse | ObstacleCourseVoxels

    start: np.ndarray | None = field(default=None)
    end: np.ndarray | None = field(default=None)

    graph: nx.Graph = field(init=False)
    line_graph: nx.DiGraph = field(init=False)

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

        # The course decides how many coordinates a point carries, so a new
        # course class needs no change here.
        dimension = self.obstacle_course.dimension

        if value.shape != (dimension,):
            raise ValueError(
                f"{attribute.name} must contain exactly {dimension} coordinates."
            )

        if not self.obstacle_course.valid_goal(value):
            raise ValueError(
                f"{attribute.name} is outside the obstacle course or inside an obstacle."
            )

    def _goal_generator(self) -> None:
        """Fill in whichever of start/end was not supplied.

        Where the goals go depends entirely on the geometry, so each course
        class owns that choice and this is only the hand-off.
        """
        self.start, self.end = self.obstacle_course.generate_goals(
            self.start,
            self.end,
        )

    def create_graph(self):
        self.graph = nx.Graph()
        vertices = self.obstacle_course.vertices()
        vertices.extend([tuple(self.start), tuple(self.end)])
        self.graph.add_nodes_from(vertices)

        for edge in itertools.combinations(vertices, 2):
            if self.obstacle_course.is_visible(*edge):
                self.graph.add_edge(
                    *edge,
                    weight=np.linalg.norm(np.array(edge[0]) - np.array(edge[1]))
                )

    def create_line_graph(self):
        """Build the *directed* line graph of the visibility graph.

        A node is an ordered pair ``(u, v)``, read as "the robot
        traverses the visibility edge ``{u, v}`` from ``u`` toward
        ``v``". An arc joins ``(u, v)`` to ``(v, w)`` and carries the
        unsigned turning angle at ``v``. Reversals ``(u, v) -> (v, u)``
        are excluded because a vine robot cannot double back along its
        own body.

        Encoding the direction of travel in the node is what makes the
        construction sound: every directed path is a legal walk in the
        original graph by construction, and every transition cost is
        evaluated with the same orientation that the walk actually uses.
        An undirected line graph cannot guarantee either, because two
        consecutive line-graph edges may share the same original vertex,
        which silently encodes an uncharged U-turn.
        """
        self.line_graph = nx.DiGraph()

        # One node per directed traversal of each visibility edge.
        self.line_graph.add_nodes_from(self.graph.to_directed().edges)

        # Every legal transition through a vertex: arrive from
        # ``previous``, leave toward ``next_point``. ``permutations``
        # already excludes ``previous == next_point``, i.e. reversals.
        transitions = [
            ((previous, shared), (shared, next_point))
            for shared in self.graph.nodes
            for previous, next_point in itertools.permutations(
                self.graph.neighbors(shared),
                2,
            )
        ]

        if transitions:
            previous_points = np.asarray(
                [tail[0] for tail, _ in transitions],
                dtype=float,
            )
            shared_points = np.asarray(
                [tail[1] for tail, _ in transitions],
                dtype=float,
            )
            next_points = np.asarray(
                [head[1] for _, head in transitions],
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
            self.line_graph.add_edges_from(
                (tail, head, {"weight": float(angle)})
                for (tail, head), angle in zip(
                    transitions,
                    turning_angles,
                    strict=True,
                )
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

        # The first segment leaves the start; the last segment arrives
        # at the goal. Neither incurs a turning cost.
        for neighbor in self.graph.neighbors(original_start):
            self.line_graph.add_edge(
                start_node,
                (original_start, neighbor),
                weight=0.0,
            )

        for neighbor in self.graph.neighbors(original_end):
            self.line_graph.add_edge(
                (neighbor, original_end),
                end_node,
                weight=0.0,
            )

        return self.line_graph

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

        # Interior nodes are directed traversals ``(u, v)`` of the
        # original graph, so the walk is just the tail of the first one
        # followed by the head of each in turn.
        directed_edges = line_path[1:-1]

        start = tuple(np.asarray(self.start, dtype=float))
        end = tuple(np.asarray(self.end, dtype=float))

        path_vertices = [directed_edges[0][0]]
        path_vertices.extend(head for _, head in directed_edges)

        if path_vertices[0] != start or path_vertices[-1] != end:
            raise RuntimeError(
                "The reconstructed angle path does not run from the "
                "start to the goal."
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

    def plot_pressure(
        self,
        ax: plt.Axes = None,
        *,
        length_units: str = "ft",
        angle_units: str = "degrees",
        points_per_segment: int = 50,
        title: str = None,
        show: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot required vine-robot pressure versus deployed path length.

        Any computed distance and angle paths are plotted on the same axes.
        Raises ``RuntimeError`` when neither path has been computed.

        Cumulative turning is reported in the legend for reference. It no
        longer determines the pressure on its own: two paths with equal
        length and equal total turning can end at different pressures
        depending on where along the path the bends fall.
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

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        if self.distance_path is not None:
            _, distance_angle_radians = path_metrics(self.distance_path)
            distance_angle_display = (
                distance_angle_radians * angle_scale
            )

            lengths, pressures = pressure_profile(
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
                    f"angle={distance_angle_display:.2f} {angle_label}, "
                    f"final pressure={pressures[-1]:.3f} psi)"
                ),
            )

        if self.angle_path is not None:
            _, measured_angle_radians = path_metrics(self.angle_path)
            angle_cost_radians = (
                measured_angle_radians
                if self.total_angle is None
                else self.total_angle
            )
            angle_cost_display = angle_cost_radians * angle_scale

            lengths, pressures = pressure_profile(
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
                    f"angle={angle_cost_display:.2f} {angle_label}, "
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

    @overload
    def plot(
        self,
        ax: plt.Axes = ...,
        show_vertices: bool = ...,
        show_labels: bool = ...,
        title: str = ...,
        show: bool = ...,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Polygonal course (``ObstacleCourse``): Matplotlib.

        Draws the obstacles, the start and end markers, and any computed paths
        on one set of 2D axes, and returns ``(figure, axes)``.
        """

    @overload
    def plot(
        self,
        plotter=...,
        color: str = ...,
        opacity: float = ...,
        show_edges: bool = ...,
        edge_color: str = ...,
        show_boundary: bool = ...,
        show_voxel_grid: bool = ...,
        show_axes: bool = ...,
        line_width: float = ...,
        marker_radius: float | None = ...,
        title: str = ...,
        window_size: tuple[int, int] = ...,
        jupyter_backend: str = ...,
        show: bool = ...,
    ):
        """Voxel course (``ObstacleCourseVoxels``): PyVista.

        Renders the walls, the start and end spheres, and any computed paths
        as tubes with a legend, and returns the ``pyvista.Plotter``.
        """

    def plot(self, *args, **kwargs):
        """Plot the course, the goals, and any computed paths.

        Every argument is forwarded to the course's ``plot_path``, so the
        parameters depend on which course is held; see the overloads above.
        Positional arguments continue from the parameter after the goals and
        paths, which this method supplies.

        Legend entries are bare names. The length, cumulative turning and
        pressure of each path are reported by :meth:`plot_pressure`.
        """
        return self.obstacle_course.plot_path(
            self.start,
            self.end,
            self.distance_path,
            self.angle_path,
            *args,
            **kwargs,
        )
