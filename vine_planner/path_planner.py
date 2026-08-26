import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

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
    line_graph: nx.DiGraph = field(init=False, default=None)

    # Nodes are integers indexing into `coords`; the array is the only place
    # coordinates live. Keying networkx by float tuples instead means every
    # cost calculation has to rebuild arrays out of them, which dominates
    # line-graph construction.
    coords: np.ndarray | None = field(init=False, default=None)
    start_index: int | None = field(init=False, default=None)
    end_index: int | None = field(init=False, default=None)

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
        """Build the visibility graph over the course vertices plus the goals.

        Nodes are integer indices into :attr:`coords`. Edge weights are
        computed for the whole edge list in one call rather than per edge.
        """
        course = self.obstacle_course

        vertices = np.asarray(course.vertices(), dtype=float)
        start = np.asarray(self.start, dtype=float)
        end = np.asarray(self.end, dtype=float)

        coords = np.vstack([vertices, start[None, :], end[None, :]])

        # A goal may coincide with an obstacle vertex. With coordinate tuples
        # as labels that merged silently; with integer labels it would instead
        # create two nodes at one point joined by a zero-length edge, which has
        # no turning angle and breaks the line graph.
        _, first_seen, inverse = np.unique(
            np.round(coords, 9),
            axis=0,
            return_index=True,
            return_inverse=True,
        )

        keep = np.sort(first_seen)
        position = np.empty(len(keep), dtype=np.int64)
        position[np.argsort(first_seen)] = np.arange(len(keep))

        self.coords = coords[keep]
        self.start_index = int(position[inverse[len(coords) - 2]])
        self.end_index = int(position[inverse[len(coords) - 1]])

        first, second = course.visible_pairs(self.coords)

        weights = np.linalg.norm(
            self.coords[first] - self.coords[second],
            axis=1,
        )

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(len(self.coords)))
        self.graph.add_weighted_edges_from(
            zip(first.tolist(), second.tolist(), weights.tolist())
        )

        return self.graph

    def create_line_graph(self):
        """Build the *directed* line graph of the visibility graph.

        A node is an ordered pair ``(u, v)``, read as "the robot traverses the
        visibility edge ``{u, v}`` from ``u`` toward ``v``". An arc joins
        ``(u, v)`` to ``(v, w)`` and carries the unsigned turning angle at
        ``v``. Reversals ``(u, v) -> (v, u)`` are excluded because a vine robot
        cannot double back along its own body.

        Encoding the direction of travel in the node is what makes the
        construction sound: every directed path is a legal walk in the original
        graph by construction, and every transition cost is evaluated with the
        same orientation the walk actually uses. An undirected line graph
        cannot guarantee either, because two consecutive line-graph edges may
        share the same original vertex, which silently encodes an uncharged
        U-turn.

        The transitions are enumerated with array arithmetic rather than
        ``itertools.permutations`` over each vertex's neighbours: sorting the
        directed edges by source groups every vertex's neighbours into a
        contiguous block, and a flat counter then decodes into the ordered
        pairs within each block.
        """
        self.line_graph = nx.DiGraph()

        # One node per directed traversal of each visibility edge.
        self.line_graph.add_nodes_from(self.graph.to_directed().edges)

        edges = np.asarray(list(self.graph.edges), dtype=np.int64)

        if len(edges):
            count = len(self.coords)

            # Each undirected edge twice, then grouped by source vertex.
            sources = np.concatenate([edges[:, 0], edges[:, 1]])
            targets = np.concatenate([edges[:, 1], edges[:, 0]])

            order = np.argsort(sources, kind="stable")
            sources, targets = sources[order], targets[order]

            degree = np.bincount(sources, minlength=count)
            block_start = np.concatenate([[0], np.cumsum(degree)[:-1]])

            # deg * (deg - 1) ordered pairs of distinct neighbours per vertex.
            pairs_per_vertex = degree * (degree - 1)
            total = int(pairs_per_vertex.sum())

        if len(edges) and total:
            base = np.repeat(block_start, pairs_per_vertex)
            local_degree = np.repeat(degree, pairs_per_vertex)

            offset = np.arange(total) - np.repeat(
                np.concatenate([[0], np.cumsum(pairs_per_vertex)[:-1]]),
                pairs_per_vertex,
            )

            incoming_slot = offset // (local_degree - 1)
            outgoing_slot = offset % (local_degree - 1)

            # Skip the pair a vertex would make with itself; shifting past it
            # is what excludes the reversal without a branch.
            outgoing_slot = outgoing_slot + (outgoing_slot >= incoming_slot)

            shared = np.repeat(np.arange(count), pairs_per_vertex)
            previous = targets[base + incoming_slot]
            following = targets[base + outgoing_slot]

            # Direction of travel into and out of the shared vertex.
            incoming_vectors = self.coords[shared] - self.coords[previous]
            outgoing_vectors = self.coords[following] - self.coords[shared]

            incoming_lengths = np.linalg.norm(incoming_vectors, axis=1)
            outgoing_lengths = np.linalg.norm(outgoing_vectors, axis=1)

            if np.any(incoming_lengths == 0) or np.any(outgoing_lengths == 0):
                raise RuntimeError(
                    "Cannot calculate a turning angle for a zero-length edge."
                )

            cosine_angles = np.einsum(
                "ij,ij->i",
                incoming_vectors,
                outgoing_vectors,
            ) / (incoming_lengths * outgoing_lengths)

            # Protect arccos from small floating-point errors such as
            # 1.0000000000000002.
            turning_angles = np.arccos(np.clip(cosine_angles, -1.0, 1.0))

            self.line_graph.add_edges_from(
                ((int(tail), int(middle)), (int(middle), int(head)),
                 {"weight": float(angle)})
                for tail, middle, head, angle in zip(
                    previous.tolist(),
                    shared.tolist(),
                    following.tolist(),
                    turning_angles.tolist(),
                )
            )

        # --------------------------------------------------------------
        # Add terminal nodes
        # --------------------------------------------------------------

        start_node = "start"
        end_node = "end"

        self.line_graph.add_node(start_node)
        self.line_graph.add_node(end_node)

        # The first segment leaves the start; the last segment arrives at the
        # goal. Neither incurs a turning cost.
        for neighbor in self.graph.neighbors(self.start_index):
            self.line_graph.add_edge(
                start_node,
                (self.start_index, neighbor),
                weight=0.0,
            )

        for neighbor in self.graph.neighbors(self.end_index):
            self.line_graph.add_edge(
                (neighbor, self.end_index),
                end_node,
                weight=0.0,
            )

        return self.line_graph

    def compute_distance_path(self) -> LineString:
        """Compute and store the minimum-distance visibility-graph path."""
        try:
            path_indices = nx.shortest_path(
                self.graph,
                source=self.start_index,
                target=self.end_index,
                weight="weight",
            )
        except nx.NetworkXNoPath as error:
            raise RuntimeError(
                "No collision-free distance path exists between start and end."
            ) from error

        self.distance_path = LineString(self.coords[path_indices])
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

        path_indices = [directed_edges[0][0]]
        path_indices.extend(head for _, head in directed_edges)

        if (
            path_indices[0] != self.start_index
            or path_indices[-1] != self.end_index
        ):
            raise RuntimeError(
                "The reconstructed angle path does not run from the "
                "start to the goal."
            )

        self.angle_path = LineString(self.coords[path_indices])
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
