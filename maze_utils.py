from gcsopt import GraphOfConvexSets
import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np
import random as rd

class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.walls = {"N": True, "S": True, "E": True, "W": True}

    def has_all_walls(self):
        return all(self.walls.values())

    def knock_down_wall(self, wall):
        self.walls[wall] = False


class Maze:
    directions = {"W": (-1, 0), "E": (1, 0), "S": (0, -1), "N": (0, 1)}

    def __init__(self, nx, ny, seed=0):
        self.nx = nx
        self.ny = ny
        self.cells = [[Cell(x, y) for y in range(ny)] for x in range(nx)]
        self.make_maze(seed)

    def get_cell(self, x, y):
        return self.cells[x][y]

    def unexplored_neighbors(self, cell):
        neighbours = []
        for direction, (dx, dy) in self.directions.items():
            x2 = cell.x + dx
            y2 = cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.get_cell(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours

    def make_maze(self, seed):
        rd.seed(seed)
        cell_stack = [self.get_cell(0, 0)]
        while len(cell_stack) > 0:
            neighbours = self.unexplored_neighbors(cell_stack[-1])
            if not neighbours:
                cell_stack.pop()
            else:
                direction, next_cell = rd.choice(neighbours)
                self.knock_down_wall(cell_stack[-1], direction)
                cell_stack.append(next_cell)

    def knock_down_wall(self, cell, wall):
        cell.knock_down_wall(wall)
        dx, dy = self.directions[wall]
        neighbor = self.get_cell(cell.x + dx, cell.y + dy)
        neighbor_wall = {"N": "S", "S": "N", "E": "W", "W": "E"}[wall]
        neighbor.knock_down_wall(neighbor_wall)

    def knock_down_walls(self, n, seed=0):
        rd.seed(seed)
        knock_downs = 0
        while knock_downs < n:
            x = rd.randint(1, self.nx - 2)
            y = rd.randint(1, self.ny - 2)
            cell = self.get_cell(x, y)
            walls = [wall for wall, has_wall in cell.walls.items() if has_wall]
            if len(walls) > 0:
                wall = rd.choice(walls)
                self.knock_down_wall(cell, wall)
                knock_downs += 1

    def plot(self):
        plt.plot([0, self.nx - 1], [self.ny, self.ny], c="k")
        plt.plot([self.nx, self.nx], [0, self.ny], c="k")
        for x in range(self.nx):
            for y in range(self.ny):
                if self.get_cell(x, y).walls["S"] and (x != 0 or y != 0):
                    plt.plot([x, x + 1], [y, y], c="k")
                if self.get_cell(x, y).walls["W"]:
                    plt.plot([x, x], [y, y + 1], c="k")


# NOTE: We might not need DFS here for finding the path.
# We might be able to get away with an even simplier algorithm of following the attached
# nodes, which I guess is essentially what DFS is doing anyway.  We can strip out a lot
# randomness and backtracking from DFS, since we know the path is a simple path with no loops.
def single_dfs(graph, source, target):
    """
    Perform a randomized depth-first search (DFS) from source to target in a
    graph.
    """

    # initialize path and set of visited vertices
    path = [source]
    visited = []

    # repeat until target is reached
    while path:
        if path[-1] == target:
            return path

        # collect neighbors and probabilities
        neighbors = []
        probabilities = []
        for edge in graph.outgoing_edges(path[-1]):
            neighbor = edge.head
            probability = edge.binary_variable.value
            if neighbor not in path + visited and probability > 0:
                neighbors.append(neighbor)
                probabilities.append(probability)

        # explore random neighbor
        if neighbors:
            probabilities = np.array(probabilities) / sum(probabilities)
            neighbor = np.random.choice(neighbors, p=probabilities)
            path.append(neighbor)

        # backtrack and prevent revisit of same vertex
        else:
            visited.append(path.pop())

    # path not found
    return None

def minimum_distance_cost(graph, maze):
    maze_side = maze.nx
    start = np.array([0.5, 0])
    goal = np.array([maze_side - 0.5, maze_side])

    # Add vertices.
    for i in range(maze_side):
        for j in range(maze_side):
            vertex = graph.add_vertex((i, j))

            # Trajectory start and end point within cell.
            x = vertex.add_variable((2, 2))

            # Minimize distance traveled within cell.
            vertex.add_cost(cp.norm2(x[1] - x[0]))

            # Constrain trajectory segment in cell.
            l = np.array([i, j])
            u = l + 1
            vertex.add_constraints([x[0] >= l, x[0] <= u])
            vertex.add_constraints([x[1] >= l, x[1] <= u])

            # Fix start and goal points.
            if all(l == 0):
                vertex.add_constraint(x[0] == start)
            elif all(u == maze_side):
                vertex.add_constraint(x[1] == goal)

    # Add edges between communicating cells.
    for i in range(maze_side):
        for j in range(maze_side):
            cell = maze.get_cell(i, j)
            tail = graph.get_vertex((i, j))
            for direction, d in maze.directions.items():
                if not cell.walls[direction]:
                    head = graph.get_vertex((i + d[0], j + d[1]))
                    edge = graph.add_edge(tail, head)

                    # Enforce trajectory continuity.
                    end_tail = tail.variables[0][1]
                    start_head = head.variables[0][0]
                    edge.add_constraint(end_tail == start_head)

def quad_over_linear_cost(graph, maze):
    maze_side = maze.nx
    start = np.array([0.5, 0])
    goal = np.array([maze_side - 0.5, maze_side])
    # Add vertices.
    for i in range(maze_side):
        for j in range(maze_side):
            vertex = graph.add_vertex((i, j))

            # Trajectory start and end point within cell.
            x = vertex.add_variable((2, 2))

            # # Minimize distance traveled within cell.
            # vertex.add_cost(cp.norm2(x[1] - x[0]))

            # Constrain trajectory segment in cell.
            l = np.array([i, j])
            u = l + 1
            vertex.add_constraints([x[0] >= l, x[0] <= u])
            vertex.add_constraints([x[1] >= l, x[1] <= u])

            # Fix start and goal points.
            if all(l == 0):
                vertex.add_constraint(x[0] == start)
            elif all(u == maze_side):
                vertex.add_constraint(x[1] == goal)

    # Add edges between communicating cells.
    for i in range(maze_side):
        for j in range(maze_side):
            cell = maze.get_cell(i, j)
            tail = graph.get_vertex((i, j))
            for direction, d in maze.directions.items():
                if not cell.walls[direction]:
                    head = graph.get_vertex((i + d[0], j + d[1]))
                    edge = graph.add_edge(tail, head)

                    # Enforce trajectory continuity.
                    p1, p2 = tail.variables[0]
                    p2_copy, p3 = head.variables[0]
                    edge.add_constraint(p2 == p2_copy)

                    # Angle cost.
                    vertical = True
                    if direction == "N":
                        # edge.add_cost(cp.quad_over_lin(p2[0] - p1[0], p2[1] - p1[1]))
                        # edge.add_cost(cp.quad_over_lin(p3[0] - p2[0], p3[1] - p2[1]))
                        edge.add_cost(cp.quad_over_lin(p2[0] - p1[0], 1))
                        edge.add_cost(cp.quad_over_lin(p3[0] - p2[0], 1))
                    elif direction == "S":
                        # edge.add_cost(cp.quad_over_lin(p2[0] - p1[0], p1[1] - p2[1]))
                        # edge.add_cost(cp.quad_over_lin(p3[0] - p2[0], p2[1] - p3[1]))
                        edge.add_cost(cp.quad_over_lin(p2[0] - p1[0], 1))
                        edge.add_cost(cp.quad_over_lin(p3[0] - p2[0], 1))
                    elif direction == "W":
                        # edge.add_cost(cp.quad_over_lin(p2[1] - p1[1], p1[0] - p2[0]))
                        # edge.add_cost(cp.quad_over_lin(p3[1] - p2[1], p2[0] - p3[0]))
                        edge.add_cost(cp.quad_over_lin(p2[1] - p1[1], 1))
                        edge.add_cost(cp.quad_over_lin(p3[1] - p2[1], 1))
                    elif direction == "E":
                        # edge.add_cost(cp.quad_over_lin(p2[1] - p1[1], p2[0] - p1[0]))
                        # edge.add_cost(cp.quad_over_lin(p3[1] - p2[1], p3[0] - p2[0]))
                        edge.add_cost(cp.quad_over_lin(p2[1] - p1[1], 1))
                        edge.add_cost(cp.quad_over_lin(p3[1] - p2[1], 1))

def construct_gcs_from_maze(
    maze_side: int = 5,
    knock_downs: int = 1,
    random_seed: int = 0,
    cost_constraints_func: callable = minimum_distance_cost
):
    maze = Maze(maze_side, maze_side, random_seed)
    maze.knock_down_walls(knock_downs)

    graph = GraphOfConvexSets()

    cost_constraints_func(graph, maze)

    return graph
