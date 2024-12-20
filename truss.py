import re
import numpy as np

import matplotlib.pyplot as plt

stiffness = 1000.0


def vec2(x=0.0, y=0.0):
    return np.array([x, y, 0.0], dtype=np.float32)


nodes = []
nodes.append(vec2(0.0, 0.0))
nodes.append(vec2(0.0, 1.0))
nodes.append(vec2(1.0, 0.5))
nodes = np.array(nodes)

members = []
members.append([0, 2])
members.append([1, 2])

loads = []
loads.append([2, vec2(0.0, -100.0)])


def plot_nodes(ax, nodes):
    ax.scatter(nodes[:, 0], nodes[:, 1])
    for i, node in enumerate(nodes):
        ax.text(node[0], node[1], f"Node {i}", fontsize=12, ha="right")


def plot_member(ax, members, nodes):
    for member in members:
        start_node = nodes[member[0]]
        end_node = nodes[member[1]]
        len = np.linalg.norm(start_node - end_node)
        ax.plot([start_node[0], end_node[0]], [start_node[1], end_node[1]], "k-")
        mid_x = (start_node[0] + end_node[0]) / 2
        mid_y = (start_node[1] + end_node[1]) / 2
        angle = np.degrees(
            np.arctan2(end_node[1] - start_node[1], end_node[0] - start_node[0])
        )
        ax.text(
            mid_x,
            mid_y,
            f"Member {member} Length: {len:.2f}",
            fontsize=12,
            ha="right",
            rotation=angle,
            rotation_mode="anchor",
        )


def plot_loads(ax: plt.Axes, nodes, loads):
    for load in loads:
        node = nodes[load[0]]
        ax.quiver(
            node[0],
            node[1],
            load[1][0],
            load[1][1],
        )


def plot_forces(ax: plt.Axes, nodes, forces):
    for i, force in enumerate(forces):
        node = nodes[i]

        ax.quiver(
            node[0],
            node[1],
            force[0],
            force[1],
        )


def compute_forces(nodes, members, displaced_nodes, forces):
    for member in members:
        start_node = displaced_nodes[member[0]]
        end_node = displaced_nodes[member[1]]

        dl = np.linalg.norm(nodes[member[1]] - nodes[member[0]]) - np.linalg.norm(
            end_node - start_node
        )

        force = -stiffness * dl
        force_vec = (
            force * (end_node - start_node) / np.linalg.norm(end_node - start_node)
        )
        forces[member[0]] += force_vec
        forces[member[1]] -= force_vec


displaced_nodes = nodes.copy()

while True:
    forces = np.zeros_like(nodes)
    compute_forces(nodes, members, displaced_nodes, forces)

    residual = loads[0][1] + forces[2]
    residual = np.linalg.norm(residual)
    print(residual)

    fig, ax = plt.subplots()
    plot_nodes(ax, displaced_nodes)
    plot_forces(ax, displaced_nodes, forces)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    plt.show()

    force_diff = []

    for i, load in loads:
        force_diff.append((i, load + forces[i, :]))

    for i, force in force_diff:
        print(i, force)
        displaced_nodes[i] += force / stiffness


# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Node Plot")
# ax.set_aspect("equal", adjustable="box")
# ax.grid(True)
# plt.show()
