from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from utils import *

np.set_printoptions(precision=5, linewidth=200, suppress=True)


psi = 6894.76
lbf = 4.44822
ft = 0.3048
inch = 0.0254

n_elements_width = 4
n_elements_height = 50

beam_width = 0.1
beam_length = 1.0
face_area = beam_width * beam_length
linear_load = 100.0 * lbf / ft
total_load = linear_load * beam_length
pressure = total_load / face_area

number_elements_face = (n_elements_width + 1) * (n_elements_height + 1)
force_per_element = total_load / number_elements_face


def generate_quad_grid(nx, ny, width, height):
    """
    Generate a regular grid of quadrilateral elements.

    Parameters:
    nx (int): Number of elements in the x-direction.
    ny (int): Number of elements in the y-direction.
    width (float): Width of the grid.
    height (float): Height of the grid.

    Returns:
    nodes (np.ndarray): Array of node coordinates.
    elements (np.ndarray): Array of element connectivity.
    """
    # Generate node coordinates
    x = np.linspace(0, width, nx + 1)
    y = np.linspace(0, height, ny + 1)
    nodes = np.array([[xi, yi] for yi in y for xi in x])

    # Generate element connectivity
    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = j * (nx + 1) + i
            n2 = n1 + 1
            n3 = n1 + (nx + 1)
            n4 = n3 + 1
            elements.append([n1, n2, n4, n3])

    return np.array(nodes, dtype=float), np.array(elements, dtype=int)


nodes2d, face2ds = generate_quad_grid(
    n_elements_width, n_elements_width, beam_width, beam_width
)

nodes, elements = stack_faces_2d(
    nodes2d, face2ds, np.linspace(0, beam_length, n_elements_height)
)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# plot_elements_no_faces(ax, nodes, elements, wireframe=True, points=False)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.axis("scaled")
# plt.show()


def solve(nodes, elements, constraints, forces):
    global_stiffness_matrix = np.zeros((nodes.size, nodes.size))

    for element in elements:
        elm_nodes = nodes[element]
        elm_stiff_matrix = hexahedral_stiffness_matrix(elm_nodes, 10_000_000 * psi, 0.3)

        dof_indices = np.array([i * 3 + j for i in element for j in range(3)])

        # Assemble global stiffness matrix
        idx = np.ix_(dof_indices, dof_indices)
        global_stiffness_matrix[idx] += elm_stiff_matrix

    free_dofs = np.where(constraints.flatten() == 0)[0]
    # print("free_dofs", free_dofs, sep="\n")

    reduced_K = global_stiffness_matrix[np.ix_(free_dofs, free_dofs)]
    reduced_forces = forces.flatten()[free_dofs]

    free_displacements = np.linalg.solve(reduced_K, reduced_forces)
    # TODO iterative solver

    # Reconstruct full displacement vector
    displacements = np.zeros_like(nodes).flatten()
    displacements[free_dofs] = free_displacements
    displacements = displacements.reshape(nodes.shape)

    forces = (global_stiffness_matrix @ displacements.flatten()).reshape(nodes.shape)

    return displacements, forces


# constrain the bottom faces in all axis
constraints = np.zeros(nodes.shape, dtype=int)
base_nodes = np.where(nodes[:, 2] == 0)[0]
constraints[base_nodes] = np.array([1, 1, 1])

forces = np.zeros(nodes.shape, dtype=float)
front_nodes = np.where(nodes[:, 1] == 0)[0]
forces[front_nodes] += np.array([0, force_per_element, 0])
displacements, forces = solve(nodes, elements, constraints, forces)

displaced_nodes = nodes + displacements * 100

print("forces", forces / lbf, sep="\n")
print("displacements", displacements / inch, sep="\n")


# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# plot_elements_no_faces(ax, nodes, elements, wireframe=True, points=False)
# plot_elements_no_faces(ax, displaced_nodes, elements, points=False)
# plot_forces(ax, displaced_nodes, forces)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.axis("scaled")
# plt.show()
# exit()

import pyvista as pv


def plot_nodes_pv(plotter, nodes, **kwargs):
    plotter.add_points(nodes, **kwargs)


def plot_elements_no_faces_pv(plotter, nodes, elements, wireframe=False, points=True):
    if points:
        plotter.add_points(nodes, color="r")

    for el in elements:
        faces = []

        top_face = el[0:4]
        bottom_face = el[4:8]
        faces.append(bottom_face)
        faces.append(top_face)

        for j in range(4):
            side_face = [
                bottom_face[j],
                bottom_face[(j + 1) % 4],
                top_face[(j + 1) % 4],
                top_face[j],
            ]
            faces.append(side_face)

        verts = np.array([nodes[face] for face in faces])
        if wireframe:
            for vert in verts:
                plotter.add_lines(
                    vert,
                    color="k",
                    width=1,
                    # opacity=0.4,
                )
        else:
            mesh = pv.PolyData(verts)
            plotter.add_mesh(
                mesh,
                color="k",
                # opacity=0.1,
                edge_color="k",
            )


def plot_forces_pv(plotter, nodes, forces, scale=0.1, min_magnitude_resolution=1):
    magnitudes = np.linalg.norm(forces, axis=1)
    max_magnitude = np.max(magnitudes)
    min_magnitude = np.min(magnitudes)
    if (
        min_magnitude_resolution is not None
        and (max_magnitude - min_magnitude) < min_magnitude_resolution
    ):
        mean_mag = (max_magnitude + min_magnitude) / 2
        max_magnitude = mean_mag + min_magnitude_resolution
        min_magnitude = mean_mag - min_magnitude_resolution

    norm = plt.Normalize(min_magnitude, max_magnitude)
    cmap = plt.get_cmap()

    for i, node in enumerate(nodes):
        magnitude = magnitudes[i]
        color = cmap(norm(magnitude))
        length = scale * magnitude / max_magnitude
        plotter.add_arrows(node, forces[i], mag=length, color=color)


def plot_mesh(plotter, nodes, elements, displacements=None, **kwargs):
    num_elements = elements.shape[0]
    cells = np.zeros((num_elements, 9), dtype=np.int32)
    cells[:, 0] = 8
    cells[:, 1:] = elements
    cells = cells.flatten()
    grid = pv.UnstructuredGrid(cells, 12 * np.ones((num_elements,)), nodes.flatten())
    if displacements is not None:
        # Calculate the displacement magnitude for each node
        displacement_magnitudes = np.linalg.norm(displacements, axis=1)

        dis = np.array([np.linalg.norm(displacement_magnitudes[el]) for el in elements])

        # Normalize the displacement magnitudes to [0, 1] for coloring
        norm = plt.Normalize(dis.min(), dis.max())
        cmap = plt.get_cmap("viridis")
        colors = cmap(norm(dis))[:, :3]  # Get RGB values

        # Assign colors to the cells
        grid.cell_data["colors"] = colors

    # Add the mesh to the plotter
    plotter.add_mesh(grid, **kwargs)


plotter = pv.Plotter()
# plot_mesh(plotter, nodes, elements, show_edges=True, opacity=0.2)
plot_mesh(
    plotter,
    displaced_nodes,
    elements,
    displacements=displacements,
    scalars="colors",
    rgb=True,
    show_edges=True,
)

plotter.show()
