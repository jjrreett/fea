import pyvista as pv

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

n_elements = 26
outer_radius = 4 * inch
inner_radius = 3.9 * inch
thetas = np.linspace(0, np.pi * 2, n_elements, endpoint=False).reshape(-1, 1)

unit_points = np.hstack([np.cos(thetas), np.sin(thetas)])
inner_points = unit_points * inner_radius
outer_points = unit_points * outer_radius

nodes2d = np.vstack([inner_points, outer_points])

face2ds = []
for i in range(n_elements):
    face2ds.append(
        [
            i,
            (i + n_elements),
            (i + 1) % n_elements + n_elements,
            (i + 1) % n_elements,
        ]
    )


forces2d = np.zeros_like(nodes2d)
force_nodes_2d = nodes2d[n_elements : (3 * n_elements) // 2]
forces2d[n_elements : (3 * n_elements) // 2, 1] = (
    -np.cos(np.pi / 2 * force_nodes_2d[:, 0] / outer_radius) * np.pi / 4 / outer_radius
)


# plt.show()


face2ds = np.array(face2ds)

nodes, elements = stack_faces_2d(
    nodes2d, face2ds, np.linspace(0, beam_length, n_elements_height)
)
print("nodes", nodes, sep="\n")
print("elements", elements, sep="\n")

forces = np.zeros_like(nodes)

forces[:, :2] = forces2d.repeat(n_elements_height, axis=0)
print("forces", forces.shape, forces, sep="\n")

# plotter = pv.Plotter()
# # plot_mesh(plotter, nodes, elements, show_edges=True, opacity=0.2)
# plot_mesh(
#     plotter,
#     nodes,
#     elements,
#     show_edges=True,
# )

# plotter.show()


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

# forces = np.zeros(nodes.shape, dtype=float)
# front_nodes = np.where(nodes[:, 1] == 0)[0]
# forces[front_nodes] += np.array([0, force_per_element, 0])
displacements, forces = solve(nodes, elements, constraints, forces)

displaced_nodes = nodes + displacements * 100

print("forces", forces / lbf, sep="\n")
print("displacements", displacements / inch, sep="\n")


plotter = pv.Plotter()
plot_mesh(plotter, nodes, elements, show_edges=True, opacity=0.2)
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
