import pyvista as pv

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_nodes(ax, nodes, **kwargs):
    ax.scatter(
        nodes[:, 0],
        nodes[:, 1],
        nodes[:, 2],
        **kwargs,
    )


def plot_elements(ax, nodes, faces, elements, wireframe=False, points=True):
    # Plot nodes
    if points:
        ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            nodes[:, 2],
            color="r",
        )

    for el in elements:
        elfaces = faces[el]
        verts = np.array([nodes[face] for face in elfaces])
        if wireframe:
            for vert in verts:
                ax.plot(
                    vert[:, 0],
                    vert[:, 1],
                    vert[:, 2],
                    color="k",
                    linewidth=1,
                    alpha=0.4,
                )
        else:
            c = Poly3DCollection(verts)
            c.set_alpha(0.1)
            c.set_edgecolor("k")
            ax.add_collection3d(c)


def plot_elements_no_faces(ax, nodes, elements, wireframe=False, points=True):
    # Plot nodes
    if points:
        ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            nodes[:, 2],
            color="r",
        )

    for el in elements:
        faces = []

        top_face = el[0:4]
        bottom_face = el[4:8]
        # Add bottom and top faces
        faces.append(bottom_face)
        faces.append(top_face)

        # Add side faces
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
                ax.plot(
                    vert[:, 0],
                    vert[:, 1],
                    vert[:, 2],
                    color="k",
                    linewidth=1,
                    alpha=0.4,
                )
        else:
            c = Poly3DCollection(verts)
            c.set_alpha(0.1)
            c.set_edgecolor("k")
            ax.add_collection3d(c)


def plot_forces(ax, nodes, forces, scale=0.1, min_magnitude_resolution=1):
    """Plot forces as arrows at each node with color and length normalized by magnitude."""
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
        length = scale * magnitude / max_magnitude  # Normalize length to [0, 1]
        ax.quiver(
            node[0],
            node[1],
            node[2],  # Starting point of the arrow
            forces[i, 0],
            forces[i, 1],
            forces[i, 2],  # Direction and length of the arrow
            color=color,
            length=length,
            normalize=True,
        )


def hexahedral_stiffness_matrix(nodes, E, nu):
    """
    Compute the stiffness matrix for an 8-node hexahedral element.

    Parameters:
    - nodes: np.ndarray (8x3), nodal coordinates of the hexahedron in the global frame.
    - E: float, Young's modulus of the material.
    - nu: float, Poisson's ratio of the material.

    Returns:
    - Ke: np.ndarray (24x24), the element stiffness matrix.
    """
    # Gauss quadrature points and weights (2-point rule)
    gauss_points = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    weights = np.array([1, 1])

    # Material stiffness matrix (3D isotropic elasticity)
    C = (E / ((1 + nu) * (1 - 2 * nu))) * np.array(
        [
            [1 - nu, nu, nu, 0, 0, 0],
            [nu, 1 - nu, nu, 0, 0, 0],
            [nu, nu, 1 - nu, 0, 0, 0],
            [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
            [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
            [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
        ]
    )

    # Initialize stiffness matrix
    Ke = np.zeros((24, 24))

    # Shape function derivatives in natural coordinates
    def shape_function_derivatives(xi, eta, zeta):
        dN_dxi = (
            np.array(
                [
                    [
                        -(1 - eta) * (1 - zeta),
                        (1 - eta) * (1 - zeta),
                        (1 + eta) * (1 - zeta),
                        -(1 + eta) * (1 - zeta),
                        -(1 - eta) * (1 + zeta),
                        (1 - eta) * (1 + zeta),
                        (1 + eta) * (1 + zeta),
                        -(1 + eta) * (1 + zeta),
                    ],
                    [
                        -(1 - xi) * (1 - zeta),
                        -(1 + xi) * (1 - zeta),
                        (1 + xi) * (1 - zeta),
                        (1 - xi) * (1 - zeta),
                        -(1 - xi) * (1 + zeta),
                        -(1 + xi) * (1 + zeta),
                        (1 + xi) * (1 + zeta),
                        (1 - xi) * (1 + zeta),
                    ],
                    [
                        -(1 - xi) * (1 - eta),
                        -(1 + xi) * (1 - eta),
                        -(1 + xi) * (1 + eta),
                        -(1 - xi) * (1 + eta),
                        (1 - xi) * (1 - eta),
                        (1 + xi) * (1 - eta),
                        (1 + xi) * (1 + eta),
                        (1 - xi) * (1 + eta),
                    ],
                ]
            )
            / 8.0
        )
        return dN_dxi

    # Loop over Gauss points
    for i, xi_pt in enumerate(gauss_points):
        for j, eta_pt in enumerate(gauss_points):
            for k, zeta_pt in enumerate(gauss_points):
                # Gauss point weight
                weight = weights[i] * weights[j] * weights[k]

                # Shape function derivatives in natural coordinates
                dN_dxi = shape_function_derivatives(xi_pt, eta_pt, zeta_pt)

                # Jacobian matrix
                J = dN_dxi @ nodes
                detJ = np.linalg.det(J)
                if detJ <= 0:
                    raise ValueError(
                        "Jacobian determinant is non-positive. Check the element shape."
                    )

                # Inverse Jacobian
                J_inv = np.linalg.inv(J)

                # Shape function derivatives in global coordinates
                dN_dx = J_inv @ dN_dxi

                # Strain-displacement matrix B
                B = np.zeros((6, 24))
                for n in range(8):  # Loop over nodes
                    B[0, n * 3] = dN_dx[0, n]
                    B[1, n * 3 + 1] = dN_dx[1, n]
                    B[2, n * 3 + 2] = dN_dx[2, n]
                    B[3, n * 3] = dN_dx[1, n]
                    B[3, n * 3 + 1] = dN_dx[0, n]
                    B[4, n * 3 + 1] = dN_dx[2, n]
                    B[4, n * 3 + 2] = dN_dx[1, n]
                    B[5, n * 3] = dN_dx[2, n]
                    B[5, n * 3 + 2] = dN_dx[0, n]

                # Add contribution to stiffness matrix
                Ke += weight * (B.T @ C @ B) * detJ

    return Ke


def example_single_element_displacements_to_forces():
    nodes = np.array(
        [
            [-1, -1, -1],
            [+1, -1, -1],
            [+1, +1, -1],
            [-1, +1, -1],
            [-1, -1, +1],
            [+1, -1, +1],
            [+1, +1, +1],
            [-1, +1, +1],
        ],
        dtype=float,
    )

    faces = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ],
        dtype=int,
    )

    elements = [
        [0, 1, 2, 3, 4, 5],
    ]

    nodes_shape = nodes.shape

    # Calculate the stiffness matrix
    elm_stiff_matrix = hexahedral_stiffness_matrix(nodes, 1000, 0.0)
    print(f"elm_stiff_matrix \n{elm_stiff_matrix}")

    # Calculate the resulting forces
    shear_amount = np.array([0.0, 0.0, -0.1])
    displacements = np.zeros(nodes_shape, dtype=float)
    displacements[4:] += shear_amount
    print(f"displacements \n{displacements}")

    displaced_nodes = nodes + displacements
    forces = (elm_stiff_matrix @ displacements.flatten()).reshape(-1, 3)
    print(f"forces \n{forces}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_elements(ax, nodes, faces, elements, wireframe=True, points=False)
    plot_elements(ax, displaced_nodes, faces, elements)
    plot_forces(ax, displaced_nodes, forces)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.axis("scaled")
    plt.show()

    return nodes, faces, elements, elm_stiff_matrix, displacements, forces


# 1 means constrained dof, 0 means free dof
# all nodes are unconstrained
def example_single_element_constraints_and_forces_to_displacements(
    nodes, faces, elements, elm_stiff_matrix, forces
):
    constraints = np.zeros(nodes.shape, dtype=int)
    # constrain the bottom face in all axis
    constraints[:4] = np.array([1, 1, 1])

    print(f"constraints \n{constraints}")

    free_dofs = np.where(constraints.flatten() == 0)[0]
    reduced_K = elm_stiff_matrix[np.ix_(free_dofs, free_dofs)]
    reduced_forces = forces.flatten()[free_dofs]

    free_displacements = np.linalg.solve(reduced_K, reduced_forces)

    # Reconstruct full displacement vector
    displacements = np.zeros_like(forces.flatten())
    displacements[free_dofs] = free_displacements

    # Reshape to (8x3) to match node coordinates
    displacements = displacements.reshape(8, 3)
    print(f"displacements \n{displacements}")
    displaced_nodes = nodes + displacements

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_elements(ax, nodes, faces, elements, wireframe=True, points=False)
    plot_elements(ax, displaced_nodes, faces, elements)
    plot_forces(ax, displaced_nodes, forces)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.axis("scaled")
    plt.show()


def example_single_element_there_and_back_again():
    nodes, faces, elements, elm_stiff_matrix, displacements, forces = (
        example_single_element_displacements_to_forces()
    )

    example_single_element_constraints_and_forces_to_displacements(
        nodes, faces, elements, elm_stiff_matrix, forces
    )


"""
Nodes: 1,2,3,4(bottom face, counter-clockwise) -> 5,6,7,8(top face, counter-clockwise)
"""


def stack_faces_2d(nodes2d, faces2d, z_heights):
    n_nodes_in_layer = nodes2d.shape[0]
    n_layers = len(z_heights)
    nodes3d = np.zeros((n_nodes_in_layer * n_layers, 3))
    elements = []

    # Stack nodes in the z direction
    for i, z in enumerate(z_heights):
        local_nodes3d = np.hstack([nodes2d, np.full((n_nodes_in_layer, 1), z)])
        nodes3d[i * n_nodes_in_layer : (i + 1) * n_nodes_in_layer] = local_nodes3d

    # Create elements
    for i in range(n_layers - 1):
        bottom_face_start = i * n_nodes_in_layer
        top_face_start = (i + 1) * n_nodes_in_layer
        for face in faces2d:
            bottom_face = face + bottom_face_start
            top_face = face + top_face_start
            elements.append(np.hstack([bottom_face, top_face]))

    return nodes3d, np.array(elements)


def faces_from_nodes2d(selection):
    base = np.array(
        [
            [0, 1, 2, 3],
        ],
        dtype=int,
    )
    mapped_faces = selection[base]
    return mapped_faces


def faces_from_nodes(selection):
    base = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ],
        dtype=int,
    )
    mapped_faces = selection[base]
    return mapped_faces


def example_faces_from_nodes():
    nodes = np.array(
        [
            [-1, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [-1, 1, 0],
            [-1, 0, 1],
            [0, 0, 1],
            [0, 1, 1],
            [-1, 1, 1],
            [1, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    faces = np.vstack(
        [
            faces_from_nodes(np.array([0, 1, 2, 3, 4, 5, 6, 7])),
            faces_from_nodes(np.array([1, 8, 9, 2, 5, 10, 11, 6])),
        ]
    )
    print(faces.shape[0] // 6)

    assert faces.shape[0] % 6 == 0, "Number of faces must be a multiple of 6"
    elements = [[6 * j + i for i in range(6)] for j in range(faces.shape[0] // 6)]

    print(elements)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_elements(ax, nodes, faces, elements, wireframe=True, points=False)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.axis("scaled")
    plt.show()


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
