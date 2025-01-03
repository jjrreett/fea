from enum import Enum
from typing import Iterable, Sequence, Optional
from attr import dataclass
import pyvista as pv

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.typing import ArrayLike
from dataclasses import dataclass


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
                print("Nodes", nodes.shape, nodes, sep="\n")
                print(f"Gauss point (xi, eta, zeta): ({xi_pt}, {eta_pt}, {zeta_pt})")
                print(f"Jacobian matrix J:\n{J}")
                print(f"Determinant of Jacobian detJ: {detJ}")
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


def wedge_stiffness_matrix(nodes, E, nu):
    """
    Compute the stiffness matrix for a 6-node wedge element.

    Parameters:
    - nodes: np.ndarray (6x3), nodal coordinates of the wedge in the global frame.
    - E: float, Young's modulus of the material.
    - nu: float, Poisson's ratio of the material.

    Returns:
    - Ke: np.ndarray (18x18), the element stiffness matrix.
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
    Ke = np.zeros((18, 18))

    # Shape function derivatives in natural coordinates
    def shape_function_derivatives_wedge(xi, eta, zeta):
        dN_dxi = (
            np.array(
                [
                    [-(1 - zeta), (1 - zeta), 0, -(1 + zeta), (1 + zeta), 0],
                    [-(1 - zeta), 0, (1 - zeta), -(1 + zeta), 0, (1 + zeta)],
                    [
                        -(1 - xi - eta),
                        -(1 - xi - eta),
                        -(1 - xi - eta),
                        (1 - xi - eta),
                        (1 - xi - eta),
                        (1 - xi - eta),
                    ],
                ]
            )
            / 2.0
        )
        return dN_dxi

    # Loop over Gauss points
    for i, xi_pt in enumerate(gauss_points):
        for j, eta_pt in enumerate(gauss_points):
            for k, zeta_pt in enumerate(gauss_points):
                # Gauss point weight
                weight = weights[i] * weights[j] * weights[k]

                # Shape function derivatives in natural coordinates
                dN_dxi = shape_function_derivatives_wedge(xi_pt, eta_pt, zeta_pt)

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
                B = np.zeros((6, 18))
                for n in range(6):  # Loop over nodes
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


    def stack_nodes(nodes2d, z_heights):
        n_nodes_in_layer = nodes2d.shape[0]
        n_layers = len(z_heights)
        nodes3d = np.zeros((n_nodes_in_layer * n_layers, 3))

        # Stack nodes in the z direction
        for i, z in enumerate(z_heights):
            local_nodes3d = np.hstack([nodes2d, np.full((n_nodes_in_layer, 1), z)])
            nodes3d[i * n_nodes_in_layer : (i + 1) * n_nodes_in_layer] = local_nodes3d
        return nodes3d


def stack_index_array(elements2d, n_layers, n_nodes):
    """
    Takes a set of 2d elements and layers the elements in 3d. Assumes elements2d is a NxM array.

    >>> stack_index_array([[0]], 1, 5)
    [[0, 5]]

    >>> stack_index_array([[1, 2, 3]], 2, 6)
    [[1, 2, 3, 7, 8, 9],
     [7, 8, 9, 13, 14, 15]]

    """
    elements3d = []
    for i in range(n_layers):
        bottom_element_start = i * n_nodes
        top_element_start = (i + 1) * n_nodes
        for element in elements2d:
            bottom_element = element + bottom_element_start
            top_element = element + top_element_start
            elements3d.append(np.hstack([bottom_element, top_element]))
    elements3d = np.array(elements3d)
    return elements3d


def stack_faces_2d(nodes2d, faces2d, z_heights, segments=None):
    n_nodes_in_layer = nodes2d.shape[0]
    n_layers = len(z_heights)
    nodes3d = np.zeros((n_nodes_in_layer * n_layers, 3))
    elements = []

    if segments is not None:
        n_segments_in_layer = segments.shape[0]
        assert segments.shape[1] == 2
        surfaces = np.zeros((n_segments_in_layer * n_layers - 1, 4))

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
    elements = np.array(elements)

    # Create surfaces
    # [[0,1],
    #  [1,2]...]
    # [[0,1,i,i+1]          # layer 1
    #  [1,2,1+i,2+i]
    #  [i,i+1,n*i,n*(i+1)], # layer n
    #  [1+i,2+i,1+i*n,2+n*(i+1)]]

    if segments is not None:
        for i in range(n_layers - 1):
            bottom_segment_start = i * n_nodes_in_layer
            top_segment_start = (i + 1) * n_nodes_in_layer
            for segment in segments:
                bottom_segment = segment + bottom_segment_start
                top_segment = segment + top_segment_start
                surfaces.append(np.hstack([bottom_segment, top_segment]))
        surfaces = np.array(elements)
        # for i in range(n_layers - 1):
        #     surfaces[i * n_segments_in_layer : (i + 1) * n_segments_in_layer, :] = (
        #         np.hstack(
        #             [
        #                 segments + i * n_segments_in_layer,
        #                 segments + (i + 1) * n_segments_in_layer,
        #             ]
        #         )
        #     )

        print(surfaces)

        return nodes3d, elements, surfaces
    return nodes3d, elements


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


def plot_mesh(plotter, nodes, elements, displacements=None, stresses=None, **kwargs):
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
    if stresses is not None:
        norm = plt.Normalize(stresses.min(), stresses.max())
        cmap = plt.get_cmap("viridis")
        colors = cmap(norm(stresses))[:, :3]  # Get RGB values
        grid.cell_data["colors"] = colors

    # Add the mesh to the plotter
    plotter.add_mesh(grid, **kwargs)


def print_matrix(name):
    print(name, eval(name), sep="\n")


def compute_element_strain_stress(
    nodes, displacements, elements, E, nu, return_gauss_points=False
):
    """
    Compute strain, stress, and Von Mises stress for each element or Gauss point.

    Parameters:
    - nodes: np.ndarray (Nx3), nodal coordinates in global space.
    - displacements: np.ndarray (Nx3), nodal displacement vector.
    - elements: np.ndarray (Mx8), list of elements defined by nodal indices.
    - E: float, Young's modulus.
    - nu: float, Poisson's ratio.
    - return_gauss_points: bool, if True return results at Gauss points, otherwise return averaged results for each element.

    Returns:
    - If return_gauss_points is True:
        - strains: list of np.ndarray, strain tensors at Gauss points for each element.
        - stresses: list of np.ndarray, stress tensors at Gauss points for each element.
        - von_mises: list of np.ndarray, Von Mises stress at Gauss points for each element.
    - If return_gauss_points is False:
        - element_strains: np.ndarray (Mx6), strain tensor for each element.
        - element_stresses: np.ndarray (Mx6), stress tensor for each element.
        - element_von_mises: np.ndarray (M,), Von Mises stress for each element.
    """
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

    # Gauss quadrature points and weights (2-point rule)
    gauss_points = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    weights = np.array([1, 1])

    # Results storage
    strains = []
    stresses = []
    von_mises = []

    for element in elements:
        elm_nodes = nodes[element]  # Extract node coordinates for the element
        elm_displacements = displacements[
            element
        ].flatten()  # Displacements at element nodes

        # Per Gauss point storage
        if return_gauss_points:
            elm_strain = []
            elm_stress = []
            elm_von_mises = []
        else:
            # For element-averaged values
            total_strain = np.zeros(6)
            total_stress = np.zeros(6)
            total_von_mises = 0.0
            total_weight = 0.0

        # Loop over Gauss points
        for i, xi_pt in enumerate(gauss_points):
            for j, eta_pt in enumerate(gauss_points):
                for k, zeta_pt in enumerate(gauss_points):
                    # Gauss point weight
                    weight = weights[i] * weights[j] * weights[k]

                    # Shape function derivatives in natural coordinates
                    dN_dxi = shape_function_derivatives(xi_pt, eta_pt, zeta_pt)

                    # Jacobian matrix and inverse
                    J = dN_dxi @ elm_nodes
                    detJ = np.linalg.det(J)
                    if detJ <= 0:
                        raise ValueError(
                            "Jacobian determinant is non-positive. Check the element shape."
                        )
                    J_inv = np.linalg.inv(J)

                    # Shape function derivatives in global coordinates
                    dN_dx = J_inv @ dN_dxi

                    # Strain-displacement matrix B
                    B = np.zeros((6, 24))
                    for n in range(8):
                        B[0, n * 3] = dN_dx[0, n]
                        B[1, n * 3 + 1] = dN_dx[1, n]
                        B[2, n * 3 + 2] = dN_dx[2, n]
                        B[3, n * 3] = dN_dx[1, n]
                        B[3, n * 3 + 1] = dN_dx[0, n]
                        B[4, n * 3 + 1] = dN_dx[2, n]
                        B[4, n * 3 + 2] = dN_dx[1, n]
                        B[5, n * 3] = dN_dx[2, n]
                        B[5, n * 3 + 2] = dN_dx[0, n]

                    # Compute strain and stress at Gauss point
                    strain = B @ elm_displacements
                    stress = C @ strain

                    # Compute Von Mises stress
                    sxx, syy, szz, txy, tyz, tzx = stress
                    von_mises_stress = np.sqrt(
                        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                        + 3 * (txy**2 + tyz**2 + tzx**2)
                    )

                    if return_gauss_points:
                        # Store Gauss point results
                        elm_strain.append(strain)
                        elm_stress.append(stress)
                        elm_von_mises.append(von_mises_stress)
                    else:
                        # Accumulate weighted values
                        total_strain += strain * weight * detJ
                        total_stress += stress * weight * detJ
                        total_von_mises += von_mises_stress * weight * detJ
                        total_weight += weight * detJ

        if return_gauss_points:
            strains.append(np.array(elm_strain))
            stresses.append(np.array(elm_stress))
            von_mises.append(np.array(elm_von_mises))
        else:
            # Average over Gauss points for the element
            strains.append(total_strain / total_weight)
            stresses.append(total_stress / total_weight)
            von_mises.append(total_von_mises / total_weight)

    if return_gauss_points:
        return strains, stresses, von_mises
    else:
        return (
            np.array(strains),
            np.array(stresses),
            np.array(von_mises),
        )


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

    K = global_stiffness_matrix[np.ix_(free_dofs, free_dofs)]
    f = forces.flatten()[free_dofs]

    u = np.linalg.solve(K, f)

    # Reconstruct full displacement vector
    displacements = np.zeros_like(nodes).flatten()
    displacements[free_dofs] = u
    displacements = displacements.reshape(nodes.shape)

    forces = (global_stiffness_matrix @ displacements.flatten()).reshape(nodes.shape)

    return displacements, forces


def solve(nodes, elements, constraints, forces, E, nu):
    global_stiffness_matrix = np.zeros((nodes.size, nodes.size))

    for element in elements:
        elm_nodes = nodes[element]
        elm_stiff_matrix = hexahedral_stiffness_matrix(elm_nodes, E, nu)

        dof_indices = np.array([i * 3 + j for i in element for j in range(3)])

        # Assemble global stiffness matrix
        idx = np.ix_(dof_indices, dof_indices)
        global_stiffness_matrix[idx] += elm_stiff_matrix

    free_dofs = np.where(constraints.flatten() == 0)[0]

    K = global_stiffness_matrix[np.ix_(free_dofs, free_dofs)]
    f = forces.flatten()[free_dofs]

    u = np.linalg.solve(K, f)

    # Reconstruct full displacement vector
    displacements = np.zeros_like(nodes).flatten()
    displacements[free_dofs] = u
    displacements = displacements.reshape(nodes.shape)

    forces = (global_stiffness_matrix @ displacements.flatten()).reshape(nodes.shape)

    # Compute strain, stress, and Von Mises stress
    strains, stresses, von_mises = compute_element_strain_stress(
        nodes, displacements, elements, E, nu
    )

    return displacements, forces, strains, stresses, von_mises


def solve_wedge(nodes, elements, constraints, forces, E, nu):
    global_stiffness_matrix = np.zeros((nodes.size, nodes.size))

    for element in elements:
        elm_nodes = nodes[element]
        # elm_stiff_matrix = wedge_stiffness_matrix(elm_nodes, E, nu)
        try:
            elm_stiff_matrix = wedge_stiffness_matrix(elm_nodes, E, nu)
        except ValueError as e:
            print(f"Error with element {element}: {e}")
            print(f"Element nodes: {elm_nodes}")
            raise

        dof_indices = np.array([i * 3 + j for i in element for j in range(3)])

        # Assemble global stiffness matrix
        idx = np.ix_(dof_indices, dof_indices)
        global_stiffness_matrix[idx] += elm_stiff_matrix

    free_dofs = np.where(constraints.flatten() == 0)[0]

    K = global_stiffness_matrix[np.ix_(free_dofs, free_dofs)]
    f = forces.flatten()[free_dofs]

    u = np.linalg.solve(K, f)

    # Reconstruct full displacement vector
    displacements = np.zeros_like(nodes).flatten()
    displacements[free_dofs] = u
    displacements = displacements.reshape(nodes.shape)

    forces = (global_stiffness_matrix @ displacements.flatten()).reshape(nodes.shape)

    # Compute strain, stress, and Von Mises stress
    strains, stresses, von_mises = compute_element_strain_stress(
        nodes, displacements, elements, E, nu
    )

    return displacements, forces, strains, stresses, von_mises


def circle(N, R, x=0, y=0):
    i = np.arange(N)
    theta = i * 2 * np.pi / N
    pts = np.stack([np.cos(theta), np.sin(theta)], axis=1) * R
    pts[:, 0] += x  # Adjust x coordinates
    pts[:, 1] += y  # Adjust y coordinates
    seg = np.stack([i, (i + 1) % N], axis=1)
    return pts, seg


def solve_fea(
    nodes: Sequence[Sequence[np.float32]],
    cells: Sequence[Sequence[np.int64]],
    cell_type: Sequence[np.int8],
    youngs_modulus: Sequence[np.float32],
    poisson_ratio: Sequence[np.float32],
    constrained_dofs: Sequence[Sequence[np.bool]],
    forces: Sequence[Sequence[np.float32]],
    constraints_matrix: np.ndarray = None,
):
    """
    Parameters
    ----------
    nodes : sequence[sequence[float]]
        Sequence of sequences of variable length. Each sub-vector fully describes the displacement of the node
        and corresponds to the cell type's stiffness matrix. Nodes will be flattened into a displacement vector.

    cells : sequence[int]
        Array of cells. Each cell contains the number of points in the
        cell and the node numbers of the cell.

    cell_type : sequence[int]
        Cell types of each cell. Each cell type number can be found from
        vtk documentation. More efficient if using ``np.uint8``. See
        example below.
        If the length of the sequence is a fraction of the number of cells, the sequence will be repeated.

    youngs_modulus : sequence[float]
        Young's modulus of the material per cell.
        If the length of the sequence is a fraction of the number of cells, the sequence will be repeated.

    poisson_ratio : sequence[float]
        Poisson's Ratio of the material per cell.
        If the length of the sequence is a fraction of the number of cells, the sequence will be repeated.

    constrained_dofs : sequence[bool]
        Constrained degrees of freedom per node. A node is considered free if the value is False.
        If the length of the sequence is a fraction of the number of nodes, the sequence will be repeated.

    forces : sequence[float]
        Sequence of sequences of force vectors corresponding to each node.
        If the length of the sequence is a fraction of the number of nodes, the sequence will be repeated.

    constraints_matrix : np.ndarray, optional
        Matrix to map from a complete displacement vector (u) to a partial displacement vector,
        assuming linear mapping. Default is None.
    """
    global_stiffness_matrix = np.zeros((nodes.size, nodes.size))

    for element in elements:
        elm_nodes = nodes[element]
        elm_stiff_matrix = hexahedral_stiffness_matrix(elm_nodes, E, nu)

        dof_indices = np.array([i * 3 + j for i in element for j in range(3)])

        # Assemble global stiffness matrix
        idx = np.ix_(dof_indices, dof_indices)
        global_stiffness_matrix[idx] += elm_stiff_matrix

    free_dofs = np.where(constraints.flatten() == 0)[0]

    K = global_stiffness_matrix[np.ix_(free_dofs, free_dofs)]
    f = forces.flatten()[free_dofs]

    u = np.linalg.solve(K, f)

    # Reconstruct full displacement vector
    displacements = np.zeros_like(nodes).flatten()
    displacements[free_dofs] = u
    displacements = displacements.reshape(nodes.shape)

    forces = (global_stiffness_matrix @ displacements.flatten()).reshape(nodes.shape)

    # Compute strain, stress, and Von Mises stress
    strains, stresses, von_mises = compute_element_strain_stress(
        nodes, displacements, elements, E, nu
    )

    return displacements, forces, strains, stresses, von_mises
