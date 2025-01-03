import numpy as np


def material_stiffness_matrix(E, nu):
    """
    Compute the material stiffness matrix for 3D isotropic elasticity.

    Parameters:
    - E: float, Young's modulus of the material.
    - nu: float, Poisson's ratio of the material.
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
    return C


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


def hex_element_tensors(nodes, E, nu):
    """
    Compute the strain-displacement matrix, Gauss weights, and Jacobian determinants for a hexahedral element.

    Parameters:
    - nodes: np.ndarray (24,), nodal coordinates of the hexahedron in the global frame.

    Returns:
    - Ke: np.ndarray (24x24). The element stiffness matrix.
    - Be: np.ndarray (6x24). The strain-displacement matrix.
    - CBe: np.ndarray (6x24). The stress-displacement matrix.
    """
    # Gauss quadrature points and weights (2-point rule)
    gauss_points = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    weights = np.array([1, 1])

    C = material_stiffness_matrix(E, nu)
    Ke = np.zeros((24, 24))  # force displacement (stiffness) matrix
    Be = np.zeros((6, 24))  # strain displacement matrix

    # Loop over Gauss points
    for i, xi_pt in enumerate(gauss_points):
        for j, eta_pt in enumerate(gauss_points):
            for k, zeta_pt in enumerate(gauss_points):
                dN_dxi = shape_function_derivatives(xi_pt, eta_pt, zeta_pt)

                # Jacobian matrix and inverse
                J = dN_dxi @ nodes
                detJ = np.linalg.det(J)
                if detJ <= 0:
                    raise ValueError(
                        "Jacobian determinant is non-positive. Check the element shape."
                    )

                integration_weight = detJ * weights[i] * weights[j] * weights[k]
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

                Be += integration_weight * B
                Ke += integration_weight * (B.T @ C @ B)

    CBe = C @ Be  # stress displacement matrix
    return Ke, Be, CBe
