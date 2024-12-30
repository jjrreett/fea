from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

from collections.abc import Sequence
import numpy.typing as npt


element_tensor_functions = {}


class ElementType(Enum):
    EMPTY = np.uint8(0)

    # 1D Elements
    ROD = np.uint8(1)
    BEAM2 = np.uint8(2)  # First-order beam element with 2 nodes
    BEAM3 = np.uint8(3)  # Second-order beam element with 3 nodes
    BEAM4 = np.uint8(4)  # Third-order beam element with 4 nodes
    CABLE = np.uint8(5)
    SPRING = np.uint8(6)  # Spring element
    PIPE = np.uint8(7)  # Pipe element (for flow or structural analysis)

    # 2D Elements (Surface)
    TRI = np.uint8(10)  # 3-node triangle
    TRI6 = np.uint8(11)  # 6-node quadratic triangle
    QUAD = np.uint8(12)  # 4-node quadrilateral
    QUAD8 = np.uint8(13)  # 8-node quadratic quadrilateral
    QUAD9 = np.uint8(14)  # 9-node quadratic quadrilateral (with center node)
    POLY = np.uint8(15)  # N-sided polygon
    MEMBRANE = np.uint8(16)  # Membrane element
    SHELL = np.uint8(17)  # Shell element
    PLATE = np.uint8(18)  # Plate element (for bending-dominated problems)

    # 3D Elements (Solid)
    TET4 = np.uint8(20)  # 4-node tetrahedron
    TET10 = np.uint8(21)  # 10-node quadratic tetrahedron
    PYRAMID5 = np.uint8(22)  # 5-node pyramid
    PRISM6 = np.uint8(23)  # 6-node wedge/prism
    PRISM15 = np.uint8(24)  # 15-node quadratic wedge/prism
    HEX8 = np.uint8(25)  # 8-node hexahedron
    HEX20 = np.uint8(26)  # 20-node quadratic hexahedron
    HEX27 = np.uint8(27)  # 27-node cubic hexahedron
    POLYHEDRON = np.uint8(28)  # N-faced polyhedron

    # Axisymmetric Elements
    AXISYM_TRI = np.uint8(30)  # Axisymmetric triangle
    AXISYM_QUAD = np.uint8(31)  # Axisymmetric quadrilateral

    # Specialized Elements
    CONTACT = np.uint8(40)  # Contact element
    MASS = np.uint8(41)  # Point mass
    RIGID = np.uint8(42)  # Rigid body element
    ACOUSTIC = np.uint8(43)  # Acoustic element for wave propagation
    COUPLED_FIELD = np.uint8(44)  # Coupled field element (e.g., thermal-mechanical)
    EMBEDDED = np.uint8(45)  # Embedded reinforcement element (e.g., rebar)

    @classmethod
    def register_tensor_functions(cls, element_type=None):
        """
        Decorator to register an element tensor function for a specific element type.
        If element_type is not provided, it infers the element type from the function name.
        """

        def decorator(func):
            nonlocal element_type
            if element_type is None:
                # Infer element type from function name
                func_name_parts = func.__name__.split("_")
                inferred_type = func_name_parts[0].upper()
                if inferred_type in ElementType.__members__:
                    element_type = ElementType[inferred_type]
                else:
                    raise ValueError(
                        f"Cannot infer element type from function name {func.__name__}"
                    )
            element_tensor_functions[element_type] = func
            return func

        if callable(element_type):
            # If element_type is a function, it means the decorator was used without arguments
            func = element_type
            element_type = None
            return decorator(func)
        else:
            # Otherwise, the decorator was used with arguments
            return decorator

    @classmethod
    def element_tensors(cls, element_type, nodes, *properties):
        """
        Calls the registered Element Tensor function for the given element type.
        """
        func = element_tensor_functions.get(
            element_type,
            lambda *args, **kwargs: (_ for _ in ()).throw(
                ValueError(
                    f"Element Tensor function not implemented for element type {element_type}."
                )
            ),
        )
        return func(nodes, *properties)


@ElementType.register_tensor_functions
def rod_tensors(nodes, E, A):
    """
    Compute the stiffness matrix for a 2-node rod element.

    Parameters:
    - nodes: npt.NDArray[np.float32] (2x6), nodal coordinates of the rod in the global frame.
    - E: float, Young's modulus of the material.
    - A: float, cross-sectional area of the rod.

    Returns:
    - Ke: npt.NDArray[np.float32] (12x12), the element stiffness matrix.
    - Be: npt.NDArray[np.float32] (6x12), the strain-displacement matrix.
    - CBe: npt.NDArray[np.float32] (6x12), the stress-displacement matrix.
    """

    # Extract only the translational coordinates (x, y, z)
    node_coords = nodes[:, :3]

    # Compute the length and direction of the rod
    vector = node_coords[1] - node_coords[0]
    length = np.linalg.norm(vector)
    direction = vector / length

    # Local stiffness matrix in the rod's local coordinates
    k_local = (E * A / length) * np.array([[1, -1], [-1, 1]], dtype=np.float32)

    # Transform local stiffness matrix into global coordinates
    T = np.outer(direction, direction)  # 3x3 transformation matrix

    # Expand to 12x12 global stiffness matrix, focusing only on translational DOFs
    Ke = np.zeros((12, 12), dtype=np.float32)
    for i in range(2):  # Loop over nodes
        for j in range(2):  # Loop over nodes
            Ke[i * 6 : i * 6 + 3, j * 6 : j * 6 + 3] = k_local[i, j] * T

    # Strain-displacement matrix (6x12), focusing on full 3D stress/strain components
    Be = np.zeros((6, 12), dtype=np.float32)
    # Axial strain (εxx)
    Be[0, :3] = -direction / length
    Be[0, 6:9] = direction / length
    # Other components are zero for a 1D rod
    # εyy, εzz, γxy, γxz, γyz remain zero

    # Stress-displacement matrix (6x12), scaled by Young's modulus
    CBe = E * Be

    return Ke, Be, CBe


def beam2_tensors_2d(
    nodes: npt.NDArray[np.float32],
    E: float,
    I: float,
    A: float,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Compute the stiffness matrix, strain-displacement matrix, and stress-displacement matrix for a 2D beam element.

    Parameters:
    - nodes: npt.NDArray[np.float32] (2x2), nodal coordinates of the beam in the global frame.
    - E: float, Young's modulus of the material.
    - I: float, second moment of area of the beam.
    - A: float, cross-sectional area of the beam.

    Returns:
    - Ke: npt.NDArray[np.float32] (4x4), the element stiffness matrix.
    - Be: npt.NDArray[np.float32] (6x4), the strain-displacement matrix.
    - CBe: npt.NDArray[np.float32] (6x4), the stress-displacement matrix.
    """
    # Validate input
    if nodes.shape != (2, 2):
        raise ValueError("nodes must be a 2x2 array containing nodal coordinates.")

    # Compute the length of the beam
    L = np.linalg.norm(nodes[1, 0] - nodes[0, 0])

    # Compute the stiffness matrix for the beam
    EI = E * I

    # Local stiffness matrix (2D beam)
    Ke = (2 * EI / L**3) * np.array(
        [
            [6, 3 * L, -6, 3 * L],
            [3 * L, 2 * L**2, -3 * L, L**2],
            [-6, -3 * L, 6, -3 * L],
            [3 * L, L**2, -3 * L, 2 * L**2],
        ],
        dtype=np.float32,
    )

    # Transformation matrix for 2D beam
    dx = nodes[1, 0] - nodes[0, 0]
    dy = nodes[1, 1] - nodes[0, 1]
    theta = np.arctan2(dy, dx)
    c = np.cos(theta)
    s = np.sin(theta)

    T = np.array(
        [[c, s, 0, 0], [-s, c, 0, 0], [0, 0, c, s], [0, 0, -s, c]], dtype=np.float32
    )

    # Global stiffness matrix
    Ke_global = T.T @ Ke @ T

    # Strain-displacement matrix (Be)
    Be = np.zeros((6, 4), dtype=np.float32)
    Be[0, 0] = -6 / L**2
    Be[0, 2] = 6 / L**2
    Be[1, 0] = -3 / L
    Be[1, 2] = 3 / L

    # Stress-displacement matrix (CBe)
    CBe = E * Be

    return Ke_global, Be, CBe


def test_beam2_tensors():
    """Test function for the beam2_tensors with assembly of global stiffness matrix."""
    E = 10_000_000  # Young's modulus in psi
    I = 5  # Second moment of area in in^4
    A = 1  # Cross-sectional area in in²
    force = 1000  # Applied force in lbf
    n_elements = 10  # Number of beam elements

    # Length of each element
    total_length = 48.0

    # Global stiffness matrix size (2 DOF per node, n_elements + 1 nodes)
    n_nodes = n_elements + 1
    nodes = np.array([[d, 0] for d in np.linspace(0, total_length, n_elements + 1)])
    elements = [[i, i + 1] for i in range(n_elements)]
    elements = np.array(elements, dtype=np.int32)

    forces = np.zeros((n_nodes, 2), dtype=np.float32)
    forces[-1, 0] = -force

    Kg = np.zeros((2 * n_nodes, 2 * n_nodes), dtype=np.float32)

    # Assemble global stiffness matrix
    for element in elements:
        # Compute element stiffness matrix
        Ke, _, _ = beam2_tensors_2d(nodes[element], E, I, A)

        dof_indices = np.array(
            [node_idx * 2 + j for node_idx in element for j in range(2)]
        )
        idx = np.ix_(dof_indices, dof_indices)
        Kg[idx] += Ke

    # Print global stiffness matrix
    print("Kg", Kg.shape, Kg, sep="\n")

    constraints = np.array([[0, 0]]).repeat(n_elements + 1, axis=0)
    constraints[0] = np.array([1, 1])

    free_dofs = np.where(constraints.flatten() == 0)[0]

    K = Kg[np.ix_(free_dofs, free_dofs)]
    print("K_reduced", K.shape, K, sep="\n")
    f = forces.flatten()[free_dofs]

    u = np.linalg.solve(K, f)
    displacements = np.zeros_like(nodes).flatten()
    displacements[free_dofs] = u
    displacements = displacements.reshape(nodes.shape)
    forces = (Kg @ displacements.flatten()).reshape(nodes.shape)
    print("displacements", displacements.shape, displacements, sep="\n")
    print("forces", forces.shape, forces, sep="\n")


@ElementType.register_tensor_functions
def beam2_tensors_3d(
    nodes: npt.NDArray[np.float32],
    E: float,
    nu: float,
    A: float,
    I: npt.NDArray[np.float32],
    J: float,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Compute the stiffness matrix, strain-displacement matrix, and stress-displacement matrix for a 3D beam element.

    Parameters:
    - nodes: npt.NDArray[np.float32] (2x6), nodal coordinates in the global frame.
    - E: float, Young's modulus of the material.
    - nu: float, Poisson's ratio of the material.
    - A: float, cross-sectional area of the beam.
    - I: npt.NDArray[np.float32] (3,), Second moments of area about principal axes (Ix, Iy, Iz).
    - J: float, Torsional constant of the beam.

    Returns:
    - Ke_global: npt.NDArray[np.float64] (12x12), the global stiffness matrix.
    - Be: npt.NDArray[np.float32] (6x12), the strain-displacement matrix.
    - CBe: npt.NDArray[np.float32] (6x12), the stress-displacement matrix.
    """
    G = E / (2 * (1 + nu))  # Shear modulus

    # Extract nodal coordinates
    x1, y1, z1 = nodes[0, :3]
    x2, y2, z2 = nodes[1, :3]

    # Compute beam length
    L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    # Direction cosines
    l = (x2 - x1) / L
    m = (y2 - y1) / L
    n = (z2 - z1) / L

    # Transformation matrix
    T = np.zeros((12, 12), dtype=np.float64)
    T[:3, :3] = T[3:6, 3:6] = T[6:9, 6:9] = T[9:12, 9:12] = np.array(
        [[l, m, n], [-m, l, 0], [-n, 0, l]]
    )

    # Local stiffness matrix
    Ke_local = np.zeros((12, 12), dtype=np.float64)

    # Axial stiffness
    Ke_local[0, 0] = Ke_local[6, 6] = E * A / L
    Ke_local[0, 6] = Ke_local[6, 0] = -E * A / L

    # Flexural stiffness about y-axis
    Ke_local[1, 1] = Ke_local[7, 7] = 12 * E * I[1] / L**3
    Ke_local[1, 7] = Ke_local[7, 1] = -12 * E * I[1] / L**3
    Ke_local[5, 1] = Ke_local[1, 5] = Ke_local[5, 7] = Ke_local[7, 5] = (
        6 * E * I[1] / L**2
    )

    # Flexural stiffness about z-axis
    Ke_local[2, 2] = Ke_local[8, 8] = 12 * E * I[2] / L**3
    Ke_local[2, 8] = Ke_local[8, 2] = -12 * E * I[2] / L**3
    Ke_local[4, 2] = Ke_local[2, 4] = Ke_local[4, 8] = Ke_local[8, 4] = (
        -6 * E * I[2] / L**2
    )

    # Torsional stiffness
    Ke_local[3, 3] = Ke_local[9, 9] = G * J / L
    Ke_local[3, 9] = Ke_local[9, 3] = -G * J / L

    # Strain-displacement matrix (Be)
    Be = np.zeros((6, 12), dtype=np.float32)
    Be[0, 0] = -1 / L
    Be[0, 6] = 1 / L

    # Flexural strains (y-axis bending)
    Be[1, 1] = Be[1, 7] = 12 / L**3
    Be[1, 5] = Be[1, 11] = 6 / L**2

    # Flexural strains (z-axis bending)
    Be[2, 2] = Be[2, 8] = 12 / L**3
    Be[2, 4] = Be[2, 10] = -6 / L**2

    # Stress-displacement matrix (CBe)
    CBe = Be.copy()

    # Transform stiffness matrix to global coordinates
    Ke_global = T.T @ Ke_local @ T

    print("Ke_global", Ke_global.shape, Ke_global, sep="\n")

    return Ke_global, Be, CBe


@ElementType.register_tensor_functions
def hex8_tensors(nodes, E, nu):
    """
    Compute the stiffness matrix for an 8-node hexahedral element.

    Parameters:
    - nodes: npt.NDArray[np.float32] (8x6), nodal coordinates of the hexahedron in the global frame.
    - E: float, Young's modulus of the material.
    - nu: float, Poisson's ratio of the material.

    Returns:
    - Ke: npt.NDArray[np.float32] (48x48), the element stiffness matrix.
    - Be: npt.NDArray[np.float32] (6x48), the strain-displacement matrix.
    - CBe: npt.NDArray[np.float32] (6x48), the stress-displacement matrix.
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
    Ke = np.zeros((48, 48))
    Be = np.zeros((6, 48))  # strain displacement matrix

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
                J = (
                    dN_dxi @ nodes[:, :3]
                )  # Only use the first 3 columns for positional coordinates
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

                # Correctly assemble stiffness and strain-displacement matrices
                for n1 in range(8):  # Loop over each node
                    for n2 in range(8):  # Loop over coupling nodes
                        # Indices for global stiffness matrix (translational DOFs only)
                        idx1 = slice(
                            n1 * 6, n1 * 6 + 3
                        )  # Translational DOFs for node n1
                        idx2 = slice(
                            n2 * 6, n2 * 6 + 3
                        )  # Translational DOFs for node n2

                        # Use np.ix_ to map local stiffness contributions
                        Ke[
                            np.ix_(
                                range(idx1.start, idx1.stop),
                                range(idx2.start, idx2.stop),
                            )
                        ] += (
                            weight
                            * (B.T @ C @ B)[
                                n1 * 3 : (n1 + 1) * 3, n2 * 3 : (n2 + 1) * 3
                            ]
                        )

                        # Strain-displacement matrix
                        Be[:, idx1] += weight * B[:, n1 * 3 : (n1 + 1) * 3] * detJ

    CBe = C @ Be  # stress displacement matrix
    return Ke, Be, CBe


@ElementType.register_tensor_functions
def prism6_tensors(nodes, E, nu):
    """
    Compute the stiffness matrix for a 6-node prism element.

    Parameters:
    - nodes: npt.NDArray[np.float32] (6x6), nodal coordinates of the prism in the global frame.
    - E: float, Young's modulus of the material.
    - nu: float, Poisson's ratio of the material.

    Returns:
    - Ke: npt.NDArray[np.float32] (18x18), the element stiffness matrix.
    """

    # Gauss quadrature points and weights (2-point rule for triangular prism)
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
    Be = np.zeros((6, 18))  # Strain-displacement matrix

    # Shape function derivatives in natural coordinates
    def shape_function_derivatives(xi, eta, zeta):
        dN_dxi = (
            np.array(
                [
                    [-(1 - eta), (1 - eta), eta, -eta, 0, 0],
                    [-(1 - xi), -xi, xi, (1 - xi), 0, 0],
                    [
                        -(1 - xi) * (1 - eta),
                        -(1 - xi) * eta,
                        xi * eta,
                        xi * (1 - eta),
                        (1 - xi) * (1 - eta),
                        xi * eta,
                    ],
                ]
            )
            / 4.0
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
                J = (
                    dN_dxi @ nodes[:, :3]
                )  # Only use the first 3 columns for positional coordinates
                detJ = np.linalg.det(J)
                if detJ <= 0:
                    raise ValueError(
                        "Jacobian determinant is non-positive. Check the element shape."
                    )

                # Inverse Jacobian
                J_inv = np.linalg.inv(J)

                # Shape function derivatives in global coordinates
                dN_dx = J_inv @ dN_dxi

                # Strain-displacement matrix
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

                # Correctly assemble stiffness and strain-displacement matrices
                for n1 in range(6):  # Loop over each node
                    for n2 in range(6):  # Loop over coupling nodes
                        # Indices for global stiffness matrix (translational DOFs only)
                        idx1 = slice(n1 * 3, n1 * 3 + 3)
                        idx2 = slice(n2 * 3, n2 * 3 + 3)

                        # Use np.ix_ to map local stiffness contributions
                        Ke[
                            np.ix_(
                                range(idx1.start, idx1.stop),
                                range(idx2.start, idx2.stop),
                            )
                        ] += (
                            weight
                            * (B.T @ C @ B)[
                                n1 * 3 : (n1 + 1) * 3, n2 * 3 : (n2 + 1) * 3
                            ]
                            * detJ
                        )

                        # Strain-displacement matrix
                        Be[:, idx1] += weight * B[:, n1 * 3 : (n1 + 1) * 3] * detJ

    CBe = C @ Be  # Stress-displacement matrix
    return Ke, Be, CBe


@dataclass
class Mesh:
    """
    Parameters
    ----------
    nodes : npt.ArrayLike[np.float32]
        Nx6 array where N is the number of nodes (control points). Each row displacemnt of the node (3 tranlational + 3 rotational).
        The nodes array will be flattened into a nuetral displacement vector.

    elements : sequence[sequence[int]]
        NxX array where each row is an index array refrencing the nodes in the mesh.
        N is the number of elements the mesh and X is variable length depending on the cell type.

    element_type : npt.ArrayLike[ElementType]
        Nx0 array of element types. See the ElementType enum for possible values.
        N is a divisor of the number of elements in the mesh. This array will be repeated untill it matches the number of elements.

    element_properties : sequence[sequence[np.float32]]
        Sequence of sequences of properties for each element. Each sub-sequence contains the properties needed by the stiffness matrix function.
        The element_stiffness_matrix function will be called with the signature `element_stiffness_matrix(nodes[element], *properties) for element, properties in zip(elements, element_properties)`.
        See ElementType class for more details on the stiffness matrix function signature.
        N is a divisor of the number of elements in the mesh. This array will be repeated until it matches the number of elements.

    constraints_vector : npt.ArrayLike[np.bool]
            Nx0 array representing the constrained degrees of freedom per node. A node is considered free if the value is False.
            The linear system of equations will be reduced to only include free degrees of freedom.
            N is a divisor of the number of elements in the mesh. This array will be repeated until it matches the number of elements.
            A second constraints vector will be constructed from the elements + element_type arrays and will be OR'd against this.
            In the context of FEA simulation, the constraints vector will be used with np.where and np.ix_ to downselect the stiffness matrix.

            - Locked degrees of freedom: These are the DOFs that are constrained and cannot move. They are represented by True in the constraints vector.
            - Free degrees of freedom: These are the DOFs that are not constrained and can move freely. They are represented by False in the constraints vector.
            - Independent degrees of freedom: These are the DOFs that remain after applying constraints and are used in the reduced system of equations.
            - Dependent degrees of freedom: These are the DOFs that are constrained and their values depend on the independent DOFs.

    constraints_matrix : npt.NDArray, optional
            Matrix to map from a complete displacement vector (u) to a partial displacement vector, assuming linear mapping.
            In the solver, this matrix will be used to reduce the displacement vector to only include independent degrees of freedom.
            A congruence transformation will be applied to the global stiffness matrix to match the reduced displacement vector.
            This will be achieved using a congruence transformation: constraints_matrix.T @ stiffness_matrix @ constraints_matrix.

    forces : npt.ArrayLike[np.float32]
        Nx6 array of force vectors corresponding to each node (3 translational + 3 rotational).
        N is a divisor of the number of elements in the mesh. This array will be repeated untill it matches the number of elements.

    displacement_vector : npt.ArrayLike[np.float32]
        Displacement vector for each node. The total displacement of the node is the neutral displacement + the displacement vector.

    strain : npt.NDArray, optional
        Strain tensor for each node. Default is None.

    von_mises_stress : npt.NDArray, optional
        Von Mises stress for each node. Default is None.

    global_stiffness_matrix : npt.NDArray, optional
        Global stiffness matrix of the system. Default is None.
    """

    nodes: npt.NDArray[np.float32]
    elements: Sequence[Sequence[int]]
    element_type: Sequence["ElementType"]
    element_properties: Sequence[Sequence[np.float32]]
    constraints_vector: Optional[npt.NDArray[np.bool]] = None
    constraints_matrix: Optional[npt.NDArray[np.float32]] = None
    forces: Optional[npt.NDArray[np.float32]] = None
    displacement_vector: Optional[npt.NDArray[np.float32]] = None
    strain: Optional[npt.NDArray] = None
    von_mises_stress: Optional[npt.NDArray] = None
    Kg: Optional[npt.NDArray] = None
    element_tensors: Optional[
        Sequence[Tuple[npt.NDArray, npt.NDArray, npt.NDArray]]
    ] = None
    element_strains: Optional[Sequence[npt.NDArray]] = None
    element_stresses: Optional[Sequence[npt.NDArray]] = None

    num_nodes = None
    num_dofs = None
    num_elements = None

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=np.float32).reshape(-1, 6)
        self.num_nodes = len(self.nodes)
        self.num_dofs = self.num_nodes * 6
        self.num_elements = len(self.elements)

        if self.num_elements % len(self.element_type) != 0:
            raise ValueError(
                "The number of element types must be a divisor of the number of elements."
            )
        self.element_type = (
            self.num_elements // len(self.element_type) * self.element_type
        )

        # Repeat element properties to match the number of elements
        # TODO cast to np.float32
        # TODO check against function signatures from type

        if self.num_elements % len(self.element_properties) != 0:
            raise ValueError(
                "The number of element properties must be a divisor of the number of elements."
            )

        self.element_properties = self.element_properties * (
            self.num_elements // len(self.element_properties)
        )

        if self.constraints_vector is None:
            self.constraints_vector = np.zeros((self.nodes.shape), dtype=bool)
        else:
            self.constraints_vector = np.asarray(self.constraints_vector, dtype=bool)
            assert (
                self.constraints_vector.shape == self.nodes.shape
            ), "Constraints vector must have the same shape as the nodes."

        if self.constraints_matrix is None:
            self.constraints_matrix = np.eye(self.num_dofs)
        else:
            self.constraints_matrix = np.asarray(self.constraints_matrix)
            assert self.constraints_matrix.shape == (self.num_dofs, self.num_dofs)

        if self.forces is None:
            self.forces = np.zeros((self.nodes.shape), dtype=np.float32)
        else:
            self.forces = self._normalize_repeat_array(
                self.forces, "forces", np.float32, (-1, 6)
            )

        if self.displacement_vector is None:
            self.displacement_vector = np.zeros((self.nodes.shape), dtype=np.float32)
        else:
            self.displacement_vector = np.asarray(
                self.displacement_vector, dtype=np.float32
            )
            assert (
                self.displacement_vector.shape == self.nodes.shape
            ), "Displacement vector must have the same shape as the nodes."

        # TODO perform checks on strain, von_mises_stress, and global_stiffness_matrix
        if self.strain is not None:
            self.strain = np.asarray(self.strain)
        if self.von_mises_stress is not None:
            self.von_mises_stress = np.asarray(self.von_mises_stress)
        if self.Kg is not None:
            self.Kg = np.asarray(self.Kg)

    def _normalize_repeat_array(self, x, name, dtype, shape):
        x = np.asarray(x, dtype=dtype).reshape(shape)
        if self.num_elements % x.shape[0] != 0:
            raise ValueError(
                f"The size of {name} must be a divisor of the number of elements."
            )
        return np.repeat(x, self.num_elements // x.size)

    def assemble_global_tensors(self):
        self.Kg = np.zeros((self.num_dofs, self.num_dofs))

        self.element_tensors = []

        for element, element_type, properties in zip(
            self.elements, self.element_type, self.element_properties
        ):
            element_tensor = ElementType.element_tensors(
                element_type, self.nodes[element], *properties
            )
            self.element_tensors.append(element_tensor)
            Ke, _, _ = element_tensor

            dof_indices = np.array(
                [node_idx * 6 + j for node_idx in element for j in range(6)]
            )
            idx = np.ix_(dof_indices, dof_indices)
            self.Kg[idx] += Ke

    def solve(self):
        free_dofs = np.where(self.constraints_vector.flatten() == 0)[0]

        K = self.Kg[np.ix_(free_dofs, free_dofs)]
        f = self.forces.flatten()[free_dofs]

        u = np.linalg.solve(K, f)

        # Reconstruct full displacement vector
        displacements = np.zeros_like(self.nodes).flatten()
        displacements[free_dofs] = u
        displacements = displacements.reshape(self.nodes.shape)

        forces = (self.Kg @ displacements.flatten()).reshape(self.nodes.shape)
        self.displacement_vector = displacements
        self.forces = forces

    def compute_element_strain_stress(self):
        self.element_strains = []
        self.element_stresses = []
        for i, element in enumerate(self.elements):
            _, Be, CBe = self.element_tensors[i]
            u = self.displacement_vector[element].flatten()
            element_strain = Be @ u
            element_stress = CBe @ u

            self.element_strains.append(element_strain)
            self.element_stresses.append(element_stress)

    def generate_pv_unstructured_mesh(self):
        import pyvista as pv

        """
        cells : sequence[int]
            Array of cells.  Each cell contains the number of points in the
            cell and the node numbers of the cell.

        cell_type : sequence[int]
            Cell types of each cell.  Each cell type numbers can be found from
            vtk documentation.  More efficient if using ``np.uint8``. See
            example below.

        points : sequence[float]
            Numpy array containing point locations.
        """
        cell_type_map = {
            ElementType.ROD: pv.CellType.LINE,
            ElementType.BEAM2: pv.CellType.LINE,
            ElementType.BEAM3: pv.CellType.EMPTY_CELL,
            ElementType.BEAM4: pv.CellType.EMPTY_CELL,
            ElementType.CABLE: pv.CellType.LINE,
            ElementType.SPRING: pv.CellType.LINE,
            ElementType.PIPE: pv.CellType.LINE,
            ElementType.TRI: pv.CellType.TRIANGLE,
            ElementType.TRI6: pv.CellType.EMPTY_CELL,
            ElementType.QUAD: pv.CellType.PIXEL,  # TODO use QUAD but fix ordering
            ElementType.QUAD8: pv.CellType.EMPTY_CELL,
            ElementType.QUAD9: pv.CellType.EMPTY_CELL,
            ElementType.POLY: pv.CellType.EMPTY_CELL,
            ElementType.MEMBRANE: pv.CellType.EMPTY_CELL,
            ElementType.SHELL: pv.CellType.EMPTY_CELL,
            ElementType.PLATE: pv.CellType.EMPTY_CELL,
            ElementType.TET4: pv.CellType.EMPTY_CELL,
            ElementType.TET10: pv.CellType.EMPTY_CELL,
            ElementType.PYRAMID5: pv.CellType.EMPTY_CELL,
            ElementType.PRISM6: pv.CellType.WEDGE,
            ElementType.PRISM15: pv.CellType.EMPTY_CELL,
            ElementType.HEX8: pv.CellType.HEXAHEDRON,
            ElementType.HEX20: pv.CellType.EMPTY_CELL,
            ElementType.HEX27: pv.CellType.EMPTY_CELL,
            ElementType.POLYHEDRON: pv.CellType.EMPTY_CELL,
            ElementType.AXISYM_TRI: pv.CellType.EMPTY_CELL,
            ElementType.AXISYM_QUAD: pv.CellType.EMPTY_CELL,
            ElementType.CONTACT: pv.CellType.EMPTY_CELL,
            ElementType.MASS: pv.CellType.EMPTY_CELL,
            ElementType.RIGID: pv.CellType.EMPTY_CELL,
            ElementType.ACOUSTIC: pv.CellType.EMPTY_CELL,
            ElementType.COUPLED_FIELD: pv.CellType.EMPTY_CELL,
            ElementType.EMBEDDED: pv.CellType.EMPTY_CELL,
        }
        cells = [[len(element), *element] for element in self.elements]
        cells = [item for sublist in cells for item in sublist]
        cell_type = [cell_type_map[cell_type] for cell_type in self.element_type]
        points = self.nodes[:, :3] + self.displacement_vector[:, :3] ** 100

        # Create the spatial reference
        grid = pv.UnstructuredGrid(cells, cell_type, points)

        return grid

    def compute_von_mises_stress(self):
        if self.element_stresses is None:
            raise ValueError("Element stresses have not been computed yet.")

        self.von_mises_stress = np.zeros((self.num_elements,))

        for i, stress in enumerate(self.element_stresses):
            sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_xz, tau_yz = stress
            von_mises = np.sqrt(
                0.5
                * (
                    (sigma_xx - sigma_yy) ** 2
                    + (sigma_yy - sigma_zz) ** 2
                    + (sigma_zz - sigma_xx) ** 2
                    + 6 * (tau_xy**2 + tau_xz**2 + tau_yz**2)
                )
            )
            self.von_mises_stress[i] = von_mises
