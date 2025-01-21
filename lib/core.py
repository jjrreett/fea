from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import time

import numpy as np
import numpy.typing as npt
import pyvista as pv
from tqdm import tqdm

import scipy.sparse
import scipy.sparse.linalg


element_tensor_functions = {}
NODES_WIDTH = 3
DEBUG = False


def np_print(name, ary):
    print(name, ary.shape, ary, sep="\n")


def debug_np_print(name, ary):
    if DEBUG:
        np_print(name, ary)


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def gauss_points_2d(order=2):
    """Returns Gauss quadrature points and weights for a triangle."""
    if order == 2:
        points = [(1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)]
        weights = [1 / 3, 1 / 3, 1 / 3]
    # Add higher-order rules if needed
    return points, weights


def gauss_points_1d(order=2):
    """Returns Gauss quadrature points and weights for 1D integration."""
    if order == 2:
        points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        weights = [1, 1]
    return points, weights


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
    - nodes: npt.NDArray[np.float32] (2x3), nodal coordinates of the rod in the global frame.
    - E: float, Young's modulus of the material.
    - A: float, cross-sectional area of the rod.

    Returns:
    - Ke: npt.NDArray[np.float32] (6x6), the element stiffness matrix.
    - Be: npt.NDArray[np.float32] (6x6), the strain-displacement matrix.
    - CBe: npt.NDArray[np.float32] (6x6), the stress-displacement matrix.
    """

    # Extract the translational coordinates
    node_coords = nodes

    # Compute the length and direction of the rod
    vector = node_coords[1] - node_coords[0]
    length = np.linalg.norm(vector)
    direction = vector / length

    # Local stiffness matrix in the rod's local coordinates
    k_local = (E * A / length) * np.array([[1, -1], [-1, 1]], dtype=np.float32)

    # Transform local stiffness matrix into global coordinates
    T = np.outer(direction, direction)  # 3x3 transformation matrix

    # Expand to 6x6 global stiffness matrix
    Ke = np.zeros((6, 6), dtype=np.float32)
    for i in range(2):  # Loop over nodes
        for j in range(2):  # Loop over nodes
            Ke[i * 3 : i * 3 + 3, j * 3 : j * 3 + 3] = k_local[i, j] * T

    # Full 6x6 strain-displacement matrix (6 strain components × 6 DOFs)
    Be = np.zeros((6, 6), dtype=np.float32)
    # Axial strain (ε_xx)
    Be[0, :3] = -direction / length
    Be[0, 3:] = direction / length
    # Other strain components (ε_yy, ε_zz, γ_xy, γ_xz, γ_yz) remain zero

    # Full 6x6 stress-displacement matrix
    CBe = np.zeros((6, 6), dtype=np.float32)
    CBe[0] = E * Be[0]  # Only the axial stress component is nonzero

    return Ke, Be, CBe


@ElementType.register_tensor_functions
def hex8_tensors(nodes, E, nu):
    """
    Compute the stiffness matrix for an 8-node hexahedral element.

    Parameters:
    - nodes: npt.NDArray[np.float32] (8x3), nodal coordinates of the hexahedron in the global frame.
    - E: float, Young's modulus of the material.
    - nu: float, Poisson's ratio of the material.

    Returns:
    - Ke: npt.NDArray[np.float32] (24x24), the element stiffness matrix.
    - Be: npt.NDArray[np.float32] (6x24), the strain-displacement matrix.
    - CBe: npt.NDArray[np.float32] (6x24), the stress-displacement matrix.
    """

    gauss_points, weights = gauss_points_1d()

    # Material stiffness matrix (3D isotropic elasticity)
    C = (E / ((1 + nu) * (1 - 2 * nu))) * np.array(
        [
            [1 - nu, nu, nu, 0, 0, 0],
            [nu, 1 - nu, nu, 0, 0, 0],
            [nu, nu, 1 - nu, 0, 0, 0],
            [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
            [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
            [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
        ],
        dtype=np.float32,
    )

    # Initialize stiffness matrix and strain-displacement matrix
    Ke = np.zeros((24, 24), dtype=np.float32)
    Be = np.zeros((6, 24), dtype=np.float32)

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
                ],
                dtype=np.float32,
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
                B = np.zeros((6, 24), dtype=np.float32)
                for n in range(8):  # Loop over nodes
                    B[0, n * 3] = dN_dx[0, n]  # ε_xx
                    B[1, n * 3 + 1] = dN_dx[1, n]  # ε_yy
                    B[2, n * 3 + 2] = dN_dx[2, n]  # ε_zz
                    B[3, n * 3] = dN_dx[1, n]  # γ_xy
                    B[3, n * 3 + 1] = dN_dx[0, n]
                    B[4, n * 3 + 1] = dN_dx[2, n]  # γ_yz
                    B[4, n * 3 + 2] = dN_dx[1, n]
                    B[5, n * 3] = dN_dx[2, n]  # γ_zx
                    B[5, n * 3 + 2] = dN_dx[0, n]

                # Stiffness matrix contribution
                Ke += weight * B.T @ C @ B * detJ

                # Strain-displacement matrix contribution
                Be += weight * B * detJ

    # Compute stress-displacement matrix
    CBe = C @ Be

    return Ke, Be, CBe


@ElementType.register_tensor_functions
def beam2_tensors(nodes, E, nu, Iyy, Izz, A, J, theta):
    """
    Generate the 12x12 stiffness matrix for a 2D beam element embedded in 3D space with 6 DOFs per node.

    Parameters:
    nodes : np.array - [[u0, v0, w0], [θx0, θy0, θz0], [u1, v1, w1], [θx1, θy1, θz1]]
        Coordinates of the beam element's start and end nodes in 3D space.
    E : float - Young's modulus of the material
    nu : float - Poisson's ratio
    I : float - Moment of inertia of the cross-section
    A : float - Cross-sectional area
    J : float - Polar moment of inertia
    theta : float - Angle of rotation about the x-axis before rodriguez rotation

    Returns:
    - Ke: npt.NDArray[np.float32] (12x12), the element stiffness matrix.
    - Be: npt.NDArray[np.float32] (6x12), the strain-displacement matrix.
    - CBe: npt.NDArray[np.float32] (6x12), the stress-displacement matrix.
    """
    K_local = np.zeros((12, 12))

    G = E / (2 * (1 + nu))  # Shear modulus
    p0, r0, p1, r1 = nodes

    # Calculate the beam length
    L = np.linalg.norm(p1 - p0)

    # Precompute stiffness terms
    EA_L = E * A / L
    GJ_L = G * J / L

    # Local stiffness matrix (12x12 for 3D beam element)
    K_local = np.zeros((12, 12))

    # Axial stiffness
    K_local[0, 0] = K_local[6, 6] = EA_L
    K_local[0, 6] = K_local[6, 0] = -EA_L

    # Torsional stiffness
    K_local[3, 3] = K_local[9, 9] = GJ_L
    K_local[3, 9] = K_local[9, 3] = -GJ_L

    L2 = L * L
    L3 = L2 * L
    # Precompute the terms involving powers of L
    term1 = 6 / L3
    term2 = 3 / L2
    term3 = 1 / L
    term4 = 2 / L

    # Bending stiffness (xy plane)
    K_local[np.ix_([1, 5, 7, 11], [1, 5, 7, 11])] = (2 * E * Iyy) * np.array(
        [
            [term1, term2, -term1, term2],
            [term2, term4, -term2, term3],
            [-term1, -term2, term1, -term2],
            [term2, term3, -term2, term4],
        ],
        dtype=np.float32,
    )

    # Bending stiffness (xz plane)
    z_nodes = np.zeros((2, 2), dtype=np.float32)
    z_nodes[0, 0] = nodes[0, 2]  # d is z along the beam axis 0 0 X
    z_nodes[0, 1] = nodes[1, 1]  # rotation about y           0 X 0
    z_nodes[1, 0] = nodes[2, 2]  # d is z along the beam axis 0 0 X
    z_nodes[1, 1] = nodes[3, 1]  # rotation about y           0 X 0
    K_local[np.ix_([2, 4, 8, 10], [2, 4, 8, 10])] = (2 * E * Izz) * np.array(
        [
            [term1, term2, -term1, term2],
            [term2, term4, -term2, term3],
            [-term1, -term2, term1, -term2],
            [term2, term3, -term2, term4],
        ],
        dtype=np.float32,
    )

    ctheta, stheta = np.cos(theta), np.sin(theta)

    R_x = np.array([[1, 0, 0], [0, ctheta, -stheta], [0, stheta, ctheta]])

    # 1. Primary axis direction vector: p1 - p0 (this defines the beam's axis)
    beam_axis = p1 - p0
    beam_axis = beam_axis / np.linalg.norm(beam_axis)  # Normalize it

    l, m, n = beam_axis

    # Transformation matrix
    T = np.zeros((12, 12), dtype=np.float64)
    T[:3, :3] = T[3:6, 3:6] = T[6:9, 6:9] = T[9:12, 9:12] = np.array(
        [[l, m, n], [-m, l, 0], [-n, 0, l]]
    )
    # Apply additional rotation around the x-axis
    T[:3, :3] = np.dot(
        R_x, T[:3, :3]
    )  # Apply additional rotation to the translation part
    T[3:6, 3:6] = np.dot(
        R_x, T[3:6, 3:6]
    )  # Apply additional rotation to the rotation part
    T[6:9, 6:9] = np.dot(
        R_x, T[6:9, 6:9]
    )  # Apply additional rotation to the translation part
    T[9:, 9:] = np.dot(R_x, T[9:, 9:])  # Apply additional rotation to the rotation part

    # Apply the transformation to the local stiffness matrix
    K_global = T.T @ K_local @ T

    # Compute strain-displacement matrix Be
    Be = np.zeros((6, 12), dtype=np.float32)

    # Strain-displacement matrix based on beam theory
    Be[0, 0] = Be[0, 6] = -6 / L**2
    Be[0, 2] = Be[0, 8] = 6 / L**2
    Be[1, 1] = Be[1, 7] = -3 / L
    Be[1, 3] = Be[1, 9] = 3 / L
    Be[2, 4] = Be[2, 10] = 1
    Be[2, 5] = Be[2, 11] = -1

    # Compute stress-displacement matrix CBe
    CBe = E * Be

    return K_global, Be, CBe


@ElementType.register_tensor_functions
def prism6_tensors(
    nodes: np.ndarray, E: float, nu: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the stiffness matrix for a 6-node element.

    Parameters:
    - nodes: npt.NDArray[np.float32] (6x3), nodal coordinates of the solid in the global frame.
    - E: float, Young's modulus of the material.
    - nu: float, Poisson's ratio of the material.

    Returns:
    - Ke: npt.NDArray[np.float32] (18x18), the element stiffness matrix.
    - Be: npt.NDArray[np.float32] (6x18), the strain-displacement matrix.
    - CBe: npt.NDArray[np.float32] (6x18), the stress-displacement matrix.
    """
    # Number of nodes and degrees of freedom per node

    # Material property matrix (C)
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

    def shape_function_derivatives(xi, eta, zeta):
        """
        Compute the derivatives of the shape functions for a 6-node wedge element
        with respect to natural coordinates (xi, eta, zeta).

        Parameters:
        - xi: float, natural coordinate xi
        - eta: float, natural coordinate eta
        - zeta: float, natural coordinate zeta

        Returns:
        - dN: np.ndarray (3, 6), derivatives of shape functions w.r.t. xi, eta, zeta
        """
        dN = np.zeros((3, 6))
        # Bottom face
        dN[0, 0] = -(1 - zeta)  # dN1/dxi
        dN[1, 0] = -(1 - zeta)  # dN1/deta
        dN[2, 0] = -(1 - xi - eta)  # dN1/dzeta

        dN[0, 1] = 1 - zeta  # dN2/dxi
        dN[1, 1] = 0  # dN2/deta
        dN[2, 1] = -xi  # dN2/dzeta

        dN[0, 2] = 0  # dN3/dxi
        dN[1, 2] = 1 - zeta  # dN3/deta
        dN[2, 2] = -eta  # dN3/dzeta

        # Top face
        dN[0, 3] = -zeta  # dN4/dxi
        dN[1, 3] = -zeta  # dN4/deta
        dN[2, 3] = 1 - xi - eta  # dN4/dzeta

        dN[0, 4] = zeta  # dN5/dxi
        dN[1, 4] = 0  # dN5/deta
        dN[2, 4] = xi  # dN5/dzeta

        dN[0, 5] = 0  # dN6/dxi
        dN[1, 5] = zeta  # dN6/deta
        dN[2, 5] = eta  # dN6/dzeta

        return dN

    # Number of integration points for triangular base and 1D extrusion
    tri_points, tri_weights = gauss_points_2d(order=2)
    z_points, z_weights = gauss_points_1d(order=2)

    # Initialize stiffness matrix Ke and strain-displacement matrix Be
    Ke = np.zeros((18, 18), dtype=np.float32)
    Be = np.zeros((6, 18), dtype=np.float32)

    for z, wz in zip(
        z_points, z_weights
    ):  # Loop over 1D (extrusion) integration points
        for (xi, eta), wt in zip(
            tri_points, tri_weights
        ):  # Loop over 2D triangular points
            dN_dxi = shape_function_derivatives(xi, eta, z)

            # for i, xi_pt in enumerate(gauss_points):
            #     for j, eta_pt in enumerate(gauss_points):
            #         for k, zeta_pt in enumerate(gauss_points):
            #             dN_dxi = shape_function_derivatives(xi_pt, eta_pt, zeta_pt)
            # Shape function derivatives (linear basis functions)
            J = dN_dxi @ nodes
            detJ = np.linalg.det(J)
            if detJ <= 0:
                print(
                    "Warning: Jacobian determinant is non-positive. It is likely something is wrong with the element geometry. Check the node order"
                )
                np_print("nodes", nodes)
                # np_print("dN_dxi", dN_dxi)
                # np_print("J", J)
                np_print("detJ", detJ)
                detJ = np.abs(detJ)
                raise ValueError(
                    "Jacobian determinant is non-positive, check element quality."
                )
            invJ = np.linalg.inv(J)

            # Shape function derivatives in global coordinates
            dN_dx = invJ @ dN_dxi

            # Construct strain-displacement matrix B
            for n in range(6):
                Be[0, n * 3] = dN_dx[0, n]
                Be[1, n * 3 + 1] = dN_dx[1, n]
                Be[2, n * 3 + 2] = dN_dx[2, n]
                Be[3, n * 3] = dN_dx[1, n]
                Be[3, n * 3 + 1] = dN_dx[0, n]
                Be[4, n * 3 + 1] = dN_dx[2, n]
                Be[4, n * 3 + 2] = dN_dx[1, n]
                Be[5, n * 3] = dN_dx[2, n]
                Be[5, n * 3 + 2] = dN_dx[0, n]

            # Integrate stiffness matrix
            Ke += Be.T @ C @ Be * detJ * wz * wt

    # Compute stress-displacement matrix CBe
    CBe = C @ Be

    return Ke, Be, CBe


@dataclass
class FEAModel:
    """
    Parameters
    ----------
    nodes : npt.ArrayLike[np.float32]
        Nx3 array where N is the number of nodes (control points). Most rows represent a node with 3 translational degrees of freedom (a point in space).
        Some [virtual] nodes will exist to represent 3 rotational degrees of freedom.
        The nodes array will be flattened into a neutral displacement vector.

    elements : sequence[sequence[int]]
        NxX array where each row is an index array referencing the nodes in the mesh.
        N is the number of elements the mesh and X is variable length depending on the cell type.

    element_type : npt.ArrayLike[ElementType]
        Nx0 array of element types. See the ElementType enum for possible values.
        N is a divisor of the number of elements in the mesh. This array will be repeated until it matches the number of elements.

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
        N is a divisor of the number of elements in the mesh. This array will be repeated until it matches the number of elements.

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
        self.nodes = np.asarray(self.nodes, dtype=np.float32).reshape(-1, NODES_WIDTH)
        self.num_nodes = len(self.nodes)
        self.num_dofs = self.nodes.size
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

        # if self.constraints_matrix is None:
        #     self.constraints_matrix = np.eye(self.num_dofs)
        # else:
        #     self.constraints_matrix = np.asarray(self.constraints_matrix)
        #     assert self.constraints_matrix.shape == (self.num_dofs, self.num_dofs)

        if self.forces is None:
            self.forces = np.zeros((self.nodes.shape), dtype=np.float32)
        else:
            self.forces = np.asarray(self.forces, dtype=np.float32).reshape(
                (-1, NODES_WIDTH)
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
            self.von_mises_stress = np.asarray(self.von_mises_stress, dtype=np.float32)
        if self.Kg is not None:
            self.Kg = np.asarray(self.Kg, dtype=np.float32)

    def _normalize_repeat_array(self, x, name, dtype, shape):
        x = np.asarray(x, dtype=dtype).reshape(shape)
        if self.num_elements % x.shape[0] != 0:
            raise ValueError(
                f"The size of {name} (shape={x.shape}) must be a divisor of the number of elements ({self.num_elements})."
            )
        return np.repeat(x, self.num_elements // x.size)

    def assemble_global_tensors(self):
        rows = []
        cols = []
        data = []
        self.element_tensors = []

        for element, element_type, properties in tqdm(
            zip(self.elements, self.element_type, self.element_properties),
            total=self.num_elements,
            desc="Assembling global tensors",
        ):
            if element_type == ElementType.RIGID:
                self.element_tensors.append((None, None, None))
                continue
            element_tensor = ElementType.element_tensors(
                element_type, self.nodes[element], *properties
            )
            self.element_tensors.append(element_tensor)

            Ke, _, _ = element_tensor

            dof_indices = np.array(
                [
                    node_idx * NODES_WIDTH + j
                    for node_idx in element
                    for j in range(NODES_WIDTH)
                ]
            )

            for i, row_idx in enumerate(dof_indices):
                for j, col_idx in enumerate(dof_indices):
                    rows.append(row_idx)
                    cols.append(col_idx)
                    data.append(Ke[i, j])

        # Create the CSR matrix directly
        self.Kg = scipy.sparse.csr_matrix(
            (data, (rows, cols)), shape=(self.num_dofs, self.num_dofs)
        )
        debug_print(
            f"Memory usage of Kg as csr_matrix: {self.Kg.data.nbytes / 1024} kb"
        )

    def compute_element_strain_stress(self):
        self.element_strains = np.zeros((self.num_elements, 6), dtype=np.float32)
        self.element_stresses = np.zeros((self.num_elements, 6), dtype=np.float32)
        for i, (element, element_type) in enumerate(
            zip(self.elements, self.element_type)
        ):
            if element_type == ElementType.RIGID:
                continue
            _, Be, CBe = self.element_tensors[i]
            u = self.displacement_vector[element].flatten()
            assert (
                u.shape[0] == Be.shape[1]
            ), f"element_type={element_type} u.shape={u.shape}, Be.shape={Be.shape}"
            assert (
                u.shape[0] == CBe.shape[1]
            ), f"u.shape={u.shape}, CBe.shape={CBe.shape}"
            self.element_strains[i] = Be @ u
            self.element_stresses[i] = CBe @ u

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

    def assemble_constraint_jacobian(self):
        rigid_elements = []
        for element, element_type in zip(self.elements, self.element_type):
            if element_type != ElementType.RIGID:
                continue
            rigid_elements.append(element)
            print("Rigid element", element_type, element, sep="\n")
            # print("Rigid element", element_type, element, self.nodes[element], sep="\n")

        # Calculate the number of constraints
        n_constraints = 0
        for element in rigid_elements:
            _, _, *daughter_translations = self.nodes[element]
            n_constraints += 3 * len(daughter_translations)

        n_dofs = self.nodes.size
        # Use LIL format for efficient row-based sparse matrix assembly
        G = scipy.sparse.lil_matrix((n_constraints, n_dofs))

        # Populate the constraint Jacobian for all rigid elements
        constraint_row = 0
        for element in rigid_elements:
            origin_translation, origin_rotation, *daughter_translations = self.nodes[
                element
            ]
            (
                origin_translation_node_idx,
                origin_rotation_node_idx,
                *daughter_translation_node_idxs,
            ) = element

            for idx, (daughter_translation, daughter_node_idx) in enumerate(
                zip(daughter_translations, daughter_translation_node_idxs)
            ):
                idx3 = constraint_row * 3  # Row offset for current constraint

                # Translation constraints
                G[
                    idx3 : idx3 + 3,
                    3 * origin_translation_node_idx : 3 * origin_translation_node_idx
                    + 3,
                ] = -np.eye(3)
                G[
                    idx3 : idx3 + 3,
                    3 * daughter_node_idx : 3 * daughter_node_idx + 3,
                ] = np.eye(3)

                # Rotational constraints
                r = daughter_translation - origin_translation
                rot = np.array(
                    [
                        [0, -r[2], r[1]],
                        [r[2], 0, -r[0]],
                        [-r[1], r[0], 0],
                    ]
                )
                G[
                    idx3 : idx3 + 3,
                    3 * origin_rotation_node_idx : 3 * origin_rotation_node_idx + 3,
                ] = -rot

                constraint_row += (
                    1  # Move to the next set of rows for the next constraint
                )

        # Convert G to CSR format for efficient operations
        return G.tocsr()

    def solve(self, use_iterative_solver=False, use_preconditioner=True, **args):
        start = time.time()
        debug_print("Start of stiffness matrix assembly")
        self.assemble_global_tensors()
        debug_print(
            f"End of stiffness matrix assembly, elapsed time: {time.time() - start}"
        )

        assert self.forces.shape == self.nodes.shape
        assert self.constraints_vector.shape == self.nodes.shape

        # Apply rigid constraints
        G_sparse = self.assemble_constraint_jacobian()

        n_constraints = G_sparse.shape[0]

        # Assemble full augmented matrix as sparse
        n_dofs = self.Kg.shape[0]
        K = self.Kg  # Global stiffness matrix (already sparse)

        # Assemble the augmented matrix A
        top = scipy.sparse.hstack([K, G_sparse.T])  # [K | G^T]
        bottom = scipy.sparse.hstack(
            [G_sparse, scipy.sparse.csr_matrix((n_constraints, n_constraints))]
        )  # [G | 0]
        A = scipy.sparse.vstack([top, bottom])  # Combine the rows: [K | G^T; G | 0]
        A = A.tocsr()  # Convert to CSR format for efficient operations

        # Assemble the right-hand side vector b
        b = np.zeros(A.shape[0])
        b[:n_dofs] = self.forces.flatten()  # External forces

        # Drop fixed DOFs from the augmented system
        fixed_dofs = np.where(self.constraints_vector.flatten() == 1)[0]
        active_dofs = np.setdiff1d(np.arange(n_dofs), fixed_dofs)

        # Reduce the augmented system
        free_augmented_dofs = np.concatenate(
            [active_dofs, np.arange(n_dofs, A.shape[0])]
        )  # Include constraint rows
        A_reduced = A[free_augmented_dofs, :][:, free_augmented_dofs]
        b_reduced = b[free_augmented_dofs]

        # Solve the reduced system
        start = time.time()
        debug_print("Start of solve")

        if use_iterative_solver:
            # Iterative solver (Conjugate Gradient)
            debug_print("Using iterative solver (Conjugate Gradient)")
            if use_preconditioner:
                debug_print("Creating Jacobi preconditioner")
                diag = A_reduced.diagonal()  # Extract diagonal
                if np.any(diag == 0):
                    debug_np_print("A_reduced", A_reduced)

                    raise ValueError(
                        "Jacobi preconditioner failed: matrix has zero diagonal entries."
                    )

                # Create the preconditioner as a scipy.sparse.linalg.LinearOperator
                M = scipy.sparse.linalg.LinearOperator(
                    A_reduced.shape, matvec=lambda x: x / diag, dtype=np.float32
                )
                u, info = scipy.sparse.linalg.cg(A_reduced, b_reduced, M=M, **args)
            else:
                u, info = scipy.sparse.linalg.cg(A_reduced, b_reduced, **args)
            if info != 0:
                raise ValueError(f"Conjugate Gradient did not converge, info: {info}")
        else:
            # Direct solver
            debug_print("Using direct solver (scipy.sparse.linalg.spsolve)")
            u = scipy.sparse.linalg.spsolve(A_reduced, b_reduced)

        debug_print(f"End of solve, elapsed time: {time.time() - start}")

        # Reconstruct full displacement vector
        displacements = np.zeros((n_dofs,), dtype=np.float32)
        displacements[active_dofs] = u[: len(active_dofs)]
        self.displacement_vector = displacements.reshape(self.nodes.shape)
        debug_np_print("Mesh.displacement_vector", self.displacement_vector)

        # Compute forces (Kg is sparse, @ is efficient for sparse matrices)
        forces = (self.Kg @ displacements.flatten()).reshape(self.nodes.shape)
        self.forces = forces
        debug_np_print("Mesh.forces", self.forces)

        # Compute strains and stresses
        self.compute_element_strain_stress()
        debug_np_print("Mesh.element_strains", self.element_strains)
        debug_np_print("Mesh.element_stresses", self.element_stresses)

        # Compute von Mises stress
        self.compute_von_mises_stress()
        debug_np_print("Mesh.von_mises_stress", self.von_mises_stress)

    def solve_precondition(self, use_iterative_solver=False, **args):
        start = time.time()
        debug_print("Start of stiffness matrix assembly")
        self.assemble_global_tensors()
        debug_print(
            f"End of stiffness matrix assembly, elapsed time: {time.time() - start}"
        )

        # Determine free degrees of freedom
        free_dofs = np.where(self.constraints_vector.flatten() == 0)[0]

        # Extract reduced stiffness matrix and force vector
        K = self.Kg[free_dofs, :][:, free_dofs]
        f = self.forces.flatten()[free_dofs]

        # Debug sparse memory usage
        debug_print(f"Memory usage of K reduced: {K.data.nbytes / 1024} kb")

        # Solve the linear system
        start = time.time()
        debug_print("Start of solve")

        if use_iterative_solver:
            # Iterative solver (Conjugate Gradient) with Jacobi Preconditioning
            debug_print("Using iterative solver (Conjugate Gradient)")

            # Jacobi preconditioner (diagonal scaling)
            debug_print("Creating Jacobi preconditioner")
            diag = K.diagonal()  # Extract diagonal
            if np.any(diag == 0):
                raise ValueError(
                    "Jacobi preconditioner failed: matrix has zero diagonal entries."
                )

            # Create the preconditioner as a scipy.sparse.linalg.LinearOperator
            M = scipy.sparse.linalg.LinearOperator(
                K.shape, matvec=lambda x: x / diag, dtype=np.float32
            )

            u, info = scipy.sparse.linalg.cg(K, f, M=M, **args)
            if info != 0:
                raise ValueError(f"Conjugate Gradient did not converge, info: {info}")
        else:
            # Direct solver
            debug_print("Using direct solver (scipy.sparse.linalg.spsolve)")
            u = scipy.sparse.linalg.spsolve(K, f)

        debug_print(f"End of solve, elapsed time: {time.time() - start}")

        # Reconstruct full displacement vector
        displacements = np.zeros((self.num_dofs,), dtype=np.float32)
        displacements[free_dofs] = u
        self.displacement_vector = displacements.reshape(self.nodes.shape)
        debug_np_print("Mesh.displacement_vector", self.displacement_vector)

        # Compute forces (Kg is sparse, @ is efficient for sparse matrices)
        forces = (self.Kg @ displacements.flatten()).reshape(self.nodes.shape)
        self.forces = forces
        debug_np_print("Mesh.forces", self.forces)

        # Compute strains and stresses
        self.compute_element_strain_stress()
        debug_np_print("Mesh.element_strains", self.element_strains)
        debug_np_print("Mesh.element_stresses", self.element_stresses)

        # Compute von Mises stress
        self.compute_von_mises_stress()
        debug_np_print("Mesh.von_mises_stress", self.von_mises_stress)

    def generate_pv_unstructured_mesh(self, displacement_scale=1.0):
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
        cells = []
        cell_type = []
        for element, element_type in zip(self.elements, self.element_type):
            if element_type not in cell_type_map:
                raise ValueError(f"Unknown element type: {element_type}")
            if element_type == ElementType.BEAM2:
                element = [element[0], element[2]]
            if element_type == ElementType.RIGID:
                continue
            cells.extend([len(element), *element])
            cell_type.extend([cell_type_map[element_type]])
        points = (
            self.nodes + self.displacement_vector * displacement_scale
        )  # TODO figure out a better way for amplifying displacements for rendering

        print("cells", cells, sep="\n")
        print("cell_type", cell_type, sep="\n")
        print("points", points.shape, points, sep="\n")

        # Create the spatial reference
        grid = pv.UnstructuredGrid(cells, cell_type, points)

        if self.displacement_vector is not None:
            grid.point_data["displacement"] = np.linalg.norm(
                self.displacement_vector, axis=1
            )

        # if self.von_mises_stress is not None:
        #     grid.cell_data["von_mises_stress"] = self.von_mises_stress

        return grid

    def _gen_pv_arrows(self, cent, direction, mag=1, **kwargs):
        import pyvista
        from pyvista import _vtk
        from pyvista.core.utilities.helpers import wrap

        """Generate arrows for the plotter.

        Parameters
        ----------
        cent : np.ndarray
            Array of centers.

        direction : np.ndarray
            Array of direction vectors.

        mag : float, optional
            Amount to scale the direction vectors.

        **kwargs : dict, optional
            See :func:`pyvista.Plotter.add_mesh` for optional
            keyword arguments.

        Returns
        -------
        pyvista.Actor
            Actor of the arrows.

        Examples
        --------
        Plot a random field of vectors and save a screenshot of it.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> rng = np.random.default_rng(seed=0)
        >>> cent = rng.random((10, 3))
        >>> direction = rng.random((10, 3))
        >>> arrows = pv.core.Mesh()._gen_pv_arrows(cent, direction, mag=2)
        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(arrows)
        >>> plotter.show()

        """
        if cent.shape != direction.shape:  # pragma: no cover
            raise ValueError("center and direction arrays must have the same shape")

        direction = direction.copy()
        if cent.ndim != 2:
            cent = cent.reshape((-1, 3))

        if direction.ndim != 2:
            direction = direction.reshape((-1, 3))

        if mag != 1:
            direction = direction * mag

        pdata = pyvista.vector_poly_data(cent, direction)
        # Create arrow object
        arrow = _vtk.vtkArrowSource()
        arrow.Update()
        glyph3D = _vtk.vtkGlyph3D()
        glyph3D.SetSourceData(arrow.GetOutput())
        glyph3D.SetInputData(pdata)
        glyph3D.SetVectorModeToUseVector()
        glyph3D.Update()

        arrows = wrap(glyph3D.GetOutput())
        return arrows

    def generate_pv_force_arrows(self, max_size=0.1):
        """
        Generate force arrows for visualization.

        Parameters:
        - max_size: float, maximum size of the largest arrow in the grid dimensions.

        Returns:
        - arrows: pyvista.PolyData, arrows representing the forces.
        """
        linear_nodes = np.ones((self.nodes.shape[0],), dtype=np.bool)
        for element, element_type in zip(self.elements, self.element_type):
            # Exclude rotational DOFs
            if element_type == ElementType.BEAM2:
                linear_nodes[[element[1], element[3]]] = False

        nodes = self.nodes[linear_nodes]
        forces = self.forces[linear_nodes]
        # Calculate the bounding box dimensions
        min_coords = np.min(nodes, axis=0)
        max_coords = np.max(nodes, axis=0)
        bounding_box_dims = max_coords - min_coords
        max_dim = np.max(bounding_box_dims)

        # Calculate the scaling factor based on the maximum force and max_size
        max_force = np.max(np.linalg.norm(forces, axis=1))
        scale_factor = max_size * max_dim / max_force

        # Create arrows
        arrows = self._gen_pv_arrows(nodes, forces, mag=scale_factor)

        return arrows


# class Mesh:
#     """
#     Defines a 3D mesh for finite element analysis.

#     Parameters
#     ----------
#     nodes : npt.ArrayLike[np.float32]
#         Nx3 array where N is the number of nodes (control points). Most rows represent a node with 3 translational degrees of freedom (a point in space).
#         Some [virtual] nodes will exist to represent 3 rotational degrees of freedom.
#         The nodes array will be flattened into a neutral displacement vector.

#     elements : sequence[sequence[int]]
#         NxX array where each row is an index array referencing the nodes in the mesh.
#         N is the number of elements the mesh and X is variable length depending on the cell type.
#     """

#     nodes: npt.NDArray[np.float32]
#     elements: Sequence[Sequence[int]]
