import core
import numpy as np
import pyvista as pv
from math import ceil

np.set_printoptions(precision=3, linewidth=400, suppress=False)

core.DEBUG = True


def test_truss1():
    E = 10_000_000  # psi
    A = 1  # in^2

    force = 1000  # lbf

    mesh = core.FEAModel(
        nodes=[
            [0, 0, 0],
            [0, 1, 0],
        ],
        elements=[
            [0, 1],
        ],
        element_type=[core.ElementType.ROD],
        element_properties=[
            [E, A],
        ],
        constraints_vector=[
            [1, 1, 1],
            [1, 0, 1],
        ],
    )

    mesh.forces[1, 1] = force

    mesh.solve()

    assert np.allclose(
        mesh.forces,
        force
        * np.array(
            [
                [0, -1, 0],
                [0, 1, 0],
            ]
        ),
    )

    expected_stress = force / A
    expected_displacement = expected_stress / E

    assert np.allclose(
        mesh.displacement_vector[1, 1], expected_displacement
    ), f"Displacement is not correct, expected: {expected_displacement} but got: {mesh.displacement_vector[1,1]}"

    assert np.allclose(
        mesh.von_mises_stress, [expected_stress]
    ), f"Stress is not correct, expected: \n{expected_stress}\nbut got: \n{mesh.element_stresses}"

    pv.set_plot_theme("dark")

    plotter = pv.Plotter()
    grid = mesh.generate_pv_unstructured_mesh()
    plotter.add_mesh(
        grid,
        show_edges=True,
        line_width=10,
        show_vertices=True,
    )

    arrows = mesh.generate_pv_force_arrows()
    plotter.add_mesh(
        arrows,
        color="red",
    )

    # # Calculate max arrow size as 10% of the grid size (grid length is the diagonal of the grid)
    # max_force = np.max(np.linalg.norm(mesh.forces, axis=1))

    # # Add forces as arrows
    # plotter.add_arrows(
    #     mesh.nodes,
    #     mesh.forces,
    #     color="red",
    #     mag=0.1 * grid.length / max_force,
    # )

    plotter.show_grid()
    plotter.show()


def test_truss():
    E = 70_000_000_000
    nu = 0
    A = (10**-2) ** 2
    I = 1
    J = 1

    force = 100

    mesh = core.FEAModel(
        nodes=[
            [0, 1, 0],
            [1, 0, 0],
            [0, -1, 0],
        ],
        elements=[
            [0, 1],
            [1, 2],
        ],
        element_type=[core.ElementType.ROD],
        element_properties=[
            [E, A],
        ],
        constraints_vector=[
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
        ],
    )

    mesh.forces[1, 1] = -force

    mesh.solve()

    assert np.allclose(
        mesh.forces,
        force
        * np.array(
            [
                [-0.5, 0.5, 0],
                [0, -1, 0],
                [0.5, 0.5, 0],
            ]
        ),
    )

    # mesh.compute_element_strain_stress()

    root2 = np.sqrt(2)

    # sqrt((Fx/2)^2 + (Fy/2)^2) / A
    stress = force / 2 * root2 / A
    # the top element is in tension, the bottom element is in compression
    expected_stress = np.array([stress, stress])

    assert np.allclose(
        mesh.von_mises_stress, expected_stress
    ), f"Stress is not correct, expected: \n{expected_stress}\nbut got: \n{mesh.von_mises_stress}"

    pv.set_plot_theme("dark")

    plotter = pv.Plotter()
    grid = mesh.generate_pv_unstructured_mesh()
    plotter.add_mesh(
        grid,
        show_edges=True,
        line_width=10,
        show_vertices=True,
    )
    arrows = mesh.generate_pv_force_arrows()
    plotter.add_mesh(
        arrows,
        color="red",
    )

    plotter.show_grid()
    plotter.show()


def test_cantilever_truss():
    # Material and geometric properties
    E = 10_000_000  # Young's modulus in psi
    A = 1  # Cross-sectional area in in²
    force = 500  # Applied force in lbf

    # Define the cantilevered truss mesh
    #
    # 0--2--4
    # | /| /
    # |/ |/
    # 1--3
    mesh = core.FEAModel(
        nodes=[
            [0, 0, 0],  # Node 0: Fixed support
            [0, -24, 0],  # Node 1 : Fixed support
            [24, 0, 0],  # Node 2
            [24, -24, 0],  # Node 3
            [48, 0, 0],  # Node 4: Load applied here
        ],
        elements=[
            [0, 2],
            [0, 1],
            [1, 2],
            [1, 3],
            [3, 2],
            [2, 4],
            [3, 4],
        ],
        element_type=[core.ElementType.ROD],
        element_properties=[
            [E, A],  # All elements share the same properties
        ],
        constraints_vector=[
            [1, 1, 1],  # Node 0: Fully fixed
            [1, 0, 1],  # Node 1: Free to move in Y
            [0, 0, 1],  # Node 2: Free to move in X and Y
            [0, 0, 1],  # Node 3: Free to move in X and Y
            [0, 0, 1],  # Node 4: Free to move in X and Y
        ],
    )

    # Apply point load at Node 3
    mesh.forces[4, 1] = -force  # Load in the negative Y direction

    # Solve the system
    mesh.solve()

    # Assert the reaction forces are as expected at the fixed node
    # Reaction forces should balance the applied load
    expected_reactions = np.zeros((5, 3))
    expected_reactions[1, 0] = 48 * force / 24
    expected_reactions[0, 0] = -48 * force / 24
    expected_reactions[0, 1] = force  # Force in the Y direction at the fixed node
    expected_reactions[4, 1] = -force
    assert np.allclose(
        mesh.forces,
        expected_reactions,
        atol=1e-3,  # Absolute tolerance
        rtol=1e-4,  # Relative tolerance
    ), f"Reaction forces incorrect. Expected: \n{expected_reactions}\nGot: \n{mesh.forces}"

    # Visualize the truss using PyVista

    plotter = pv.Plotter()

    pvmesh = mesh.generate_pv_unstructured_mesh()
    plotter.add_mesh(
        pvmesh,
        scalars="displacement",
        show_edges=True,
        cmap="viridis",
        line_width=10,
        show_vertices=True,
    )

    arrows = mesh.generate_pv_force_arrows()
    plotter.add_mesh(
        arrows,
        color="red",
    )

    plotter.show_grid()
    plotter.show()


def test_cantilever_beam():
    # Material and geometric properties
    E = 10_000_000_000  # Young's modulus in psi
    nu = 0.3  # Poisson's ratio
    A = 1  # Cross-sectional area in in²
    L = np.array(
        [5, 0.1, 0],
    )  # in2
    J = 4.0

    force = 1000  # Applied force in lbf
    n_elements = 5

    nodes = [[x, 0, 0, 0, 0, 0] for x in np.linspace(0, 48, n_elements + 1)]
    nodes = np.array(nodes, dtype=np.float32)
    print("nodes", nodes.shape, nodes, sep="\n")
    elements = [[i, i + 1] for i in range(n_elements)]
    elements = np.array(elements, dtype=np.int32)
    print("elements", elements.shape, elements, sep="\n")
    constraints = np.zeros(nodes.shape, dtype=int)
    constraints = np.array([[0, 0, 0, 0, 0, 0]]).repeat(n_elements + 1, axis=0)
    constraints[0] = np.array([1, 1, 1, 1, 1, 1])
    print("constraints", constraints.shape, constraints, sep="\n")

    mesh = core.FEAModel(
        nodes=nodes,
        elements=elements,
        element_type=[core.ElementType.BEAM2],
        element_properties=[[E, nu, A, L, J]],
        constraints_vector=constraints,
    )

    mesh.forces[n_elements, 1] = -force

    mesh.solve()

    plotter = pv.Plotter()

    pvmesh = mesh.generate_pv_unstructured_mesh()
    plotter.add_mesh(
        pvmesh,
        scalars="von_mises_stress",
        show_edges=True,
        cmap="viridis",
        line_width=10,
        show_vertices=True,
    )
    arrows = mesh.generate_pv_force_arrows()
    plotter.add_mesh(
        arrows,
        color="red",
    )

    plotter.show_grid()
    plotter.show()


def test_single_hex8():
    E = 10_000_000  # Young's modulus in psi
    nu = 0
    force = 500  # Applied force in lbf

    nodes = np.array(
        [
            [0, 0, 0],  # Node 1
            [1, 0, 0],  # Node 4
            [1, 1, 0],  # Node 3
            [0, 1, 0],  # Node 2
            [0, 0, 1],  # Node 5
            [1, 0, 1],  # Node 8
            [1, 1, 1],  # Node 7
            [0, 1, 1],  # Node 6
        ]
    )

    elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    constraints = np.zeros(nodes.shape, dtype=np.int8)
    # Fix all translational DOFs at the bottom face
    constraints[:4, :] = 1

    print("nodes", nodes.shape, nodes, sep="\n")
    print("elements", elements.shape, elements, sep="\n")
    print("constraints", constraints.shape, constraints, sep="\n")

    mesh = core.FEAModel(
        nodes=nodes,
        elements=elements,
        element_type=[core.ElementType.HEX8],
        element_properties=[
            [E, nu],  # All elements share the same properties
        ],
        constraints_vector=constraints,
    )

    # Apply point load at Node 3
    mesh.forces[-1, 1] = force  # Load in the negative Y direction

    # Solve the system
    mesh.solve()

    plotter = pv.Plotter()

    pvmesh = mesh.generate_pv_unstructured_mesh()

    plotter.add_mesh(
        pvmesh,
        scalars="displacement",
        show_edges=True,
        cmap="viridis",
    )
    arrows = mesh.generate_pv_force_arrows()
    plotter.add_mesh(
        arrows,
        color="red",
    )

    plotter.show_grid()
    plotter.show()


def test_cantilever_beam_hex8():
    # Material and geometric properties
    E = 10_000_000  # Young's modulus in psi
    nu = 0
    force = 500  # Applied force in lbf

    # Define the cantilevered truss mesh
    # 6--7--8
    # |  |  |
    # 3--4--5
    # |  |  |
    # 0--1--2...

    element_size = 0.5

    x_size = 2
    y_size = 2
    z_size = 10

    n_elements_x = int(ceil(x_size / element_size))
    n_elements_y = int(ceil(y_size / element_size))
    n_elements_z = int(ceil(z_size / element_size))
    n_nodes_x = n_elements_x + 1
    n_nodes_y = n_elements_y + 1
    n_nodes_z = n_elements_z + 1

    x = np.linspace(0, x_size, n_nodes_x)
    y = np.linspace(0, y_size, n_nodes_y)
    z = np.linspace(0, z_size, n_nodes_z)
    nodes = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    elements = np.array(
        [
            [
                i + j * n_nodes_x + k * n_nodes_x * n_nodes_y,
                i + (j + 1) * n_nodes_x + k * n_nodes_x * n_nodes_y,
                i + 1 + (j + 1) * n_nodes_x + k * n_nodes_x * n_nodes_y,
                i + 1 + j * n_nodes_x + k * n_nodes_x * n_nodes_y,
                i + j * n_nodes_x + (k + 1) * n_nodes_x * n_nodes_y,
                i + (j + 1) * n_nodes_x + (k + 1) * n_nodes_x * n_nodes_y,
                i + 1 + (j + 1) * n_nodes_x + (k + 1) * n_nodes_x * n_nodes_y,
                i + 1 + j * n_nodes_x + (k + 1) * n_nodes_x * n_nodes_y,
            ]
            for k in range(n_elements_z)
            for j in range(n_elements_y)
            for i in range(n_elements_x)
        ]
    )

    constraints = np.zeros(nodes.shape, dtype=np.int8)
    # Fix all translational DOFs at the bottom face
    constraints[: n_nodes_x * n_nodes_y, :] = 1

    print("nodes", nodes.shape, nodes, sep="\n")
    print("elements", elements.shape, elements, sep="\n")
    print("constraints", constraints.shape, constraints, sep="\n")

    mesh = core.FEAModel(
        nodes=nodes,
        elements=elements,
        element_type=[core.ElementType.HEX8],
        element_properties=[
            [E, nu],  # All elements share the same properties
        ],
        constraints_vector=constraints,
    )

    # plotter = pv.Plotter()

    # pvmesh = mesh.generate_pv_unstructured_mesh()

    # plotter.add_mesh(
    #     pvmesh,
    #     show_edges=True,
    # )

    # plotter.show_grid()
    # plotter.show()

    # Apply point load at Node 3
    mesh.forces[-1, 1] = force  # Load in the negative Y direction

    # Solve the system
    mesh.solve()

    # # Assert the reaction forces are as expected at the fixed node
    # # Reaction forces should balance the applied load
    # expected_reactions = np.zeros((5, 6))
    # expected_reactions[1, 0] = 48 * force / 24
    # expected_reactions[0, 0] = -48 * force / 24
    # expected_reactions[0, 1] = force  # Force in the Y direction at the fixed node
    # expected_reactions[4, 1] = -force
    # assert np.allclose(
    #     mesh.forces,
    #     expected_reactions,
    #     atol=1e-3,  # Absolute tolerance
    #     rtol=1e-4,  # Relative tolerance
    # ), f"Reaction forces incorrect. Expected: \n{expected_reactions}\nGot: \n{mesh.forces}"

    plotter = pv.Plotter()

    pvmesh = mesh.generate_pv_unstructured_mesh()

    plotter.add_mesh(
        pvmesh,
        scalars="von_mises_stress",
        show_edges=True,
        cmap="viridis",
    )
    arrows = mesh.generate_pv_force_arrows()
    plotter.add_mesh(
        arrows,
        color="red",
    )

    plotter.show_grid()
    plotter.show()


def test_prism6_1elm():
    """
    Test function for the Mesh class using prism6 elements.
    """

    # Material and geometric properties
    E = 10_000_000  # Young's modulus in psi
    nu = 0.3  # Poisson's ratio
    force = 100  # Applied force in lbf

    #    2
    #  / |
    # 0--1

    nodes = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    )

    print("nodes", nodes.shape, nodes, sep="\n")

    elements = np.array([[0, 1, 2, 3, 4, 5]])
    print("elements", elements.shape, elements, sep="\n")

    constraints = np.zeros(nodes.shape, dtype=np.int8)
    # Fix all translational DOFs at the bottom face
    constraints[:3, :] = 1

    print("constraints", constraints.shape, constraints, sep="\n")
    # Initialize the mesh
    mesh = core.FEAModel(
        nodes=nodes,
        elements=elements,
        element_type=[core.ElementType.PRISM6],
        element_properties=[[E, nu]],  # All elements share the same properties
        constraints_vector=constraints,
    )

    mesh.forces[-1, 1] = -force  # Apply force in the -Y direction

    mesh.solve()

    # Visualization (optional)
    plotter = pv.Plotter()
    pvmesh = mesh.generate_pv_unstructured_mesh()

    plotter.add_mesh(
        pvmesh,
        scalars="displacement",
        show_edges=True,
        cmap="viridis",
    )
    arrows = mesh.generate_pv_force_arrows()
    plotter.add_mesh(
        arrows,
        color="red",
    )

    plotter.show_grid()
    plotter.show()


def test_prism6_2elm():
    """
    Test function for the Mesh class using prism6 elements.
    """

    # Material and geometric properties
    E = 10_000_000  # Young's modulus in psi
    nu = 0.3  # Poisson's ratio
    force = 100  # Applied force in lbf

    # 3--2
    # |/ |
    # 0--1

    nodes = np.array(
        [
            [0, 0, 0],  # Node 0
            [1, 0, 0],  # Node 1
            [1, 1, 0],  # Node 2
            [0, 1, 0],  # Node 3
            [0, 0, 1],  # Node 4
            [1, 0, 1],  # Node 5
            [1, 1, 1],  # Node 6
            [0, 1, 1],  # Node 7
        ]
    )

    print("nodes", nodes.shape, nodes, sep="\n")

    elements = np.array([[0, 1, 2, 4, 5, 6], [0, 2, 3, 4, 6, 7]])
    print("elements", elements.shape, elements, sep="\n")

    constraints = np.zeros(nodes.shape, dtype=np.int8)
    # Fix all translational DOFs at the bottom face
    constraints[:3, :] = 1

    print("constraints", constraints.shape, constraints, sep="\n")
    # Initialize the mesh
    mesh = core.FEAModel(
        nodes=nodes,
        elements=elements,
        element_type=[core.ElementType.PRISM6],
        element_properties=[[E, nu]],  # All elements share the same properties
        constraints_vector=constraints,
    )

    mesh.forces[-1, 1] = -force  # Apply force in the -Y direction

    mesh.solve()

    # Visualization (optional)
    plotter = pv.Plotter()
    pvmesh = mesh.generate_pv_unstructured_mesh()

    plotter.add_mesh(
        pvmesh,
        scalars="displacement",
        show_edges=True,
        cmap="viridis",
    )
    arrows = mesh.generate_pv_force_arrows()
    plotter.add_mesh(
        arrows,
        color="red",
    )

    plotter.show_grid()
    plotter.show()


def tube_prism6():
    # Material and geometric properties
    E = 10_000_000  # Young's modulus in psi
    nu = 0.3  # Poisson's ratio
    force = 100  # Applied force in lbf

    outer_radius = 4.0
    thickness = 0.05
    inner_radius = outer_radius - thickness
    height = 5.0

    elm_size = 0.05

    n_points = int(ceil(2 * np.pi * outer_radius / elm_size))
    n_layers = int(ceil(height / elm_size))

    # n_points = 40
    # n_layers = 40

    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    theta_out = theta
    theta_out = theta + np.pi / n_points
    theta_in = theta
    # theta_in = theta - np.pi / n_points

    outer_surface = np.array(
        [[outer_radius * np.cos(t), outer_radius * np.sin(t), 0] for t in theta_out]
    )

    inner_surface = np.array(
        [[inner_radius * np.cos(t), inner_radius * np.sin(t), 0] for t in theta_in]
    )

    # Initialize an empty list to hold nodes
    nodes = []

    # Generate layers by stacking surfaces along the z-axis
    for z in np.linspace(0, height, n_layers + 1):
        outer_layer = outer_surface.copy()
        outer_layer[:, 2] = z  # Update z-coordinate
        inner_layer = inner_surface.copy()
        inner_layer[:, 2] = z  # Update z-coordinate
        nodes.append(outer_layer)
        nodes.append(inner_layer)

    # Convert list of layers to a single numpy array
    nodes = np.vstack(nodes)

    elements = []
    for layer in range(n_layers):
        for i in range(n_points):
            first_point_in_layer = layer * 2 * n_points
            # Node indices for outer and inner layers
            o0 = first_point_in_layer + i
            o1 = (
                first_point_in_layer + (i + 1) % n_points
            )  # Wrap to the start of the layer
            i0 = o0 + n_points
            i1 = o1 + n_points

            # Node indices for the next layer
            o0_next = o0 + 2 * n_points
            o1_next = o1 + 2 * n_points
            i0_next = i0 + 2 * n_points
            i1_next = i1 + 2 * n_points

            # Add wedge elements (6-node elements)
            elements.append([o0, o1, i1, o0_next, o1_next, i1_next])  # Outer wedge
            elements.append([i1, i0, o0, i1_next, i0_next, o0_next])  # Inner wedge

    # Convert elements to numpy array
    elements = np.array(elements)

    print("nodes", nodes.shape, nodes, sep="\n")
    print("elements", elements.shape, elements, sep="\n")

    constraints = np.zeros(nodes.shape, dtype=np.int8)
    constraints[: 2 * n_points, :] = 1

    print("constraints", constraints.shape, constraints, sep="\n")

    mesh = core.FEAModel(
        nodes=nodes,
        elements=elements,
        element_type=[core.ElementType.PRISM6],
        element_properties=[[E, nu]],  # All elements share the same properties
        constraints_vector=constraints,
    )

    mesh.forces[
        n_points * 2 * n_layers : n_points * 2 * (n_layers + 1), 1
    ] = -force  # Apply force in the -Y direction

    print(mesh.forces)

    mesh.solve(use_iterative_solver=True)

    plotter = pv.Plotter()
    m = mesh.generate_pv_unstructured_mesh()
    plotter.add_mesh(m, scalars="von_mises_stress", cmap="viridis")
    m = mesh.generate_pv_force_arrows()
    plotter.add_mesh(m, color="red")

    poly = pv.PolyData(nodes)
    poly["id"] = np.arange(len(nodes))
    # plotter.add_point_labels(poly, "id")

    plotter.show_grid()
    plotter.show()


def test_beam_point_load():
    wy = 4
    wz = 1
    # R = 1  # in
    E = 10_000_000
    nu = 0.3

    from geometry import second_moment_of_rect

    Izz, Iyy, J, A = second_moment_of_rect(wy, wz)
    print("Izz, Iyy, J, A", Izz, Iyy, J, A)
    L = 10
    theta = 0
    L = 16 * 12  # in
    w = 160  # lb

    dy = w * L**3 / (3 * E * Iyy)
    n_elements = 100
    n_nodes = n_elements + 1
    x = np.linspace(0, L, n_nodes)
    nodes = np.hstack([x.reshape(-1, 1), np.zeros((n_nodes, 2))])
    rot_nodes = np.zeros_like(nodes)
    nodes = np.vstack([nodes, rot_nodes])
    # print("nodes", nodes.shape, nodes, sep="\n")
    forces = np.zeros_like(nodes)
    forces[n_nodes - 1, 1] = +w
    # print("forces", forces.shape, forces, sep="\n")

    constraints = np.zeros_like(nodes)
    constraints[0, :] = 1
    constraints[n_nodes, :] = 1
    # print("constraints", constraints.shape, constraints, sep="\n")

    elements = [[i, i + n_nodes, i + 1, i + n_nodes + 1] for i in range(n_elements)]
    # print("elements", elements, sep="\n")

    # core.DEBUG = True
    fea = core.FEAModel(
        nodes,
        elements,
        element_type=[core.ElementType.BEAM2],
        element_properties=[(E, nu, Iyy, Izz, A, J, 0)],
        forces=forces,
        constraints_vector=constraints,
    )

    fea.solve()

    print("dy", dy)
    print("displacement", fea.displacement_vector[n_nodes - 1, 1], sep="\n")

    assert np.isclose(dy, fea.displacement_vector[n_nodes - 1, 1], atol=1e-1)


def test_beam_distributed_load():
    wy = 4
    wz = 1
    # R = 1  # in
    E = 10_000_000
    nu = 0.3

    from geometry import second_moment_of_rect

    Izz, Iyy, J, A = second_moment_of_rect(wy, wz)
    print("Izz, Iyy, J, A", Izz, Iyy, J, A)
    L = 10
    theta = 0
    L = 16 * 12  # in
    w = 160  # lb
    # Uniform load
    q = w / L
    print("q", q)
    dy = q * L**4 / (8 * E * Iyy)
    n_elements = 100
    n_nodes = n_elements + 1
    x = np.linspace(0, L, n_nodes)
    nodes = np.hstack([x.reshape(-1, 1), np.zeros((n_nodes, 2))])
    rot_nodes = np.zeros_like(nodes)
    nodes = np.vstack([nodes, rot_nodes])
    # print("nodes", nodes.shape, nodes, sep="\n")
    forces = np.zeros_like(nodes)
    # forces[n_nodes - 1, 1] = +w
    forces[:n_nodes, 1] = w / (n_nodes - 1)
    # print("forces", forces.shape, forces, sep="\n")

    constraints = np.zeros_like(nodes)
    constraints[0, :] = 1
    constraints[n_nodes, :] = 1
    # print("constraints", constraints.shape, constraints, sep="\n")

    elements = [[i, i + n_nodes, i + 1, i + n_nodes + 1] for i in range(n_elements)]
    # print("elements", elements, sep="\n")

    # core.DEBUG = True
    fea = core.FEAModel(
        nodes,
        elements,
        element_type=[core.ElementType.BEAM2],
        element_properties=[(E, nu, Iyy, Izz, A, J, 0)],
        forces=forces,
        constraints_vector=constraints,
    )

    fea.solve()

    print("dy", dy)
    print("displacement", fea.displacement_vector[n_nodes - 1, 1], sep="\n")

    assert np.isclose(dy, fea.displacement_vector[n_nodes - 1, 1], atol=1e-1)

def test_shell_bending():
    # Material and geometric properties
    E = 10_000_000  # Young's modulus in psi
    nu = 0.3  # Poisson's ratio
    force = 100  # Applied force in lbf
    thickness = 0.05  # Shell thickness in inches

    outer_radius = 4.0  # Outer radius in inches
    height = 10.0  # Beam length in inches

    # Mesh resolution
    elm_size = 0.5  # Element size in inches
    n_points = int(ceil(2 * np.pi * outer_radius / elm_size))  # Circumferential divisions
    n_layers = int(ceil(height / elm_size))  # Longitudinal divisions

    # Node coordinates for a thin-walled tube
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    nodes = []

    for z in np.linspace(0, height, n_layers + 1):  # Layer along the beam height
        layer_nodes = [[outer_radius * np.cos(t), outer_radius * np.sin(t), z] for t in theta]
        for node in layer_nodes:
            nodes.append(node)  # Add translational DOF node
            nodes.append([0.0, 0.0, 0.0])  # Add rotational DOF placeholder

    nodes = np.array(nodes)

    # Element connectivity for 4-node quadrilateral shell elements
    elements = []
    for layer in range(n_layers):
        for i in range(n_points):
            first_point_in_layer = layer * n_points * 2
            n0 = first_point_in_layer + i * 2
            n1 = first_point_in_layer + ((i + 1) % n_points) * 2  # Wrap around
            n2 = n0 + n_points * 2
            n3 = n1 + n_points * 2
            elements.append([n0, n1, n3, n2])  # Counter-clockwise quad definition

    elements = np.array(elements)

    # Thickness vector (magnitude and direction for each node)
    thickness_vector = np.array([[thickness, 0.0, 0.0]] * 4)

    # Boundary conditions: fix the bottom layer of the tube
    constraints = np.zeros(nodes.shape, dtype=np.int8)
    constraints[:n_points * 2:2, :] = 1  # Fix translational DOFs of the bottom layer
    constraints[:n_points * 2:2, 3:] = 1  # Fix rotational DOFs of the bottom layer

    # Create the FEA model
    fea = core.FEAModel(
        nodes=nodes,
        elements=elements,
        element_type=[core.ElementType.SHELL],
        element_properties=[[E, nu, thickness_vector]],  # Pass material properties and thickness vector
        constraints_vector=constraints,
    )

    # Apply a distributed load to the top layer in the negative Y direction
    top_layer_start = n_points * n_layers * 2
    fea.forces[top_layer_start:top_layer_start + n_points * 2:2, 1] = -force / n_points

    # Solve the model
    fea.solve(use_iterative_solver=True)

    # Visualization with PyVista
    plotter = pv.Plotter()
    m = fea.generate_pv_unstructured_mesh()
    plotter.add_mesh(m, scalars="von_mises_stress", cmap="viridis", show_edges=True)

    # Add force arrows for visualization
    m = fea.generate_pv_force_arrows()
    plotter.add_mesh(m, color="red")

    # Add point labels (optional)
    poly = pv.PolyData(nodes[::2])  # Only label translational nodes
    poly["id"] = np.arange(len(nodes[::2]))
    # plotter.add_point_labels(poly, "id")

    plotter.show_grid()
    plotter.show()


# test_truss1()
# test_truss()
# test_cantilever_truss()
# # test_cantilever_beam() # Doesn't work
# test_single_hex8()
# test_cantilever_beam_hex8()
# test_prism6_1elm()
# test_prism6_2elm()
# # tube_prism6()
# test_beam_point_load()
# test_beam_distributed_load()
test_shell_bending()