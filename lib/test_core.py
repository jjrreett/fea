import core
import numpy as np
import pyvista as pv

np.set_printoptions(precision=3, linewidth=400, suppress=False)


def test_truss1():
    E = 10_000_000  # psi
    A = 1  # in^2

    force = 1000  # lbf

    mesh = core.Mesh(
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

    mesh.assemble_global_tensors()

    mesh.forces[1, 1] = force

    mesh.solve()
    print("displacement_vector", mesh.displacement_vector, sep="\n")
    print("forces", mesh.forces, sep="\n")

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

    mesh.compute_element_strain_stress()

    expected_stress = force / A
    expected_displacement = expected_stress / E

    print("strain", mesh.strain, sep="\n")
    print("stress", mesh.element_stresses, sep="\n")

    mesh.compute_von_mises_stress()
    print("von_mises_stress", mesh.von_mises_stress, sep="\n")
    # print("element_stress", mesh.element_stresses, sep="\n")

    assert np.allclose(
        mesh.displacement_vector[1, 1], expected_displacement
    ), f"Displacement is not correct, expected: {expected_displacement} but got: {mesh.displacement_vector[1,1]}"

    assert np.allclose(
        mesh.von_mises_stress, [expected_stress]
    ), f"Stress is not correct, expected: \n{expected_stress}\nbut got: \n{mesh.element_stresses}"

    pv.set_plot_theme("dark")

    # plotter = pv.Plotter()
    # grid = mesh.generate_pv_unstructured_mesh()
    # plotter.add_mesh(
    #     grid,
    #     show_edges=True,
    #     line_width=10,
    #     show_vertices=True,
    # )

    # # Calculate max arrow size as 10% of the grid size (grid length is the diagonal of the grid)
    # max_force = np.max(np.linalg.norm(mesh.forces, axis=1))

    # # Add forces as arrows
    # plotter.add_arrows(
    #     mesh.nodes,
    #     mesh.forces,
    #     color="red",
    #     mag=0.1 * grid.length / max_force,
    # )

    # plotter.show_grid()
    # plotter.show()


def test_truss():
    E = 70_000_000_000
    nu = 0
    A = (10**-2) ** 2
    I = 1
    J = 1

    force = 100

    mesh = core.Mesh(
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

    mesh.assemble_global_tensors()

    mesh.forces[1, 1] = -force

    mesh.solve()
    print("displacement_vector", mesh.displacement_vector, sep="\n")
    print("forces", mesh.forces, sep="\n")

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

    mesh.compute_element_strain_stress()

    root2 = np.sqrt(2)

    # sqrt((Fx/2)^2 + (Fy/2)^2) / A
    stress = force / 2 * root2 / A
    # the top element is in tension, the bottom element is in compression
    expected_stress = np.array([stress, stress])
    # stress along x y axis (stress * normalized direction vectors)

    # local_displacement = expected_stress_vector / E

    mesh.compute_element_strain_stress()
    print("element_strains", mesh.element_strains, sep="\n")
    print("element_stresses", mesh.element_stresses, sep="\n")

    mesh.compute_von_mises_stress()
    print("von_mises_stress", mesh.von_mises_stress, sep="\n")

    assert np.allclose(
        mesh.von_mises_stress, expected_stress
    ), f"Stress is not correct, expected: \n{expected_stress}\nbut got: \n{mesh.von_mises_stress}"

    pv.set_plot_theme("dark")

    # plotter = pv.Plotter()
    # grid = mesh.generate_pv_unstructured_mesh()
    # plotter.add_mesh(
    #     grid,
    #     show_edges=True,
    #     line_width=10,
    #     show_vertices=True,
    # )

    # # Calculate max arrow size as 10% of the grid size (grid length is the diagonal of the grid)
    # max_force = np.max(np.linalg.norm(mesh.forces, axis=1))

    # # Add forces as arrows
    # plotter.add_arrows(
    #     mesh.nodes,
    #     mesh.forces,
    #     color="red",
    #     mag=0.1 * grid.length / max_force,
    # )

    # plotter.show_grid()
    # plotter.show()


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
    mesh = core.Mesh(
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

    # Assemble global stiffness matrix
    mesh.assemble_global_tensors()

    # Apply point load at Node 3
    mesh.forces[4, 1] = -force  # Load in the negative Y direction

    # Solve the system
    mesh.solve()
    print("displacement_vector", mesh.displacement_vector, sep="\n")
    print("forces", mesh.forces, sep="\n")

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

    mesh.compute_element_strain_stress()

    # Print the results
    print("element_strains", mesh.element_strains, sep="\n")
    print("element_stresses", mesh.element_stresses, sep="\n")

    mesh.compute_von_mises_stress()
    print("von_mises_stress", mesh.von_mises_stress, sep="\n")

    # Visualize the truss using PyVista

    plotter = pv.Plotter()

    pvmesh = mesh.generate_pv_unstructured_mesh()

    pvmesh.point_data["displacement"] = np.linalg.norm(mesh.displacement_vector, axis=1)

    pvmesh.cell_data["von_mises_stress"] = mesh.von_mises_stress

    plotter.add_mesh(
        pvmesh,
        scalars="displacement",
        show_edges=True,
        cmap="viridis",
        line_width=10,
        show_vertices=True,
    )

    # Calculate max arrow size as 10% of the grid size
    max_force = np.max(np.linalg.norm(mesh.forces, axis=1))

    # Add forces as arrows
    plotter.add_arrows(
        mesh.nodes,
        mesh.forces,
        color="red",
        mag=0.1 * pvmesh.length / max_force,
    )

    plotter.show_grid()
    plotter.show()


test_truss1()
test_truss()
test_cantilever_truss()


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

    mesh = core.Mesh(
        nodes=nodes,
        elements=elements,
        element_type=[core.ElementType.BEAM2],
        element_properties=[[E, nu, A, L, J]],
        constraints_vector=constraints,
    )

    mesh.forces[n_elements, 1] = -force

    mesh.assemble_global_tensors()
    mesh.solve()
    mesh.compute_element_strain_stress()
    mesh.compute_von_mises_stress()

    print("displacement_vector", mesh.displacement_vector, sep="\n")
    print("forces", mesh.forces, sep="\n")
    print("element_strains", mesh.element_strains, sep="\n")
    print("element_stresses", mesh.element_stresses, sep="\n")
    print("von_mises_stress", mesh.von_mises_stress, sep="\n")

    plotter = pv.Plotter()

    pvmesh = mesh.generate_pv_unstructured_mesh()

    pvmesh.point_data["displacement"] = np.linalg.norm(
        mesh.displacement_vector[:, :3], axis=1
    )

    pvmesh.cell_data["von_mises_stress"] = mesh.von_mises_stress

    plotter.add_mesh(
        pvmesh,
        scalars="von_mises_stress",
        show_edges=True,
        cmap="viridis",
        line_width=10,
        show_vertices=True,
    )

    # Calculate max arrow size as 10% of the grid size
    max_force = np.max(np.linalg.norm(mesh.forces[:, :3], axis=1))

    # Add forces as arrows
    plotter.add_arrows(
        mesh.nodes[:, :3],
        mesh.forces[:, :3],
        color="red",
        mag=0.1 * pvmesh.length / max_force,
    )

    plotter.show_grid()
    plotter.show()


# Doesn't work
# test_cantilever_beam()

from math import ceil


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

    mesh = core.Mesh(
        nodes=nodes,
        elements=elements,
        element_type=[core.ElementType.HEX8],
        element_properties=[
            [E, nu],  # All elements share the same properties
        ],
        constraints_vector=constraints,
    )
    mesh.assemble_global_tensors()

    # Apply point load at Node 3
    mesh.forces[-1, 1] = force  # Load in the negative Y direction

    # Solve the system
    mesh.solve()
    mesh.compute_element_strain_stress()
    mesh.compute_von_mises_stress()
    print("displacement_vector", mesh.displacement_vector, sep="\n")
    print("forces", mesh.forces, sep="\n")

    plotter = pv.Plotter()

    pvmesh = mesh.generate_pv_unstructured_mesh()

    pvmesh.point_data["displacement"] = np.linalg.norm(mesh.displacement_vector, axis=1)

    pvmesh.cell_data["von_mises_stress"] = mesh.von_mises_stress

    plotter.add_mesh(
        pvmesh,
        scalars="displacement",
        show_edges=True,
        cmap="viridis",
    )

    # Calculate max arrow size as 10% of the grid size
    max_force = np.max(np.linalg.norm(mesh.forces, axis=1))

    # Add forces as arrows
    plotter.add_arrows(
        mesh.nodes,
        mesh.forces,
        color="red",
        mag=0.1 * pvmesh.length / max_force,
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

    mesh = core.Mesh(
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

    # Assemble global stiffness matrix
    mesh.assemble_global_tensors()

    # Apply point load at Node 3
    mesh.forces[-1, 1] = force  # Load in the negative Y direction

    # Solve the system
    mesh.solve()
    print("displacement_vector", mesh.displacement_vector, sep="\n")
    print("forces", mesh.forces, sep="\n")

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

    # Compute element stresses and strains
    mesh.compute_element_strain_stress()
    mesh.compute_von_mises_stress()

    # Print the results
    print("element_strains", mesh.element_strains, sep="\n")
    print("element_stresses", mesh.element_stresses, sep="\n")
    print("von_mises_stress", mesh.von_mises_stress, sep="\n")

    plotter = pv.Plotter()

    pvmesh = mesh.generate_pv_unstructured_mesh()

    pvmesh.point_data["displacement"] = np.linalg.norm(mesh.displacement_vector, axis=1)

    pvmesh.cell_data["von_mises_stress"] = mesh.von_mises_stress

    plotter.add_mesh(
        pvmesh,
        scalars="von_mises_stress",
        show_edges=True,
        cmap="viridis",
    )

    # Calculate max arrow size as 10% of the grid size
    max_force = np.max(np.linalg.norm(mesh.forces, axis=1))

    # Add forces as arrows
    plotter.add_arrows(
        mesh.nodes,
        mesh.forces,
        color="red",
        mag=0.1 * pvmesh.length / max_force,
    )

    plotter.show_grid()
    plotter.show()


def test_prism6():
    """
    Test function for the Mesh class using prism6 elements.
    """

    # Material and geometric properties
    E = 10_000_000  # Young's modulus in psi
    nu = 0.3  # Poisson's ratio
    force = 1000  # Applied force in lbf

    # Define a small test mesh
    x_size = 1.0
    y_size = 1.0
    z_size = 1.0
    element_size = 0.5

    # 6--7--8
    # |/ |/ |
    # 3--4--5
    # |/ |/ |
    # 0--1--2...

    n_elements_x = int(np.ceil(x_size / element_size))
    n_elements_y = int(np.ceil(y_size / element_size))
    n_elements_z = int(np.ceil(z_size / element_size))
    n_nodes_x = n_elements_x + 1
    n_nodes_y = n_elements_y + 1
    n_nodes_z = n_elements_z + 1

    x = np.linspace(0, x_size, n_nodes_x)
    y = np.linspace(0, y_size, n_nodes_y)
    z = np.linspace(0, z_size, n_nodes_z)
    nodes = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    nodes = np.hstack((nodes, np.zeros((nodes.shape[0], 3))))  # Add rotational DOFs

    print("nodes", nodes.shape, nodes, sep="\n")

    # Generate prism6 elements (triangular prisms)
    elements = []
    for k in range(n_elements_z):  # Loop over Z layers (vertical stacking)
        for j in range(n_elements_y - 1):  # Loop over Y layers (rows of triangles)
            for i in range(
                n_elements_x - 1
            ):  # Loop over X layers (columns of triangles)
                # Indices for the lower triangle
                n0 = i + j * n_nodes_x + k * n_nodes_x * n_nodes_y
                n1 = n0 + 1
                n2 = n0 + n_nodes_x

                # Indices for the upper triangle (directly above lower triangle)
                n3 = n0 + n_nodes_x * n_nodes_y
                n4 = n1 + n_nodes_x * n_nodes_y
                n5 = n2 + n_nodes_x * n_nodes_y

                # Define the prism element
                elements.append([n0, n1, n2, n3, n4, n5])

    elements = np.array(elements)
    print("elements", elements.shape, elements, sep="\n")

    constraints = np.zeros((nodes.shape[0], 6), dtype=np.int8)
    # Fix all translational DOFs at the bottom face
    constraints[: n_nodes_x * n_nodes_y, :3] = 1
    # Fix all rotational DOFs
    constraints[:, 3:] = 1

    print("constraints", constraints.shape, constraints, sep="\n")
    # Initialize the mesh
    mesh = core.Mesh(
        nodes=nodes,
        elements=elements,
        element_type=[core.ElementType.PRISM6],
        element_properties=[[E, nu]],  # All elements share the same properties
        constraints_vector=constraints,
    )

    # Assemble global stiffness matrix
    mesh.assemble_global_tensors()

    # Apply point load at the top center node
    top_center_node = -1  # Assuming the last node is at the top center
    mesh.forces[top_center_node, 1] = -force  # Apply force in the -Y direction

    # Solve the system
    mesh.solve()

    # Compute element stresses and strains
    mesh.compute_element_strain_stress()
    mesh.compute_von_mises_stress()

    # Visualization (optional)
    plotter = pv.Plotter()

    pvmesh = mesh.generate_pv_unstructured_mesh()
    pvmesh.point_data["displacement"] = np.linalg.norm(
        mesh.displacement_vector[:, :3], axis=1
    )
    pvmesh.cell_data["von_mises_stress"] = mesh.von_mises_stress

    plotter.add_mesh(
        pvmesh,
        scalars="von_mises_stress",
        show_edges=True,
        cmap="viridis",
    )

    max_force = np.max(np.linalg.norm(mesh.forces[:, :3], axis=1))
    plotter.add_arrows(
        mesh.nodes[:, :3],
        mesh.forces[:, :3],
        color="red",
        mag=0.1 * pvmesh.length / max_force,
    )

    plotter.show_grid()
    plotter.show()

    print("Prism6 mesh test passed successfully!")


test_single_hex8()
test_cantilever_beam_hex8()
# test_prism6()
