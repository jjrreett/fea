import numpy as np
import matplotlib.pyplot as plt

# Beam properties
E = 210e9  # Young's modulus in Pascals
I = 1e-6  # Moment of inertia in m^4
L = 1.0  # Length of the beam in meters
q = 1000  # Uniform load in N/m

# Number of elements
n_elements = 100
n_nodes = n_elements + 1

# Element length
element_length = L / n_elements

# Initialize global stiffness matrix and load vector
global_stiffness_matrix = np.zeros((2 * n_nodes, 2 * n_nodes))
load_vector = np.zeros(2 * n_nodes)

# Element stiffness matrix for Euler-Bernoulli beam
element_stiffness_matrix = (E * I / element_length**3) * np.array(
    [
        [12, 6 * element_length, -12, 6 * element_length],
        [
            6 * element_length,
            4 * element_length**2,
            -6 * element_length,
            2 * element_length**2,
        ],
        [-12, -6 * element_length, 12, -6 * element_length],
        [
            6 * element_length,
            2 * element_length**2,
            -6 * element_length,
            4 * element_length**2,
        ],
    ]
)

# Assemble global stiffness matrix and load vector
for i in range(n_elements):
    # Global DOF indices for the element
    dof = [2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3]

    # Add element stiffness matrix to global stiffness matrix
    for a in range(4):
        for b in range(4):
            global_stiffness_matrix[dof[a], dof[b]] += element_stiffness_matrix[a, b]

    # Add element load vector to global load vector
    load_vector[dof] += (
        q
        * element_length
        / 2
        * np.array([1, element_length / 6, 1, -element_length / 6])
    )

# Apply boundary conditions (fixed at both ends)
# 0, 1 displacement, rotation respectively
fixed_dofs = [0, 1, 2 * n_nodes - 2, 2 * n_nodes - 1]
free_dofs = list(set(range(2 * n_nodes)) - set(fixed_dofs))

# Reduce system of equations
K_reduced = global_stiffness_matrix[np.ix_(free_dofs, free_dofs)]
F_reduced = load_vector[free_dofs]

# Solve for displacements
U_reduced = np.linalg.solve(K_reduced, F_reduced)

# Insert displacements back into global displacement vector
displacement_vector = np.zeros(2 * n_nodes)
displacement_vector[free_dofs] = U_reduced

# Compute bending moments and shear forces
moment_vector = np.zeros(n_nodes)
shear_vector = np.zeros(n_nodes)
for i in range(n_elements):
    dof = [2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3]
    u_e = displacement_vector[dof]
    moment_vector[i] = (
        E
        * I
        / element_length**2
        * (
            12 * u_e[0]
            - 6 * element_length * u_e[1]
            - 12 * u_e[2]
            + 6 * element_length * u_e[3]
        )
    )
    shear_vector[i] = (
        E
        * I
        / element_length**3
        * (
            6 * element_length * u_e[0]
            + 2 * element_length**2 * u_e[1]
            - 6 * element_length * u_e[2]
            + 4 * element_length**2 * u_e[3]
        )
    )

print("Displacements:", displacement_vector)
print("Bending Moments:", moment_vector)
print("Shear Forces:", shear_vector)

x = np.linspace(0, L, n_nodes)

plt.figure(figsize=(12, 8))

# Plot displacements
plt.subplot(3, 1, 1)
plt.plot(x, displacement_vector[::2], marker="o")
plt.title("Displacements")
plt.xlabel("Position along the beam (m)")
plt.ylabel("Displacement (m)")

# Plot bending moments
plt.subplot(3, 1, 2)
plt.plot(x, moment_vector, marker="o")
plt.title("Bending Moments")
plt.xlabel("Position along the beam (m)")
plt.ylabel("Bending Moment (Nm)")

# Plot shear forces
plt.subplot(3, 1, 3)
plt.plot(x, shear_vector, marker="o")
plt.title("Shear Forces")
plt.xlabel("Position along the beam (m)")
plt.ylabel("Shear Force (N)")

plt.tight_layout()
plt.show()
