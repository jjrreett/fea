import numpy as np
from lib.geometry import circle
from lib import core
import numpy.typing as npt
import pyvista as pv

np.set_printoptions(precision=3, linewidth=400, suppress=True)


def vertices_2d_to_3d(vertices: npt.NDArray) -> npt.NDArray:
    return np.hstack((vertices, np.zeros((len(vertices), 1))))


# Nodes and constraints setup
hole_loc = [0, 0, 0]
n_points = 10
radius = 1
circle_vertices, circle_segments = circle(n_points, radius, *hole_loc[:2])
circle_vertices = vertices_2d_to_3d(circle_vertices)

circle_forces = circle_vertices
z_forces = np.zeros_like(circle_vertices)
# Force in z direction based on y distance from xz plane
z_forces[:, 2] = circle_vertices[:, 1]
print("z_forces", z_forces.shape, z_forces, sep="\n")


hole_rot = [0, 0, 0]
nodes = np.vstack((hole_loc, hole_rot, circle_vertices))
print("nodes", nodes.shape, nodes, sep="\n")

# External forces and moments
f = np.zeros_like(nodes)
f[2:] = z_forces
print("f (external forces)", f.shape, f, sep="\n")
f = f.flatten()

element = np.arange(n_points + 2)
origin_translation, origin_rotation, *daughter_translations = nodes[element]
(
    origin_translation_node_idx,
    origin_rotation_node_idx,
    *daughter_translation_node_idxs,
) = element


# Stiffness coefficients
k_t = 100.0  # Translational stiffness (N/m)
k_r = 100.0  # Rotational stiffness (N·m/rad)

# System stiffness matrix (identity for unconstrained DOFs)
n_dofs = nodes.size
K = np.zeros((n_dofs, n_dofs))
for i in range(3):  # Translational stiffness for origin
    K[3 * origin_translation_node_idx + i, 3 * origin_translation_node_idx + i] = k_t
for i in range(3):  # Rotational stiffness for origin
    K[3 * origin_rotation_node_idx + i, 3 * origin_rotation_node_idx + i] = k_r

print("K (stiffness matrix)", K.shape, K, sep="\n")


# Constraint Jacobian
n_constraints = 3 * len(daughter_translations)
G = np.zeros((n_constraints, n_dofs))

# Populate the constraint Jacobian
for idx, (daughter_translation, daughter_node_idx) in enumerate(
    zip(daughter_translations, daughter_translation_node_idxs)
):
    idx3 = idx * 3
    G[idx3 : idx3 + 3, 0:3] = -np.eye(3)
    G[idx3 : idx3 + 3, idx3 + 6 : idx3 + 9] = np.eye(3)
    rot = np.zeros((3, 3))
    r = daughter_translation - origin_translation

    rot = np.array(
        [
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0],
        ]
    )
    G[idx3 : idx3 + 3, 3:6] = -rot

print("G (constraint Jacobian)", G.shape, G, sep="\n")

# Assemble augmented matrix
A = np.zeros((K.shape[0] + n_constraints, K.shape[1] + n_constraints))
A[: K.shape[0], : K.shape[1]] = K  # Stiffness matrix
A[: K.shape[0], K.shape[1] :] = G.T  # Coupling with constraints
A[K.shape[0] :, : K.shape[1]] = G  # Constraint Jacobian

print("A (augmented matrix)", A.shape, A, sep="\n")

# Right-hand side vector
b = np.zeros(A.shape[0])
b[: f.size] = f  # External forces

# Solve the augmented system
solution = np.linalg.solve(A, b)

# Extract displacements and Lagrange multipliers
u = solution[: K.shape[1]].reshape(-1, 3)
lambda_ = solution[K.shape[1] :]

print("u (displacements)", u.shape, u, sep="\n")
print("Lagrange multipliers", lambda_.shape, lambda_, sep="\n")

# Exclude node 1 (origin_rotation) from the mesh nodes and forces
exclude_idx = 1
mesh_nodes = np.delete(nodes + u, exclude_idx, axis=0)
mesh_forces = np.delete(f.reshape(-1, 3), exclude_idx, axis=0)
mesh = pv.PolyData(mesh_nodes)

# mesh = pv.PolyData()
# mesh.points = nodes

plotter = pv.Plotter()
plotter.add_mesh(mesh, point_size=10, show_vertices=True)
plotter.add_arrows(mesh_nodes, mesh_forces, mag=0.1)
plotter.show_grid()
plotter.show()
