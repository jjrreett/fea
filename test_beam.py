from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from math import pi
from typing import Optional, Tuple
import time

import numpy as np
import numpy.typing as npt
import pyvista as pv
from tqdm import tqdm

import scipy.sparse
import scipy.sparse.linalg
from lib.core import beam2_tensors
from lib import core

np.set_printoptions(precision=3, linewidth=400, suppress=True)


def second_moment_of_area_circular(r):
    pir4_2 = np.pi * (r**4) / 2
    pir4_4 = pir4_2 / 2
    Izz = pir4_4
    Iyy = pir4_4
    Ixx = pir4_2
    A = np.pi * (r**2)
    return Izz, Iyy, Ixx, A


def second_moment_of_area_annulus(ro, ri):
    pir4_2 = np.pi * (ro**4 - ri**4) / 2
    pir4_4 = pir4_2 / 2
    Izz = pir4_4
    Iyy = pir4_4
    Ixx = pir4_2
    A = np.pi * (ro**2 - ri**2)
    return Izz, Iyy, Ixx, A


def second_moment_of_area_tube(r, t):
    pir3t = 2 * np.pi * (r**3) * t
    pir3t_2 = pir3t / 2
    Izz = pir3t_2
    Iyy = pir3t_2
    Ixx = pir3t
    A = 2 * np.pi * r * t
    return Izz, Iyy, Ixx, A


def second_moment_of_rect(wy, wz):
    Iyy = wz * wy**3 / 12
    Izz = wz**3 * wy / 12
    Ixx = wz * wy * (wz**2 + wy**2) / 12
    A = wz * wy
    return Izz, Iyy, Ixx, A


wy = 4
wz = 1
# R = 1  # in
E = 10_000_000
nu = 0.3

Izz, Iyy, J, A = second_moment_of_rect(wy, wz)
print("Izz, Iyy, J, A", Izz, Iyy, J, A)
L = 10
theta = 0

# # 3D Case

# nodes = np.array([[0, 0, 0], [0, 0, 0], [L, 0, 0], [0, 0, 0]], dtype=np.float32)
# num_dofs = nodes.size

# Ke, _, _ = beam2_tensors(nodes, E, nu, Iyy, Izz, A, J, theta)
# print("Ke", Ke.shape, Ke, sep="\n")

# forces = np.zeros_like(nodes)
# forces[2, 0] = 1
# forces[2, 1] = -1
# forces[2, 2] = -1

# constraints = np.zeros_like(nodes)
# constraints[0, :] = 1  # constrain the translation of the first node
# constraints[1, :] = 1  # constrain the rotation of the first node

# free_dofs = np.where(constraints.flatten() == 0)[0]


# K_reduced = Ke[free_dofs, :][:, free_dofs]
# f = forces.flatten()[free_dofs]
# u = np.linalg.solve(K_reduced, f)
# displacements = np.zeros((num_dofs,), dtype=np.float32)
# displacements[free_dofs] = u
# forces = (Ke @ displacements).reshape(nodes.shape)
# displacements = displacements.reshape(nodes.shape)

# print("displacements", displacements.shape, displacements, sep="\n")
# print("forces", forces.shape, forces, sep="\n")

# Fx, Fy, Fz = forces[2]
# dy = Fy * L**3 / (3 * E * Iyy)
# dz = Fz * L**3 / (3 * E * Izz)

# print("dy", dy)
# print("dz", dz)


L = 16 * 12  # in
w = 160  # lb
# Uniform load
q = w / L
print("q", q)
dy = q * L**4 / (8 * E * Iyy)
# ClearCalc's predicts δ=−0.0138

# point load
# dy = w * L**3 / (3 * E * Iyy)

displacements = []

for i in range(1, 100):
    n_elements = i
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
    displacements.append((n_nodes, fea.displacement_vector[n_nodes - 1, 1]))

for n_nodes, dy in displacements:
    print(n_nodes, dy)

# plt = pv.Plotter()
# plt.add_mesh(fea.generate_pv_unstructured_mesh(), line_width=20, show_vertices=True)
# plt.add_mesh(fea.generate_pv_force_arrows())
# plt.show_grid()
# plt.show()
