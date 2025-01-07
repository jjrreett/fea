from typing import Sequence
import numpy as np
import pyvista as pv
import triangle as tr
import numpy.typing as npt
from naca_airfoil import naca_points, naca_cutouts
import matplotlib.pyplot as plt
from lib import core

from lib.geometry import circle


def vertices_2d_to_3d(vertices: npt.NDArray) -> npt.NDArray:
    return np.hstack((vertices, np.zeros((len(vertices), 1))))


def rotation_matrix2d(angle, degrees=True):
    if degrees:
        angle = np.radians(angle)
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def rotation_matrix3d(angle, axis=2, degrees=True):
    if degrees:
        angle = np.radians(angle)
    c = np.cos(angle)
    s = np.sin(angle)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 2:
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError("Invalid axis")


class Mesh:
    points: npt.NDArray
    point_data: dict[str, npt.NDArray]

    _geometries: list["Geometry"]

    def __init__(self):
        self.points = np.zeros((0, 3))
        self.point_data = {}
        self._geometries = []

    def new_geometry(self, points: npt.ArrayLike = None, cells=None, cell_types=None):
        # TODO allow for repeating cell types array to match n_cells
        g = Geometry(self)

        if points is None:
            offset = 0
        else:
            offset = len(self.points)
            self.points = np.vstack((self.points, np.asarray(points)))

        if cells is not None:
            g.cells = [[idx + offset for idx in e] for e in cells]

        if cell_types is not None:
            g.cell_types = cell_types

        self._geometries.append(g)
        return g

    def get_unstructured_grid(self):
        cells = []
        cell_types = []
        for g in self._geometries:
            # TODO go straight to numpy arrays
            cells.extend(g.get_cells_vtk())
            cell_types.extend(list(g.cell_types))
        try:
            grid = pv.UnstructuredGrid(cells, cell_types, self.points)
        except Exception as e:
            print("cells", cells, sep="\n")
            print("cell_types", cell_types, sep="\n")
            print("points", self.points, sep="\n")
            raise e
        return grid


class Geometry:
    mesh: "Mesh"
    cells: Sequence[Sequence[int]]
    cell_types: Sequence[int]
    cell_data: dict[str, npt.NDArray]

    def __init__(self, mesh: "Mesh"):
        self.mesh = mesh
        self.cell_data = {}

    def get_cells_vtk(self):
        return self.to_cells_vtk(self.cells)

    @staticmethod
    def to_cells_vtk(cells):
        # TODO go straight to numpy arrays
        new_cells = []
        for e in cells:
            new_cells.append(len(e))
            new_cells.extend(e)
        return new_cells

    @staticmethod
    def from_cells_vtk(cells_1d):
        cells = []
        idx = 0
        while idx < len(cells_1d):
            length = cells_1d[idx]
            cells.append(cells_1d[idx + 1 : idx + 1 + length])
            idx += length + 1
        return cells

    def get_unstructured_grid(self):
        cells = self.get_cells_vtk()
        cell_types = list(self.cell_types)
        grid = pv.UnstructuredGrid(cells, cell_types, self.mesh.points)
        grid.cell_data.update(self.cell_data)
        return grid

    def get_cell_midpoints(self):
        points = self.mesh.points[self.cells]
        mid_points = np.mean(points, axis=1)
        return mid_points

    def new_mesh_from_geometry(self) -> tuple[Mesh, "Geometry"]:
        flat_cells = [c for cell in self.cells for c in cell]
        unique_point_idxs, unique_point_idxs_map = np.unique(
            flat_cells, return_inverse=True
        )

        points = self.mesh.points[unique_point_idxs]

        # Manually reshape unique_point_idxs_map to match the structure of self.cells
        new_cells = []
        idx = 0
        for cell in self.cells:
            new_cell = unique_point_idxs_map[idx : idx + len(cell)]
            new_cells.append(new_cell.tolist())
            idx += len(cell)

        m = Mesh()
        m.points = points
        for key, data in self.mesh.point_data.items():
            m.point_data[key] = data[unique_point_idxs]

        g = m.new_geometry(cells=new_cells, cell_types=self.cell_types)
        for key, data in self.cell_data.items():
            g.cell_data[key] = data
        return m, g


def layer_points(points, z_heights) -> npt.NDArray:
    points = np.asarray(points)
    n_points = len(points)
    points = np.tile(points, (len(z_heights), 1))
    z_values = np.repeat(z_heights, n_points)
    points[:, 2] = z_values
    return points


def layer_geometry(geometry, z_heights) -> list[Geometry]:
    # Assume the mesh points are already layered
    n_points_per_layer = len(geometry.mesh.points) // len(z_heights)

    geometries = [geometry]
    for i in range(1, len(z_heights) - 1):
        cells = [[n + i * n_points_per_layer for n in cell] for cell in geometry.cells]
        geometries.append(
            mesh.new_geometry(cells=cells, cell_types=geometry.cell_types)
        )
    return geometries


def extrude_geometry(geometry: Geometry, z_heights) -> Geometry:
    n_points_per_layer = len(geometry.mesh.points) // len(z_heights)

    new_cells = []
    new_cell_types = []
    new_cell_data = {}

    cell_type_map = {
        pv.CellType.VERTEX: pv.CellType.LINE,
        pv.CellType.LINE: pv.CellType.PIXEL,
        pv.CellType.TRIANGLE: pv.CellType.WEDGE,
        pv.CellType.QUAD: pv.CellType.HEXAHEDRON,
    }

    for i in range(len(z_heights) - 1):
        for cell, cell_type in zip(geometry.cells, geometry.cell_types):
            new_cells.append(
                [n + i * n_points_per_layer for n in cell]
                + [n + (i + 1) * n_points_per_layer for n in cell]
            )
            new_cell_types.append(cell_type_map[cell_type])

    for key, data in geometry.cell_data.items():
        data = np.tile(data, len(z_heights) - 1)
        new_cell_data[key] = data

    new_geometry = mesh.new_geometry(cells=new_cells, cell_types=new_cell_types)
    new_geometry.cell_data = new_cell_data

    return new_geometry


def compute_area_normals(geometry) -> npt.NDArray:
    area_normals = []

    for cell_type, cell in zip(geometry.cell_types, geometry.cells):
        if cell_type == pv.CellType.TRIANGLE:
            area_normal = (
                np.cross(
                    mesh.points[cell[1]] - mesh.points[cell[0]],
                    mesh.points[cell[2]] - mesh.points[cell[0]],
                )
                / 2
            )
        elif cell_type == pv.CellType.PIXEL:
            # ASSUME pixels are orthogonal
            area_normal = np.cross(
                mesh.points[cell[1]] - mesh.points[cell[0]],
                mesh.points[cell[2]] - mesh.points[cell[0]],
            )
        elif cell_type == pv.CellType.QUAD:
            area_normal = np.cross(
                mesh.points[cell[1]] - mesh.points[cell[0]],
                mesh.points[cell[2]] - mesh.points[cell[0]],
            ) + np.cross(
                mesh.points[cell[2]] - mesh.points[cell[0]],
                mesh.points[cell[3]] - mesh.points[cell[0]],
            )
        else:
            raise ValueError(
                f"Unsupported cell type: {cell_type}. Only 2D cells are supported."
            )
        area_normals.append(area_normal)
    return np.array(area_normals)


mesh = Mesh()

# # Add airfoil part
n_points = 101
cord = 5.6 * 12
airfoil_points, t = naca_points(n_points, 0.2, 0.02, 0.12, return_t_points=True)
airfoil_points -= np.array([[0.25, 0]])  # move quarter chord to origin
# airfoil_points = (rotation_matrix2d(-10) @ airfoil_points.T).T
airfoil_points = vertices_2d_to_3d(airfoil_points)
airfoil_points *= cord

mean_segment_length = np.mean(np.linalg.norm(np.diff(airfoil_points, axis=0), axis=1))

airfoil = mesh.new_geometry(
    points=airfoil_points,
    cells=[(i, i + 1) for i in range(n_points - 1)] + [(n_points - 1, 0)],
    cell_types=[pv.CellType.LINE] * n_points,
)


def get_cp_data():
    """Generated by xfoil"""
    with open("cp_data.txt", "r") as f:
        airfoil_name = f.readline().strip()
        alpha_re_flap = f.readline().strip()

        # Extract alpha, Re, and flap info
        alpha_re_flap_parts = alpha_re_flap.split()
        alpha = float(alpha_re_flap_parts[2])
        re = float(alpha_re_flap_parts[5])
        xflap = float(alpha_re_flap_parts[8])
        yflap = float(alpha_re_flap_parts[9])

        # Load the rest of the data
        data = np.loadtxt(f, skiprows=1)

    # Extract coordinates and cp data
    vertices = data[:, 0:2]
    cp_data = data[:, 2]

    distance_between_vertices = np.sqrt(np.sum(np.diff(vertices, axis=0) ** 2, axis=1))

    dist = np.cumsum(distance_between_vertices)
    dist = np.insert(dist, 0, 0)

    dist = dist / np.max(dist)

    return vertices, dist, cp_data, alpha, re, xflap, yflap


cp_polar_cords, cp_polar_t, cp, alpha, *_ = get_cp_data()
cp = np.interp(t, cp_polar_t, cp)
cp_segments = np.mean(cp[airfoil.cells], axis=1)
airfoil.cell_data["cp"] = cp_segments


# # Add circle part
# hole_loc = [0, 2]
# r = 2
# circumference = 2 * np.pi * r
# mean_segment_length = np.mean(
#     np.linalg.norm(np.diff(mesh.points[airfoil.cells], axis=1).squeeze(), axis=1)
# )
# n_points = int(np.ceil(circumference / mean_segment_length))
# circle_vertices, circle_segments = circle(n_points, 2, *hole_loc)
# circle_vertices = vertices_2d_to_3d(circle_vertices)
# spar_cutout = mesh.new_geometry(
#     circle_vertices, circle_segments, [pv.CellType.LINE] * n_points
# )


# # Add spar/hole for meshing
# spar_point = mesh.new_geometry(
#     points=vertices_2d_to_3d([hole_loc]), cells=[[0]], cell_types=[pv.CellType.VERTEX]
# )


# A = {
#     "vertices": mesh.points[:, :2],
#     "segments": airfoil.cells + spar_cutout.cells,
#     "holes": mesh.points[spar_point.cells[0], :2],
# }

circles_x, circles_y, circles_r = naca_cutouts(0.2, 0.02, 0.12, 6, 1 / cord)
circles_x -= 0.25
circles_x *= cord
circles_y *= cord
circles_r *= cord

hole_locs = np.array(list(zip(circles_x, circles_y)))

spar_cutouts = []
# Add all holes to the mesh
for hole_x, hole_y, hole_r in zip(circles_x, circles_y, circles_r):
    # Calculate the circle properties
    circumference = 2 * np.pi * hole_r

    n_points = int(np.ceil(circumference / mean_segment_length))

    # Generate the circle vertices and segments
    circle_vertices, circle_segments = circle(n_points, hole_r, hole_x, hole_y)
    circle_vertices = vertices_2d_to_3d(circle_vertices)

    # Add the hole geometry to the mesh
    g = mesh.new_geometry(
        circle_vertices, circle_segments, [pv.CellType.LINE] * n_points
    )
    spar_cutouts.append(g)

# Optionally, calculate the mean element area for meshing
mean_element_area = mean_segment_length**2 / 2 * 1.5


# Prepare the dictionary for the meshing process
A = {
    "vertices": mesh.points[:, :2],
    "segments": airfoil.cells + [seg for g in spar_cutouts for seg in g.cells],
    "holes": hole_locs,
}


tr.plot(plt.axes(), **A)
plt.show()


B = tr.triangulate(A, f"qpa{mean_element_area}")
triangles = B["triangles"].tolist()

mesh.points = vertices_2d_to_3d(B["vertices"])
airfoil_face = mesh.new_geometry(
    cells=triangles, cell_types=[pv.CellType.TRIANGLE] * len(triangles)
)


cord = mesh.new_geometry(
    points=airfoil_points[[0, 51]],
    cells=[[0, 1]],
    cell_types=[pv.CellType.LINE],
)


wing_length = 1 * 12
element_length = 3
num_elements = wing_length // element_length
z_heights = np.linspace(0, 2, 3)
z_heights = np.linspace(0, wing_length, num_elements + 1)
mesh.points = layer_points(mesh.points, z_heights)

cords = layer_geometry(cord, z_heights)
spar_cutout = [cell[0] for cell in spar_cutouts[0].cells]
# print("spar_cutout", spar_cutout, sep="\n")
g = mesh.new_geometry(cells=[spar_cutout], cell_types=[pv.CellType.LINE])
spar_cutouts = layer_geometry(g, z_heights)

wing_surface = extrude_geometry(airfoil, z_heights)
wing_volume = extrude_geometry(airfoil_face, z_heights)

# mesh.get_unstructured_grid().plot()

wing_surface.cell_data["area_normals"] = compute_area_normals(wing_surface)
wing_surface.cell_data["area"] = np.linalg.norm(
    wing_surface.cell_data["area_normals"], axis=1
)
mid_points = wing_surface.get_cell_midpoints()


# Constants
rho_kg_m3 = 1.225  # Air density in kg/m³
rho_slug_ft3 = 0.002378

V_m_s = 12.8611  # Air velocity in m/s
V_ft_s = V_m_s * 3.28084  # Convert to ft/s

# Dynamic pressure in lb/ft²
q_lb_ft2 = 0.5 * rho_slug_ft3 * V_ft_s**2

# Convert dynamic pressure to PSI
q_psi = q_lb_ft2 / 144  # 1 PSI = 144 lb/ft²


wing_surface.cell_data["forces"] = (
    -q_psi
    * wing_surface.cell_data["area_normals"]
    * wing_surface.cell_data["cp"].reshape(-1, 1)
)


total_force = np.sum(wing_surface.cell_data["forces"], axis=0)
total_force = (rotation_matrix3d(-10) @ total_force.T).T
# Total force: [-13.40860057 263.18725021   0.        ]

print("Total force:", total_force)

# TODO how can drag be negative at 10 AoA

# Distribute forces to nodes
mesh.point_data["forces"] = np.zeros_like(mesh.points)
for cell, force in zip(wing_surface.cells, wing_surface.cell_data["forces"]):
    mesh.point_data["forces"][cell] += force / len(cell)


# Simplify the mesh
mesh, wing_volume = wing_volume.new_mesh_from_geometry()


# # Plot forces
# plotter = pv.Plotter()
# plotter.add_mesh(wing_volume.get_unstructured_grid(), show_edges=True)
# plotter.add_arrows(
#     wing_surface.get_cell_midpoints(), wing_surface.cell_data["forces"], mag=1
# )
# plotter.add_arrows(np.array([[0, 0, 0]]), total_force.reshape(-1, 3), mag=1)
# # plotter.add_arrows(mesh.points, mesh.point_data["forces"], mag=1)
# plotter.show_grid()
# plotter.show()

points = np.array(mesh.points, dtype=np.float32)
forces = np.array(mesh.point_data["forces"], dtype=np.float32)

constraints = np.zeros_like(points)
# constrain points on the z=0 plane
constraints[np.where(points[:, 2] == 0)] = 1

elements = wing_volume.cells
element_types = [core.ElementType.PRISM6] * len(elements)

foam_E = 1000  # Young's modulus in psi
foam_nu = 0.0  # Poisson's ratio
element_properties = [(foam_E, foam_nu)] * len(elements)


# ADD WING SPAR
spar_points = []
spar_loc = (circles_x[0], circles_y[0])
spar_radius = circles_r[0]
rigid_elements = []
spar_elements = []
spar_constraints = []
idx = len(points)
for i, geo in enumerate(spar_cutouts):
    z = np.unique(geo.get_cell_midpoints()[:, 2])
    assert len(z) == 1
    z = z[0]
    # Add the center point
    spar_points.append([circles_x[0], circles_y[0], z])
    # add the rotation point
    spar_points.append([0, 0, 0])

    # complete the element
    e = [idx, idx + 1]
    e.extend(geo.cells[0])
    # print("geo.cells", geo.cells, geo.cell_types, sep="\n")
    rigid_elements.append(e)

    if i != 0:
        spar_elements.append([idx - 2, idx - 1, idx - 0, idx + 1])
        spar_constraints.append([0, 0, 0])
        spar_constraints.append([0, 0, 0])
    else:
        spar_constraints.append([1, 1, 1])
        spar_constraints.append([1, 1, 1])

    idx += 2

print("rigid_elements", rigid_elements, sep="\n")
print("spar_elements", spar_elements, sep="\n")
print("spar_constraints", spar_constraints, sep="\n")

# exit()
points = np.vstack((points, np.array(spar_points, dtype=np.float32)))
forces = np.vstack((forces, np.zeros_like(spar_points, dtype=np.float32)))
constraints = np.vstack((constraints, np.array(spar_constraints)))

assert constraints.shape == points.shape
assert forces.shape == points.shape
elements.extend(rigid_elements)
element_types.extend([core.ElementType.RIGID] * len(rigid_elements))
element_properties.extend([tuple()] * len(rigid_elements))

elements.extend(spar_elements)
element_types.extend([core.ElementType.BEAM2] * len(spar_elements))

alum_E = 10_000_000
alum_nu = 0.3

from lib.geometry import second_moment_of_area_tube

Izz, Iyy, Ixx, A = second_moment_of_area_tube(2, 0.065)

element_properties.extend([(alum_E, alum_nu, Iyy, Izz, A, Ixx, 0)] * len(spar_elements))


constraints = np.zeros_like(points)

# constrain points on the z=0 plane
constraints[np.where(points[:, 2] == 0)] = 1


E = 1000  # Young's modulus in psi
nu = 0.3  # Poisson's ratio
725.189

core.DEBUG = True
feamesh = core.FEAModel(
    nodes=points,
    elements=elements,
    element_type=element_types,
    element_properties=element_properties,
    constraints_vector=constraints,
    forces=forces,
)


import inspect


def callback_maker(iters=10):
    i = 0
    last_solve = None

    def callback(x):
        nonlocal i
        nonlocal last_solve
        i += 1
        if i % iters == 0:
            frame = inspect.currentframe().f_back
            # print(list(frame.f_locals.keys()))
            res = np.linalg.norm(frame.f_locals["r"])
            print(f"Iteration {i} Residual {res:e}")
        last_solve = x

    return callback


feamesh.solve(use_iterative_solver=True, callback=callback_maker(100), rtol=1e-4)

plotter = pv.Plotter()
m = feamesh.generate_pv_unstructured_mesh()
plotter.add_mesh(m, scalars="von_mises_stress", cmap="viridis", show_edges=True)
m = feamesh.generate_pv_force_arrows()
# plotter.add_mesh(m)
plotter.show_grid()
plotter.show()
