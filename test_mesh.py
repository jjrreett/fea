# import numpy as np
# import pyvista as pv
# import triangle as tr
# import numpy.typing as npt
# from naca_airfoil import naca_points
# import matplotlib.pyplot as plt

# from lib.geometry import circle


# class MeshComposer2D:
#     def __init__(self):
#         self.vertices = np.empty((0, 2))
#         self.elements: list[list[np.intp]] = []
#         self.named_sets: dict[str, tuple[int, npt.NDArray]] = {}

#     def add_named_elements_set(
#         self,
#         name: str,
#         vertices: npt.NDArray,
#         elements: list[npt.NDArray],
#         pv_type: int,
#     ):
#         offset = len(self.vertices)
#         elements = [[idx + offset for idx in e] for e in elements]

#         # Add the indices of the new elements to the named set
#         self.named_sets[name] = (
#             pv_type,
#             np.arange(len(self.elements), len(self.elements) + len(elements)),
#         )
#         self.elements.extend(elements)

#         # # Combine vertices and remove duplicates
#         self.vertices = np.vstack((self.vertices, vertices))
#         # self.vertices, mapping_idx = np.unique(
#         #     self.vertices, axis=0, return_inverse=True
#         # )
#         # new_elements = []
#         # for element in self.elements:
#         #     new_elements.append([mapping_idx[idx] for idx in element])
#         # self.elements = new_elements

#     def layer(self, z_heights) -> "MeshComposer3D":
#         """
#         Layers 2d vertices into 3d vertices along the z-axis.
#         Leaves elements and named sets unchanged.
#         Those can be extruded or layered separately.

#         Parameters:
#         - z_heights: list of z heights to layer the vertices.

#         Returns:
#         - MeshComposer3D: the extruded 3D mesh.
#         """
#         n_nodes_in_layer = self.vertices.shape[0]
#         n_layers = len(z_heights)
#         nodes3d = np.zeros((n_nodes_in_layer * n_layers, 3))

#         # Stack nodes in the z direction
#         for i, z in enumerate(z_heights):
#             local_nodes3d = np.hstack(
#                 [self.vertices, np.full((n_nodes_in_layer, 1), z)]
#             )
#             nodes3d[i * n_nodes_in_layer : (i + 1) * n_nodes_in_layer] = local_nodes3d

#         return MeshComposer3D(nodes3d, self.elements, self.named_sets)


# class MeshComposer3D:
#     def __init__(self, vertices=None, elements=None, named_sets=None):
#         self.vertices = vertices if vertices is not None else np.empty((0, 3))
#         self.elements = elements if elements is not None else []
#         self.named_sets: dict[str, tuple[int, npt.NDArray]] = (
#             named_sets if named_sets else {}
#         )
#         self.z_heights = np.unique(self.vertices[:, 2])
#         self.vertices_per_layer = len(self.vertices) // max(len(self.z_heights), 1)

#     def add_named_elements_set(self, *args, **kwargs):
#         MeshComposer2D.add_named_elements_set(self, *args, **kwargs)

#     def layer_set(self, name: str):
#         """
#         Creates new named sets by layering the vertices of an existing named set.

#         Returns:
#         - MeshComposer3D: the extruded 3D mesh.
#         """
#         pv_type, indices = self.named_sets[name]
#         elements = []
#         for idx in indices:
#             element = self.elements[idx]
#             elements.append(element)

#         for i, z in enumerate(self.z_heights):
#             new_name = f"{name}_{i}"
#             vertices = self.vertices[
#                 i * self.vertices_per_layer : (i + 1) * self.vertices_per_layer
#             ]
#             self.add_named_elements_set(new_name, vertices, elements, pv_type)

#     def extrude_set(self, name: str):
#         """
#         Extrudes a named set along the z-axis.
#         """
#         pv_type, indices = self.named_sets[name]
#         # Map from the nd to the (n+1)d element type
#         element_type_map = {
#             pv.CellType.VERTEX: pv.CellType.LINE,
#             pv.CellType.LINE: pv.CellType.PIXEL,  # TODO use reorder elements to use quad
#             pv.CellType.TRIANGLE: pv.CellType.WEDGE,
#             pv.CellType.PIXEL: pv.CellType.VOXEL,
#             pv.CellType.QUAD: pv.CellType.HEXAHEDRON,
#         }
#         new_pv_type = element_type_map[pv_type]

#         elements = []
#         for idx in indices:
#             element = self.elements[idx]
#             elements.append(element)

#         new_elements = []
#         for element in elements:
#             # Repeat nodes in layers
#             # [1, 2] -> [1, 2, self.vertices_per_layer + 1, self.vertices_per_layer + 2]
#             for layer in range(len(self.z_heights) - 1):
#                 new_element = [
#                     idx + (layer) * self.vertices_per_layer for idx in element
#                 ]
#                 new_element.extend(
#                     [idx + (layer + 1) * self.vertices_per_layer for idx in element]
#                 )
#                 new_elements.append(new_element)
#             # new_element = element + [idx + self.vertices_per_layer for idx in element]
#             # new_elements.append(new_element)

#         new_name = f"{name}_3d"
#         self.named_sets[new_name] = (
#             new_pv_type,
#             np.arange(len(self.elements), len(self.elements) + len(new_elements)),
#         )
#         self.elements.extend(new_elements)

#     def normalize(self):
#         self.vertices, mapping_idx = np.unique(
#             self.vertices, axis=0, return_inverse=True
#         )
#         new_elements = []
#         for element in self.elements:
#             new_elements.append([mapping_idx[idx] for idx in element])
#         self.elements = new_elements

#     def get_combined_mesh(self):
#         cells = []
#         cell_types = []
#         for name, (pv_type, indices) in self.named_sets.items():
#             for idx in indices:
#                 element = self.elements[idx]
#                 cells.append(len(element))
#                 cells.extend(element)
#                 cell_types.append(pv_type)

#         grid = pv.UnstructuredGrid(cells, cell_types, self.vertices)
#         return grid

#     def get_single_mesh(self, name):
#         cells = []
#         cell_types = []
#         pv_type, indices = self.named_sets[name]
#         for idx in indices:
#             element = self.elements[idx]
#             cells.append(len(element))
#             cells.extend(element)
#             cell_types.append(pv_type)
#         grid = pv.UnstructuredGrid(cells, cell_types, self.vertices)
#         return grid

#     def plot(self):
#         grid = self.get_combined_mesh()
#         plotter = pv.Plotter()
#         # plotter.add_mesh(pv.PointSet(self.vertices))
#         plotter.add_mesh(grid, show_edges=True)
#         plotter.show()


# # Create the MeshComposer instance
# composer = MeshComposer2D()

# # # Add airfoil part
# cord = 5.6 * 12
# airfoil_vertices, t = naca_points(101, 0.4, 0.04, 0.12, return_t_points=True)
# airfoil_vertices *= cord
# airfoil_segments = [(i, i + 1) for i in range(len(airfoil_vertices) - 1)]
# airfoil_segments.append((len(airfoil_vertices) - 1, 0))  # Close the loop
# # airfoil_segments = np.array(airfoil_segments)
# composer.add_named_elements_set(
#     "airfoil", airfoil_vertices, airfoil_segments, pv.CellType.LINE
# )

# airfoil = pv.PolyData()
# airfoil.points = np.hstack((airfoil_vertices, np.zeros((len(airfoil_vertices), 1))))
# airfoil.lines = [[len(s)] + list(s) for s in airfoil_segments]


# # Add circle part
# r = 2
# circumference = 2 * np.pi * r
# mean_segment_length = np.mean(
#     np.linalg.norm(
#         np.diff(airfoil_vertices[airfoil_segments], axis=1).squeeze(), axis=1
#     )
# )
# n_points = int(np.ceil(circumference / mean_segment_length))
# circle_vertices, circle_segments = circle(n_points, 2, x=10, y=1.26)
# composer.add_named_elements_set(
#     "circle", circle_vertices, circle_segments.tolist(), pv.CellType.LINE
# )

# hole = pv.PolyData()
# hole.points = np.hstack((circle_vertices, np.zeros((len(circle_vertices), 1))))
# hole.lines = [[len(s)] + list(s) for s in circle_segments]

# mean_element_area = mean_segment_length**2 / 2 * 1.5


# A = {
#     "vertices": composer.vertices[:, :2],
#     "segments": composer.elements,
#     "holes": np.array([[10, 1.26]]),
# }
# B = tr.triangulate(A, f"qpa{mean_element_area}")
# triangles = B["triangles"].tolist()

# composer.add_named_elements_set(
#     "triangles", B["vertices"], triangles, pv.CellType.TRIANGLE
# )

# composer.add_named_elements_set(
#     "spar", np.array([[10, 1.26]]), [[0]], pv.CellType.VERTEX
# )

# composer.add_named_elements_set(
#     "cord", airfoil_vertices[[0, 51]], [[0, 1]], pv.CellType.LINE
# )

# 16 * 12
# composer = composer.layer(np.linspace(0, 16 * 12, (16 * 12) // 2 + 1))
# # composer.extrude_set("airfoil")
# # composer.extrude_set("circle")
# composer.extrude_set("triangles")
# composer.extrude_set("spar")
# composer.layer_set("cord")
# composer.normalize()
# composer.plot()

# grid = composer.get_single_mesh("airfoil_1")
# plotter = pv.Plotter()
# plotter.add_mesh(grid, show_edges=True)
# plotter.show()
