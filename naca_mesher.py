import matplotlib.pyplot as plt
import numpy as np

import triangle as tr
from pathlib import Path
from pprint import pformat
from units import *
from utils import *
import pickle


# def interp1d(x, y):
#     return lambda new_x: np.interp(new_x, x, y)


# def multiInterp(x, xp, fp):
#     i = np.arange(x.size)
#     j = np.searchsorted(xp, x) - 1
#     d = (x - xp[j]) / (xp[j + 1] - xp[j])
#     print("i", i, "j", j, "d", d)
#     return (1 - d) * fp[i, j] + fp[i, j + 1] * d


# def naca_airfoil(
#     camber=0.04, camber_pos=0.4, thickness=0.12, cord_length=1, n_points=100
# ):
#     def T(x, thickness):
#         return (
#             5
#             * thickness
#             * (
#                 0.2969 * np.sqrt(x)
#                 - 0.1260 * x
#                 - 0.3516 * x**2
#                 + 0.2843 * x**3
#                 - 0.1036 * x**4
#             )
#         )

#     def cord(x, m, p):
#         return np.where(
#             x < p,
#             m / p**2 * (2 * p * x - x**2),
#             m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2),
#         )

#     def dcord(x, m, p):
#         return np.where(
#             x < p, m / p**2 * (2 * p - 2 * x), m / (1 - p) ** 2 * (2 * p - 2 * x)
#         )

#     def alpha(x, m, p):
#         return np.arctan(dcord(x, m, p))

#     def upper(x, m, p, t):
#         xs = x - T(x, t) * np.sin(alpha(x, m, p))
#         ys = cord(x, m, p) + T(x, t) * np.cos(alpha(x, m, p))
#         o = np.array([xs, ys]).T
#         return o

#     def lower(x, m, p, t):
#         xs = x + T(x, t) * np.sin(alpha(x, m, p))
#         ys = cord(x, m, p) - T(x, t) * np.cos(alpha(x, m, p))
#         o = np.array([xs, ys]).T
#         return o

#     x = np.linspace(0, 1, n_points)
#     x = 2 * x
#     ux = np.linspace(1, 0, n_points)
#     lx = np.linspace(0, 1, n_points)[1:-2]
#     # x = np.linspace(0, 2, n_points)
#     # combined_x, combined_y = full(x, camber, camber_pos, thickness)
#     vertices = np.vstack(
#         [
#             upper(ux, camber, camber_pos, thickness),
#             lower(lx, camber, camber_pos, thickness),
#         ]
#     )
#     print("vertices", vertices, sep="\n")

#     distance_between_vertices = np.sqrt(np.sum(np.diff(vertices, axis=0) ** 2, axis=1))
#     # Calculate cumulative distance along the combined surface
#     dist = np.cumsum(distance_between_vertices)
#     dist = np.insert(dist, 0, 0)

#     # Interpolate to get evenly spaced points by distance
#     even_dist = np.linspace(0, dist[-1], n_points)
#     vertices = multiInterp(even_dist, dist, vertices)

#     return vertices


from naca_airfoil import naca_points

vertices = naca_points(100, 0.4, 0.04, 0.12) * 5.6 * ft

# for i, pos in enumerate(vertices):
#     plt.text(*pos, str(i))
# plt.scatter(vertices[:, 0], vertices[:, 1])
# plt.axis("equal")
# plt.show()


segments = [(i, i + 1) for i in range(len(vertices) - 1)]
segments.append((len(vertices) - 1, 0))  # Close the loop
segments = np.array(segments)


def circle(N, R, x=0, y=0):
    i = np.arange(N)
    theta = i * 2 * np.pi / N
    pts = np.stack([np.cos(theta), np.sin(theta)], axis=1) * R
    pts[:, 0] += x  # Adjust x coordinates
    pts[:, 1] += y  # Adjust y coordinates
    seg = np.stack([i, (i + 1) % N], axis=1)
    return pts, seg


pts0, seg0 = circle(15, 2 * inch, x=10 * inch, y=1.26 * inch)

vertices = np.vstack([vertices, pts0])
segments = np.vstack([segments, seg0 + segments.shape[0]])


A = {
    "vertices": vertices,
    "segments": segments,
    "holes": np.array([[10 * inch, 1.26 * inch]]),
}
B = tr.triangulate(A, "qpa0.001")  # 'p' ensures that the segments are respected


with open("airfoil.pickle", "wb") as f:
    pickle.dump(B, f)

tr.plot(plt.gca(), **B)
plt.show()
