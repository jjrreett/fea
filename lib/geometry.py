import numpy as np


def circle(N, R, x=0, y=0, theta_offset=0):
    """
    Creates a 2d circle with N points and radius R centered at (x, y).
    Parameters
    ----------
    N : int
        Number of points
    R : float
        Radius
    x : float
        x coordinate of the center
    y : float
        y coordinate of the center
    theta_offset : float
        Offset angle in radians

    Returns
    -------
    pts : ndarray
        2d coordinates of the points
    seg : ndarray
        2d segments of the circle
    """
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False) + theta_offset
    pts = np.stack([np.cos(theta), np.sin(theta)], axis=1) * R
    pts[:, 0] += x  # Adjust x coordinates
    pts[:, 1] += y  # Adjust y coordinates
    seg = np.stack([np.arange(N), (np.arange(N) + 1) % N], axis=1)
    return pts, seg


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
