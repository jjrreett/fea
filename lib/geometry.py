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
    """
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False) + theta_offset
    pts = np.stack([np.cos(theta), np.sin(theta)], axis=1) * R
    pts[:, 0] += x  # Adjust x coordinates
    pts[:, 1] += y  # Adjust y coordinates
    seg = np.stack([np.arange(N), (np.arange(N) + 1) % N], axis=1)
    return pts, seg
