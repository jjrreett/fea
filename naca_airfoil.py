import numpy as np


def multiInterp(x, xp, fp):
    return np.array([np.interp(x, xp, fp[:, i]) for i in range(fp.shape[1])]).T


def naca_funcs(p, m, k):
    def cord(x):
        x = np.asarray(x)
        result = np.zeros_like(x)
        mask1 = (x >= 0) & (x <= p)
        mask2 = (x > p) & (x <= 1)
        result[mask1] = (m / p**2) * (2 * p * x[mask1] - x[mask1] ** 2)
        result[mask2] = (m / (1 - p) ** 2) * (
            (1 - 2 * p) + 2 * p * x[mask2] - x[mask2] ** 2
        )
        return result

    def thickness(x):
        x = np.asarray(x)
        return (
            5
            * k
            * (
                0.2969 * np.sqrt(x)
                - 0.1260 * x
                - 0.3516 * x**2
                + 0.2843 * x**3
                - 0.1036 * x**4
            )
        )

    def dcord_dx(x):
        x = np.asarray(x)
        result = np.zeros_like(x)
        mask1 = (x >= 0) & (x <= p)
        mask2 = (x > p) & (x <= 1)
        result[mask1] = (m / p**2) * (2 * p - 2 * x[mask1])
        result[mask2] = (m / (1 - p) ** 2) * (2 * p - 2 * x[mask2])
        return result

    return cord, thickness, dcord_dx


def naca(t, p, m, k):
    cord, thickness, dcord_dx = naca_funcs(p, m, k)

    t[t >= 1] = (t - 1)[t >= 1]
    t[t < 0] = (t + 1)[t < 0]
    upper_mask = (2 * t >= 0) & (2 * t <= 1)
    lower_mask = (2 * t > 1) & (2 * t < 2)
    tup = t[upper_mask]
    tlow = t[lower_mask]
    xup = 1 - 2 * tup
    xlow = 2 * tlow - 1
    out = np.zeros((len(t), 2))

    # Precompute C, T, and C_d
    Cup = cord(xup)
    Clow = cord(xlow)
    Tup = thickness(xup)
    Tlow = thickness(xlow)
    Cdup = dcord_dx(xup)
    Cdlow = dcord_dx(xlow)

    out[upper_mask, 0] = xup - Tup * Cdup / np.sqrt(1 + Cdup**2)
    out[upper_mask, 1] = Cup + Tup / np.sqrt(1 + Cdup**2)
    out[lower_mask, 0] = xlow + Tlow * Cdlow / np.sqrt(1 + Cdlow**2)
    out[lower_mask, 1] = Clow - Tlow / np.sqrt(1 + Cdlow**2)

    return out


def naca_points(n_points, p, m, k, debug=False, return_t_points=False):
    t = np.linspace(0, 1, n_points, endpoint=False)
    for i in range(100):
        if i == 0:
            t = np.linspace(0, 1, n_points, endpoint=False)
        else:
            distance_between_vertices = np.sqrt(
                np.sum(np.diff(vertices, axis=0) ** 2, axis=1)
            )
            # Calculate cumulative distance along the combined surface
            dist = np.cumsum(distance_between_vertices)
            dist = np.insert(dist, 0, 0)

            # Interpolate to get evenly spaced points by distance
            even_dist = np.linspace(0, dist[-1], n_points)
            nt = np.interp(even_dist, dist, t)
            convergence = np.linalg.norm(nt - t)
            if debug:
                print("convergence =", convergence)
            if convergence < 1e-6:
                break
            t = nt
        vertices = naca(t, p, m, k)

    if return_t_points:
        return vertices, t
    else:
        return vertices


def naca_cutouts(p, m, k, num_cutouts, web):
    C, T, C_d = naca_funcs(p, m, k)
    # Transforming circles_x to match thickness spacing
    circles_x = [0.1]  # Start point for circles
    for _ in range(1, num_cutouts):
        prev_x = circles_x[-1]
        local_thickness = T(prev_x)
        next_thickness = local_thickness

        for _ in range(10):
            next_x = prev_x + local_thickness + next_thickness - web
            next_thickness = T(next_x)

        circles_x.append(next_x)

    circles_x = np.array(circles_x)
    circles_y = C(circles_x)
    circles_r = T(circles_x) - web

    return circles_x, circles_y, circles_r


if __name__ == "__main__":
    p = 0.4
    m = 0.04
    k = 0.12
    vertices = naca_points(100, p, m, k)
    from matplotlib import pyplot as plt

    plt.scatter(vertices[:, 0], vertices[:, 1])
    plt.axis("equal")
    plt.show()
