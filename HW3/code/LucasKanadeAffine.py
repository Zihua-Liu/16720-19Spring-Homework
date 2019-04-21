import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1):
    # Input: 
    #   It: template image
    #   It1: Current image
    # Output:
    #   M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = M.flatten()

    x1, y1, x2, y2 = 0, 0, It.shape[1] - 1, It.shape[0] - 1
    threshold = 1e-3
    dp = np.array([float("inf")] * 6)

    interpolated_It1 = RectBivariateSpline(
            x = np.array([i for i in range(int(It1.shape[0]))]), 
            y = np.array([i for i in range(int(It1.shape[1]))]), 
            z = It1
        )

    while np.sum(np.square(dp)) >= threshold:
        x = np.arange(x1, x2 + .5)
        y = np.arange(y1, y2 + .5)
        X = np.array([x for i in range(len(y))])
        Y = np.array([y for i in range(len(x))]).T
        warp_X = p[0] * X + p[1] * Y + p[2]
        warp_Y = p[3] * X + p[4] * Y + p[5]
        valid_points = (warp_X >= x1) & (warp_X <= x2) & (warp_Y >= y1) & (warp_Y <= y2)
        X, Y = X[valid_points], Y[valid_points]
        warp_X, warp_Y = warp_X[valid_points], warp_Y[valid_points]
        warp_It1x = interpolated_It1.ev(warp_Y, warp_X)

        dx = interpolated_It1.ev(warp_Y, warp_X, dx = 0, dy = 1).flatten()
        dy = interpolated_It1.ev(warp_Y, warp_X, dx = 1, dy = 0).flatten()

        A = np.array([
                dx * X.flatten(),
                dx * Y.flatten(),
                dx,
                dy * X.flatten(),
                dy * Y.flatten(),
                dy,
            ]).T
        b = (It[valid_points] - warp_It1x).flatten()

        dp = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
        p += dp.flatten()

    M = np.copy(p).reshape(2, 3)
    M = np.vstack((M, np.array([[0, 0, 1]])))
    return M
