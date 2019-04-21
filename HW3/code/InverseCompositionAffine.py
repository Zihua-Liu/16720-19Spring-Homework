import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1):
    # Input: 
    #   It: template image
    #   It1: Current image

    # Output:
    #   M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = M.flatten()

    x1, y1, x2, y2 = 0, 0, It.shape[1] - 1, It.shape[0] - 1
    threshold = 1e-2
    dp = np.array([float("inf")] * 6)

    interpolated_It = RectBivariateSpline(
            x = np.array([i for i in range(int(It.shape[0]))]), 
            y = np.array([i for i in range(int(It.shape[1]))]), 
            z = It
        )
    interpolated_It1 = RectBivariateSpline(
            x = np.array([i for i in range(int(It1.shape[0]))]), 
            y = np.array([i for i in range(int(It1.shape[1]))]), 
            z = It1
        )
    x = np.arange(x1, x2 + .5)
    y = np.arange(y1, y2 + .5)
    X = np.array([x for i in range(len(y))])
    Y = np.array([y for i in range(len(x))]).T

    dx = interpolated_It.ev(Y, X, dx = 0, dy = 1).flatten()
    dy = interpolated_It.ev(Y, X, dx = 1, dy = 0).flatten()

    A = np.array([
            dx * X.flatten(),
            dx * Y.flatten(),
            dx,
            dy * X.flatten(),
            dy * Y.flatten(),
            dy,
        ]).T
    while np.sum(np.square(dp)) >= threshold:
        warp_X = p[0] * X + p[1] * Y + p[2]
        warp_Y = p[3] * X + p[4] * Y + p[5]
        valid_points = (warp_X >= x1) & (warp_X <= x2) & (warp_Y >= y1) & (warp_Y <= y2)
        warp_X, warp_Y = warp_X[valid_points], warp_Y[valid_points]
        warp_It1x = interpolated_It1.ev(warp_Y, warp_X)

        A_valid = A[valid_points.flatten()]
        b = (warp_It1x - It[valid_points]).flatten()

        dp = np.dot(np.linalg.inv(np.dot(A_valid.T, A_valid)), np.dot(A_valid.T, b))

        M = np.copy(p).reshape(2, 3)
        M = np.vstack((M, np.array([[0, 0, 1]])))
        dM = np.vstack((np.copy(dp).reshape(2, 3), np.array([[0, 0, 1]])))
        dM[0, 0] += 1
        dM[1, 1] += 1
        M = np.dot(M, np.linalg.inv(dM))
        p = M[:2, :].flatten()


    return M
