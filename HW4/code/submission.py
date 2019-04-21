"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper
import cv2
import matplotlib.pyplot as plt

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    x1, y1, x2, y2 = pts1[:, 0] / M, pts1[:, 1] / M, pts2[:, 0] / M, pts2[:, 1] / M
    T = np.array([[1 / M, 0, 0], [0, 1 / M, 0], [0, 0, 1]])
    A = np.vstack((x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones(x1.shape))).T
    _, _, vh = np.linalg.svd(A)
    F = vh[-1, :].reshape(3, 3)
    F = helper.refineF(F, pts1 / M, pts2 / M)
    F = helper._singularize(F)
    F = np.dot(np.dot(T.T, F), T)
    return F


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def solution(alpha, F1, F2):
    return np.linalg.det(alpha * F1 + (1 - alpha) * F2)

def sevenpoint(pts1, pts2, M):
    x1, y1, x2, y2 = pts1[:, 0] / M, pts1[:, 1] / M, pts2[:, 0] / M, pts2[:, 1] / M
    T = np.array([[1 / M, 0, 0], [0, 1 / M, 0], [0, 0, 1]])
    A = np.vstack((x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones(x1.shape))).T
    _, _, vh = np.linalg.svd(A)
    F1 = vh[-1, :].reshape(3, 3)
    F2 = vh[-2, :].reshape(3, 3)
    a0 = solution(0.0, F1, F2)
    a1 = 2.0 * (solution(1.0, F1, F2) - solution(-1.0, F1, F2)) / 3.0 - (solution(2.0, F1, F2) - solution(-2.0, F1, F2)) / 12.0
    a2 = (solution(1.0, F1, F2) + solution(-1.0, F1, F2)) / 2.0 - a0
    a3 = (solution(1.0, F1, F2) - solution(-1.0, F1, F2)) / 2.0 - a1
    alphas = np.roots(np.array([a3, a2, a1, a0]))
    Farray = [alpha * F1 + (1 - alpha) * F2 for alpha in alphas]
    Farray = [helper.refineF(F, pts1 / M, pts2 / M) for F in Farray]
    Farray = [np.dot(np.dot(T.T, F), T) for F in Farray]
    return Farray


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    return np.dot(np.dot(K2.T, F), K1)


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    x1, y1, x2, y2 = pts1[:, 0], pts1[:, 1], pts2[:, 0], pts2[:, 1]
    A1 = np.vstack((C1[0, 0] - C1[2, 0] * x1, C1[0, 1] - C1[2, 1] * x1, C1[0, 2] - C1[2, 2] * x1, C1[0, 3] - C1[2, 3] * x1)).T
    A2 = np.vstack((C1[1, 0] - C1[2, 0] * y1, C1[1, 1] - C1[2, 1] * y1, C1[1, 2] - C1[2, 2] * y1, C1[1, 3] - C1[2, 3] * y1)).T
    A3 = np.vstack((C2[0, 0] - C2[2, 0] * x2, C2[0, 1] - C2[2, 1] * x2, C2[0, 2] - C2[2, 2] * x2, C2[0, 3] - C2[2, 3] * x2)).T
    A4 = np.vstack((C2[1, 0] - C2[2, 0] * y2, C2[1, 1] - C2[2, 1] * y2, C2[1, 2] - C2[2, 2] * y2, C2[1, 3] - C2[2, 3] * y2)).T
    w = np.zeros((pts1.shape[0], 3))
    for i in range(pts1.shape[0]):
        A = np.vstack((A1[i, :], A2[i, :], A3[i, :], A4[i, :]))
        _, _, vh = np.linalg.svd(A)
        w[i, :] = vh[-1, :3] / vh[-1, -1]
    W = np.hstack((w, np.ones((pts1.shape[0], 1))))
    error = 0
    for i in range(pts1.shape[0]):
        projection1 = np.dot(C1, W[i, :].T)
        projection2 = np.dot(C2, W[i, :].T)
        projection1 = (projection1[:2] / projection1[-1]).T
        projection2 = (projection2[:2] / projection2[-1]).T
        error += np.sum(np.square(projection1 - pts1[i]) + np.square(projection2 - pts2[i]))
    return w, error


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    window_size = 20
    sigma = 5
    template = im1[y1 - window_size // 2 : y1 + window_size // 2 + 1, x1 - window_size // 2 : x1 + window_size // 2 + 1]
    H, W = im1.shape[0], im1.shape[1]

    p = np.array([x1, y1, 1])
    line = np.dot(F, p)
    line /= np.linalg.norm(line)

    Y = np.arange(H)
    X = np.around(-(line[1] * Y + line[2]) / line[0]).astype(np.int)
    valid = (X - window_size // 2 >= 0) & (X + window_size // 2 < W) & (Y - window_size // 2 >= 0) & (Y + window_size // 2 < H)
    X, Y = X[valid], Y[valid]
    window = np.arange(-window_size // 2, window_size // 2 + 1, 1)
    x, y = np.meshgrid(window, window)
    weight = np.exp(-(x ** 2 + y ** 2) / (2.0 * (sigma ** 2)))
    weight /= np.sum(weight)
    weight = np.repeat(weight[:, :, None], im1.shape[-1], axis = 2)

    min_error = float("inf")
    x2, y2 = None, None
    for i in range(len(X)):
        patch = im2[Y[i] -window_size // 2 : Y[i] + window_size // 2 + 1, X[i] - window_size // 2 : X[i] + window_size // 2 + 1]
        error = np.sum(np.abs(template - patch) * weight)
        if error < min_error:
            min_error = error
            x2, y2 = X[i], Y[i]
    return x2, y2




# if __name__ == "__main__":
    # Q2.1
    # pts = np.load('../data/some_corresp.npz')
    # pts1 = pts["pts1"]
    # pts2 = pts["pts2"]
    # im1 = plt.imread('../data/im1.png')
    # im2 = plt.imread('../data/im2.png')
    # M = np.max(im1.shape)
    # F = eightpoint(pts1, pts2, M)
    # print(F)
    # helper.displayEpipolarF(im1, im2, F)
    # np.savez('q2_1.npz', F = F, M = M)

    # Q2.2
    # pts = np.load('../data/some_corresp.npz')
    # index = [0, 1, 4, 8, 12, 19, 22]
    # pts1 = pts["pts1"][index, :]
    # pts2 = pts["pts2"][index, :]
    # im1 = plt.imread('../data/im1.png')
    # im2 = plt.imread('../data/im2.png')
    # M = np.max(im1.shape)
    # Farray = sevenpoint(pts1, pts2, M)
    # helper.displayEpipolarF(im1, im2, Farray[0])
    # print(Farray[0])
    # np.savez('q2_2.npz', F = Farray[0], M = M)

    # Q3.1
    # intrinsic = np.load('../data/intrinsics.npz')
    # K1, K2 = intrinsic['K1'], intrinsic['K2']
    # E = essentialMatrix(F, K1, K2)
    # print(E)

    #Q4.1
    # helper.epipolarMatchGUI(im1, im2, F)
    # np.savez('q4_1.npz', F = F, pts1 = pts1, pts2 = pts2)
    
