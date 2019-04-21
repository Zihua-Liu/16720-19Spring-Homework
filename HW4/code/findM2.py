'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import matplotlib.pyplot as plt
from submission import *
import helper

if __name__ == "__main__":
    pts = np.load('../data/some_corresp.npz')
    pts1 = pts["pts1"]
    pts2 = pts["pts2"]
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    M = np.max(im1.shape)
    F = eightpoint(pts1, pts2, M)

    intrinsic = np.load('../data/intrinsics.npz')
    K1, K2 = intrinsic['K1'], intrinsic['K2']
    E = essentialMatrix(F, K1, K2)
    M2s = helper.camera2(E)

    M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    C1 = np.dot(K1, M1)

    for i in range(M2s.shape[-1]):
    	M2 = M2s[:, :, i]
    	C2 = np.dot(K2, M2)
    	w, err = triangulate(C1, pts1, C2, pts2)
    	if np.min(w[:, -1]) > 0:
    		break
    C2 = np.dot(K2, M2)
    np.savez('q3_3.npz', M2 = M2, C2 = C2, P = w)