import numpy as np
import cv2
from BRIEF import briefLite, briefMatch


def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1] == p2.shape[1])
    assert(p1.shape[0] == 2)
    num_points = p1.shape[1]
    A = np.zeros((2 * num_points, 9))
    A[np.array(
        [row for row in range(2 * num_points) if row % 2 == 0]
    ), 0:3] = np.hstack((p2.T, np.ones((num_points, 1))))
    A[np.array(
        [row for row in range(2 * num_points) if row % 2 == 1]
    ), 3:6] = np.hstack((p2.T, np.ones((num_points, 1))))
    A[np.array(
        [row for row in range(2 * num_points) if row % 2 == 0]
    ), 6:9] = -np.hstack((p2.T, np.ones((num_points, 1)))) * p1.T[:, 0].reshape(-1, 1)
    A[np.array(
        [row for row in range(2 * num_points) if row % 2 == 1]
    ), 6:9] = -np.hstack((p2.T, np.ones((num_points, 1)))) * p1.T[:, 1].reshape(-1, 1)
    V = np.dot(A.T, A)
    eigen_values, eigen_vecs = np.linalg.eigh(V)
    squeezed_H = eigen_vecs[:, 0]
    H2to1 = squeezed_H.reshape(3, 3)
    return H2to1


def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    '''
    points1, points2 = [], []
    for item in matches:
        point1 = locs1[item[0]][:-1]
        point2 = locs2[item[1]][:-1]
        points1.append(point1)
        points2.append(point2)
    points1, points2 = np.array(points1).T, np.array(points2).T
    num_points = points1.shape[1]
    max_inlier_points = -1
    bestH = None
    for i in range(num_iter):
        try:
            random_indexes = np.sort(np.random.randint(
                low=0, high=num_points + 1, size=4))
            p1 = points1[:, random_indexes]
            p2 = points2[:, random_indexes]
            H2to1 = computeH(p1, p2)
            # print(points1)
            homo_points1 = np.vstack((points1, np.ones((1, num_points))))
            homo_points2 = np.vstack((points2, np.ones((1, num_points))))
            projected_points1 = np.dot(H2to1, homo_points2)
            projected_points1 /= projected_points1[2, :]
            distance = np.sqrt(
                np.sum(np.square(homo_points1 - projected_points1), axis=0))
            num_inlier_points = distance[distance < tol].shape[0]
            if num_inlier_points > max_inlier_points:
                max_inlier_points = num_inlier_points
                bestH = H2to1
        except:
            continue
    return bestH


if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png.jpg')
    im2 = cv2.imread('../data/incline_R.png.jpg')
    # im2 = cv2.imread('../data/model_chickenbroth.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
