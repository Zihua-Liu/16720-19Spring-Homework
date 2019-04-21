import cv2
from BRIEF import briefLite, briefMatch, plotMatches
import numpy as np
import matplotlib.pyplot as plt 


def rotate_image(original_image, angle):
    H, W, _ = original_image.shape
    image_center = (W / 2, H / 2)
    rotate_matrix = cv2.getRotationMatrix2D(image_center, angle, 1)
    abs_cos = abs(rotate_matrix[0, 0])
    abs_sin = abs(rotate_matrix[0, 1])
    bound_w = int(H * abs_sin + W * abs_cos)
    bound_h = int(H * abs_cos + W * abs_sin)
    rotate_matrix[0, 2] += bound_w / 2 - image_center[0]
    rotate_matrix[1, 2] += bound_h / 2 - image_center[1]
    rotate_image = cv2.warpAffine(
        original_image, rotate_matrix, (bound_w, bound_h))
    return rotate_image, rotate_matrix


def perform_match(original_image, rotation_image, display=False):
    locs1, desc1 = briefLite(original_image)
    locs2, desc2 = briefLite(rotation_image)
    matches = briefMatch(desc1, desc2)
    if display:
        plotMatches(original_image, rotation_image, matches, locs1, locs2)
    original_points = []
    rotation_points = []
    for item in matches:
        point1 = locs1[item[0]][:-1]
        point2 = locs2[item[1]][:-1]
        original_points.append(point1)
        rotation_points.append(point2)
    return np.array(original_points), np.array(rotation_points)


def count_match_points(original_points, rotation_points, rotate_matrix):
    ans = 0
    ones = np.ones(shape=(len(original_points), 1))
    points_ones = np.hstack([original_points, ones])
    transformed_points = rotate_matrix.dot(points_ones.T).T.astype(int)
    for point1, point2 in zip(transformed_points, rotation_points):
        if abs(point1[0] - point2[0]) <= 1 and abs(point1[1] - point2[1]) <= 1:
            ans += 1
    return ans


if __name__ == "__main__":
    original_image = cv2.imread("../data/model_chickenbroth.jpg")
    x, y = [], []
    for angle in range(0, 370, 10):
        rotation_image, rotate_matrix = rotate_image(
            original_image, angle=angle)
        original_points, rotation_points = perform_match(
                original_image, rotation_image, display=False)
        num_match_points = count_match_points(
            original_points, rotation_points, rotate_matrix)
        x.append(str(angle))
        y.append(num_match_points)
        print("Rotation Angle: {}, Number of Matched Points: {}".format(
            angle, num_match_points))
    plt.bar(x, y)
    plt.show()
