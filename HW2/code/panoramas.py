import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite, briefMatch, plotMatches


def blend_mask(im):
    mask = np.zeros((im.shape[0], im.shape[1]))
    mask[0, :] = 1
    mask[-1, :] = 1
    mask[:, 0] = 1
    mask[:, -1] = 1
    mask = distance_transform_edt(1 - mask)
    mask = mask / np.max(mask)
    return mask


def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    H1, W1, _ = im1.shape
    H2, W2, _ = im2.shape
    lefttop = np.array([0, 0, 1]).reshape(3, 1)
    righttop = np.array([W2 - 1, 0, 1]).reshape(3, 1)
    leftbottom = np.array([0, H2 - 1, 1]).reshape(3, 1)
    rightbottom = np.array([W2 - 1, H2 - 1, 1]).reshape(3, 1)
    projected_lefttop = np.dot(H2to1, lefttop)
    projected_lefttop /= projected_lefttop[2]
    projected_righttop = np.dot(H2to1, righttop)
    projected_righttop /= projected_righttop[2]
    projected_leftbottom = np.dot(H2to1, leftbottom)
    projected_leftbottom /= projected_leftbottom[2]
    projected_rightbottom = np.dot(H2to1, rightbottom)
    projected_rightbottom /= projected_rightbottom[2]
    new_W = int(max(projected_righttop[0], projected_rightbottom[0]))
    new_H = int(max(projected_leftbottom[1], projected_rightbottom[1]))
    warp_im2 = cv2.warpPerspective(im2, H2to1, (new_W, new_H))
    cv2.imwrite('../results/q6_1.jpg', warp_im2)
    mask1 = cv2.warpPerspective(blend_mask(im1), np.float32(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), (new_W, new_H))
    im1 = cv2.warpPerspective(im1, np.float32(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), (new_W, new_H))
    mask2 = cv2.warpPerspective(blend_mask(
        im2), H2to1, (new_W, new_H))
    sum_mask = mask1 + mask2
    mask1 /= sum_mask
    mask1 = np.repeat(mask1, 3).reshape(im1.shape)
    mask2 /= sum_mask
    mask2 = np.repeat(mask2, 3).reshape(warp_im2.shape)
    pano_im = im1 * mask1 + warp_im2 * mask2
    np.save("../results/q6_1.npy", H2to1)
    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    '''
    H1, W1, _ = im1.shape
    H2, W2, _ = im2.shape
    lefttop = np.array([0, 0, 1]).reshape(3, 1)
    righttop = np.array([W2 - 1, 0, 1]).reshape(3, 1)
    leftbottom = np.array([0, H2 - 1, 1]).reshape(3, 1)
    rightbottom = np.array([W2 - 1, H2 - 1, 1]).reshape(3, 1)
    projected_lefttop = np.dot(H2to1, lefttop)
    projected_lefttop /= projected_lefttop[2]
    projected_righttop = np.dot(H2to1, righttop)
    projected_righttop /= projected_righttop[2]
    projected_leftbottom = np.dot(H2to1, leftbottom)
    projected_leftbottom /= projected_leftbottom[2]
    projected_rightbottom = np.dot(H2to1, rightbottom)
    projected_rightbottom /= projected_rightbottom[2]
    new_W = int(max(projected_righttop[0], projected_rightbottom[0]))
    new_H = int(max(projected_leftbottom[1], projected_rightbottom[1]))
    move_W = int(max(-projected_lefttop[0], -projected_leftbottom[0], 0))
    move_H = int(max(-projected_lefttop[1], -projected_righttop[1], 0))
    new_W += move_W
    new_H += move_H
    M = np.float32([[1, 0, move_W], [0, 1, move_H], [0, 0, 1]])
    warp_im1 = cv2.warpPerspective(im1, M, (new_W, new_H))
    warp_im2 = cv2.warpPerspective(im2, np.dot(M, H2to1), (new_W, new_H))
    cv2.imwrite('../results/q6_2_im1.jpg', warp_im1)
    cv2.imwrite('../results/q6_2_im2.jpg', warp_im2)
    mask1 = cv2.warpPerspective(blend_mask(im1), M, (new_W, new_H))
    mask2 = cv2.warpPerspective(blend_mask(
        im2), np.dot(M, H2to1), (new_W, new_H))
    sum_mask = mask1 + mask2
    mask1 /= sum_mask
    mask1 = np.repeat(mask1, 3).reshape(warp_im1.shape)
    mask2 /= sum_mask
    mask2 = np.repeat(mask2, 3).reshape(warp_im2.shape)
    pano_im = warp_im1 * mask1 + warp_im2 * mask2
    cv2.imwrite('../results/q6_2_pan.jpg', pano_im)
    return pano_im


def generatePanorama(im1, im2):
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    return pano_im


if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    # im1 = cv2.imread('../data/1.jpg')
    # im2 = cv2.imread('../data/2.jpg')
    im3 = generatePanorama(im1, im2)
    cv2.imwrite('../results/q6_3.jpg', im3)
