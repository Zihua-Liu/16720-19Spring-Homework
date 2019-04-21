import numpy as np
import scipy.ndimage.morphology as morphology
from scipy.interpolate import RectBivariateSpline
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2):
    # Input:
    #   Images at time t and t+1 
    # Output:
    #   mask: [nxm]
    # put your implementation here
    
    # mask = np.ones(image1.shape, dtype=bool)
    mask = np.zeros(image2.shape, dtype=bool)
    M = LucasKanadeAffine(image1, image2)
    # M = InverseCompositionAffine(image1, image2)
    interpolated_im1 = RectBivariateSpline(
            x = np.array([i for i in range(int(image1.shape[0]))]), 
            y = np.array([i for i in range(int(image1.shape[1]))]), 
            z = image1
        )
    interpolated_im2 = RectBivariateSpline(
            x = np.array([i for i in range(int(image2.shape[0]))]), 
            y = np.array([i for i in range(int(image2.shape[1]))]), 
            z = image2
        )
    x = np.arange(0, image1.shape[1])
    y = np.arange(0, image1.shape[0])
    X = np.array([x for i in range(len(y))])
    Y = np.array([y for i in range(len(x))]).T
    warp_X = M[0, 0] * X + M[0, 1] * Y + M[0, 2]
    warp_Y = M[1, 0] * X + M[1, 1] * Y + M[1, 2]
    invalid_points = (warp_X < 0)|(warp_X >= image2.shape[1])|(warp_Y < 0)|(warp_Y >= image2.shape[0]) 
    im1 = interpolated_im1.ev(Y, X)
    warp_im2 = interpolated_im2.ev(warp_Y, warp_X)
    im1[invalid_points] = .0
    warp_im2[invalid_points] = .0
    diff = np.abs(im1 - warp_im2)
    thresold = 2e-1
    index = (diff > thresold) & (im1 != .0) & (warp_im2 != .0)
    mask[index] = 1
    ker = np.array(([0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]))
    mask = morphology.binary_dilation(mask, structure=ker).astype(mask.dtype)
    return mask
