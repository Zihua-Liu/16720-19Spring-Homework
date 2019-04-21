import numpy as np
import cv2


def createGaussianPyramid(im, sigma0=1,
                          k=np.sqrt(2), levels=[-1, 0, 1, 2, 3, 4]):
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max() > 10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i
        im_pyramid.append(cv2.GaussianBlur(im, (0, 0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid


def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(
        im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0)  # press any key to exit
    cv2.destroyAllWindows()


def displayKeypoints(image, keypoints, zoom=1):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x, y, _ in keypoints:
        cv2.circle(image, (x, y), 1, [0, 255, 0])
    H, W, C = image.shape
    image = cv2.resize(image, (W * zoom, H * zoom))
    cv2.imshow("a", image)
    cv2.waitKey(0)


def createDoGPyramid(gaussian_pyramid, levels=[-1, 0, 1, 2, 3, 4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    gaussian_pyramid_list = [
        blured_image.squeeze(2) for blured_image in np.split(gaussian_pyramid, gaussian_pyramid.shape[2], axis=2)
    ]
    for l in range(1, len(gaussian_pyramid_list)):
        DoG_pyramid.append(
            gaussian_pyramid_list[l] - gaussian_pyramid_list[l - 1]
        )
    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)
    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels


def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid

    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid

    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each
                          point contains the curvature ratio R for the
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = []
    DoG_layers = [
        layer.squeeze(2) for layer in np.split(DoG_pyramid, DoG_pyramid.shape[2], axis=2)
    ]
    for layer in DoG_layers:
        dx = cv2.Sobel(layer, ddepth=-1, dx=1, dy=0)
        dy = cv2.Sobel(layer, ddepth=-1, dx=0, dy=1)
        dxx = cv2.Sobel(dx, ddepth=-1, dx=1, dy=0)
        dxy = cv2.Sobel(dx, ddepth=-1, dx=0, dy=1)
        dyx = cv2.Sobel(dy, ddepth=-1, dx=1, dy=0)
        dyy = cv2.Sobel(dy, ddepth=-1, dx=0, dy=1)
        det = dxx * dyy - dxy * dyx
        det[det == 0.0] = 1e-6  # To solve the devide by zero problem
        trace = dxx + dyy
        R = trace ** 2 / det
        principal_curvature.append(R)
    principal_curvature = np.stack(principal_curvature, axis=-1)
    return principal_curvature


def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
                    th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = []
    DoG_layers = [
        layer.squeeze(2) for layer in np.split(DoG_pyramid, DoG_pyramid.shape[2], axis=2)
    ]
    principal_curvature_layers = [
        layer.squeeze(2) for layer in np.split(principal_curvature, principal_curvature.shape[2], axis=2)
    ]
    for l, DoG_layer in enumerate(DoG_layers):
        valid_sub_area = DoG_layer[1:-1, 1:-1]
        left_neighbor = DoG_layer[1:-1, 0:-2]
        right_neighbor = DoG_layer[1:-1, 2:]
        top_neighbor = DoG_layer[0:-2, 1:-1]
        bottom_neighbor = DoG_layer[2:, 1:-1]
        lefttop_neighbor = DoG_layer[0:-2, 0:-2]
        leftbottom_neighbor = DoG_layer[2:, 0:-2]
        righttop_neighbor = DoG_layer[0:-2, 2:]
        rightbottom_neighbor = DoG_layer[2:, 2:]
        neighbors = [
            left_neighbor, right_neighbor, top_neighbor, bottom_neighbor,
            lefttop_neighbor, leftbottom_neighbor, righttop_neighbor, rightbottom_neighbor
        ]
        if l > 0:
            neighbors.append(DoG_layers[l - 1][1:-1, 1:-1])
        if l < len(DoG_layers) - 1:
            neighbors.append(DoG_layers[l + 1][1:-1, 1:-1])
        neighbors = np.array(neighbors)
        local_maximum = valid_sub_area > np.max(neighbors, axis=0)
        local_minmum = valid_sub_area < np.min(neighbors, axis=0)
        DoG_response_magnitude = np.abs(valid_sub_area) > th_contrast
        not_edge_like = np.abs(
            principal_curvature_layers[l][1:-1, 1:-1]) <= th_r
        coordinates = np.where(((local_maximum | local_minmum)
                                & DoG_response_magnitude & not_edge_like) == True)
        y_list, x_list = coordinates[0] + 1, coordinates[1] + 1
        for x, y in zip(x_list, y_list):
            locsDoG.append((x, y, DoG_levels[l]))
    return locsDoG


def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1, 0, 1, 2, 3, 4],
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    gauss_pyramid = createGaussianPyramid(im)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(
        DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    return locsDoG, gauss_pyramid


if __name__ == '__main__':
    #  test gaussian pyramid
    # levels = [-1, 0, 1, 2, 3, 4]
    # im = cv2.imread('../data/model_chickenbroth.jpg')
    # im_pyr = createGaussianPyramid(im)
    # # displayPyramid(im_pyr)

    # # test DoG pyramid
    # DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    # # displayPyramid(DoG_pyr)

    # # test compute principal curvature
    # pc_curvature = computePrincipalCurvature(DoG_pyr)

    # # test get local extrema
    # th_contrast = 0.03
    # th_r = 12
    # locsDoG = getLocalExtrema(
    #     DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)#

    # test DoG detector
    im = cv2.imread('../data/incline_L.png')
    # im = cv2.imread('../data/chickenbroth_01.jpg')
    locsDoG, gaussian_pyramid = DoGdetector(im)
    displayKeypoints(im, locsDoG)
