import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeBasis(It, It1, rect, bases):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	bases: [n, m, k] where nxm is the size of the template.
	# Output:
	#	p: movement vector [dp_x, dp_y]

    # Put your implementation here
    p = np.zeros(2)

    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
    threshold = 1e-3
    dp = np.array([float("inf"), float("inf")])
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
    B = np.array([bases[:, :, i].flatten() for i in range(bases.shape[2])]).T
    B_null = np.eye(B.shape[0]) - np.dot(B, B.T)

    while np.sum(np.square(dp)) >= threshold:
    	warp_x = np.arange(x1 + p[0], x2 + p[0] + .5)
    	warp_y = np.arange(y1 + p[1], y2 + p[1] + .5)
    	warp_X = np.array([warp_x for i in range(len(warp_y))])
    	warp_Y = np.array([warp_y for i in range(len(warp_x))]).T
    	warp_It1x = interpolated_It1.ev(warp_Y, warp_X)

    	x = np.arange(x1, x2 + .5)
    	y = np.arange(y1, y2 + .5)
    	X = np.array([x for i in range(len(y))])
    	Y = np.array([y for i in range(len(x))]).T
    	Itx = interpolated_It.ev(Y, X)

    	A = np.array([
    			interpolated_It1.ev(warp_Y, warp_X, dx = 0, dy = 1).flatten(),
    			interpolated_It1.ev(warp_Y, warp_X, dx = 1, dy = 0).flatten()
    		]).T
    	A = np.dot(B_null, A)
    	b = (Itx - warp_It1x).flatten()
    	b = b.reshape(b.shape[0], 1)
    	b = np.dot(B_null, b)

    	dp = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
    	p += dp.flatten()
    	
    return p
    
