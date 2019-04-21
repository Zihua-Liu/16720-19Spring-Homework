import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import cv2
from LucasKanadeBasis import LucasKanadeBasis
from LucasKanade import LucasKanade

if __name__ == "__main__":
	sylvseq = np.load("../data/sylvseq.npy")
	bases = np.load("../data/sylvbases.npy")
	frames = [sylvseq[:, :, i] for i in range(sylvseq.shape[2])]
	rect_basis = np.array([101, 61, 155, 107])
	rect = np.array([101, 61, 155, 107])
	sylvseqrects = [rect_basis]
	for i in range(0, len(frames) - 1):
		frame, next_frame = frames[i], frames[i + 1]
		p = LucasKanadeBasis(frame, next_frame, rect_basis, bases)
		p_ = LucasKanade(frame, next_frame, rect)
		rect_basis = rect_basis + np.array([p[0], p[1], p[0], p[1]])
		rect = rect + np.array([p_[0], p_[1], p_[0], p_[1]])
		sylvseqrects.append(rect_basis)
		img = np.repeat(next_frame[:, :, np.newaxis], 3, axis=2)
		cv2.rectangle(
			img, 
			(int(rect[0]), int(rect[1])), 
			(int(rect[2]), int(rect[3])),
			color=(0, 255, 0)
			)
		cv2.rectangle(
			img, 
			(int(rect_basis[0]), int(rect_basis[1])), 
			(int(rect_basis[2]), int(rect_basis[3])),
			color=(0, 0, 255)
			)
		cv2.imshow("image", img)
		cv2.waitKey(1)
		if i in [1, 200, 300, 350, 400]:
			cv2.imwrite("../result/q2-3_{}.jpg".format(i), img * 255)
	sylvseqrects = np.array(sylvseqrects)
	print(sylvseqrects.shape)
	np.save("../result/sylvseqrects.npy", sylvseqrects)
