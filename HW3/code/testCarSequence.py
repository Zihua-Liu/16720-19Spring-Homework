import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import cv2
from LucasKanade import LucasKanade

if __name__ == "__main__":
	carseq = np.load("../data/carseq.npy")
	frames = [carseq[:, :, i] for i in range(carseq.shape[2])]
	rect = np.array([59, 116, 145, 151])
	carseqrects = [rect]
	for i in range(0, len(frames) - 1):
		frame, next_frame = frames[i], frames[i + 1]
		p = LucasKanade(frame, next_frame, rect)
		rect = rect + np.array([p[0], p[1], p[0], p[1]])
		carseqrects.append(rect)
		img = np.repeat(next_frame[:, :, np.newaxis], 3, axis=2)
		cv2.rectangle(
			img, 
			(int(rect[0]), int(rect[1])), 
			(int(rect[2]), int(rect[3])),
			color=(0, 0, 255)
			)
		cv2.imshow("image", img)
		cv2.waitKey(1)
		if i in [1, 100, 200, 300, 400]:
			cv2.imwrite("../result/q1-3_{}.jpg".format(i), img * 255)
	carseqrects = np.array(carseqrects)
	print(carseqrects.shape)
	np.save("../result/carseqrects.npy", carseqrects)
		

