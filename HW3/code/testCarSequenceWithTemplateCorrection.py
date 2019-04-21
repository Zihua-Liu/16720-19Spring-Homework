import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import cv2
from LucasKanade import LucasKanade

if __name__ == "__main__":
	carseq = np.load("../data/carseq.npy")
	carseqrects = np.load("../result/carseqrects.npy")
	frames = [carseq[:, :, i] for i in range(carseq.shape[2])]
	frame = frames[0]
	rect = np.array([59, 116, 145, 151])
	carseqrects_wcrt = [rect]
	threshold = 0.5
	previous_p = None
	for i in range(1, len(frames)):
		next_frame = frames[i]
		p = LucasKanade(frame, next_frame, rect)
		next_rect = rect + np.array([p[0], p[1], p[0], p[1]])
		carseqrects_wcrt.append(next_rect)
		img = np.repeat(next_frame[:, :, np.newaxis], 3, axis=2)
		cv2.rectangle(
			img, 
			(int(next_rect[0]), int(next_rect[1])), 
			(int(next_rect[2]), int(next_rect[3])),
			color=(0, 255, 0)
			)
		cv2.rectangle(
			img, 
			(int(carseqrects[i][0]), int(carseqrects[i][1])), 
			(int(carseqrects[i][2]), int(carseqrects[i][3])),
			color=(0, 0, 255)
			)
		cv2.imshow("image", img)
		cv2.waitKey(1)
		if i in [1, 100, 200, 300, 400]:
			cv2.imwrite("../result/q1-4_{}.jpg".format(i), img * 255)
		if previous_p is None or np.sqrt(np.sum(np.square(p - previous_p))) < threshold:
			rect = next_rect
			previous_p = np.copy(p)
			frame = next_frame
	carseqrects_wcrt = np.array(carseqrects_wcrt)
	print(carseqrects_wcrt.shape)
	np.save("../result/carseqrects-wcrt.npy", carseqrects_wcrt)
