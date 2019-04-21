import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import os
import cv2
from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation
if __name__ == '__main__':
    aerialseq = np.load('../data/aerialseq.npy')
    frames = [aerialseq[:, :, i] for i in range(aerialseq.shape[2])]
    for i in range(0, len(frames) - 1):
        frame, next_frame = frames[i], frames[i + 1]
        mask = SubtractDominantMotion(frame, next_frame)
        img = np.repeat(next_frame[:, :, np.newaxis], 3, axis=2)
        img[:, :, 2][mask == 1] = 1
        cv2.imshow('image', img)
        cv2.waitKey(1)
        if i in [30, 60, 90, 120]:
            cv2.imwrite("../result/q3-3_{}.jpg".format(i), img * 255)
