import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

import string

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Sequential(
				nn.Conv2d(1, 4, kernel_size = 3, stride = 1, padding = 1),
				nn.ReLU(),
				nn.Conv2d(4, 8, kernel_size = 3, stride = 1, padding = 1),
				nn.ReLU(),
				nn.MaxPool2d(stride = 2, kernel_size = 2)
			)
		self.conv2 = nn.Sequential(
				nn.Conv2d(8, 16, kernel_size = 3, stride = 1, padding = 1),
				nn.ReLU(),
				nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
				nn.ReLU(),
				nn.MaxPool2d(stride = 2, kernel_size = 2)
			)
		self.fc = nn.Sequential(
				nn.Linear(7 * 7 * 32, 1024),
				nn.ReLU(),
				nn.Linear(1024, 47)
			)
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(-1, 7 * 7 * 32)
		x = self.fc(x)
		return x

cnn = CNN()
cnn.load_state_dict(torch.load("trained_weights"))

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw, cmap = "gray")
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    points = []
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        center = ((minr + maxr) // 2, (minc + maxc) // 2)
        points.append((center, bbox))
    points = sorted(points, key = lambda x: x[0])
    rows = []
    for point in points:
        find = False
        center, bbox = point
        for row in rows:
            average_height = sum([p[1][2] - p[1][0] for p in row]) / float(len(row))
            average_center_r = sum([p[0][0] for p in row]) / float(len(row))
            if average_center_r - average_height < center[0] < average_center_r + average_height:
                row.append(point)
                find = True
                break
        if not find:
            rows.append([point])
    for i in range(len(rows)):
        rows[i] = sorted(rows[i], key = lambda x: x[0][1])
    
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    dataset = []
    for row in rows:
        data_row = []
        for point in row:
            center, bbox = point
            minr, minc, maxr, maxc = bbox
            image = bw[minr : maxr + 1, minc : maxc + 1]
            H, W = image.shape
            if H > W:
                W_left = (H - W) // 2
                W_right = H - W - W_left
                image = np.pad(image, ((H // 20, H // 20), (W_left + H // 20, W_right + H // 20)), "constant", constant_values = (1, 1))
            else:
                H_top = (W - H) // 2
                H_bottom = W - H - H_top
                image = np.pad(image, ((H_top + W // 20, H_bottom + W // 20), (W // 20, W // 20)), "constant", constant_values = (1, 1))
            image = skimage.transform.resize(image, (28, 28))
            image = skimage.morphology.erosion(image, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
            image = np.ones(image.shape) - image
            data_row.append(np.transpose(image))
            # plt.imshow(image, cmap='gray')
            # plt.show()
        dataset.append(np.array(data_row))
    

    letters = np.array([str(_) for _ in range(10)] + [_ for _ in string.ascii_uppercase[:26]] + ["a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t"])
    # load the weights
    # run the crops through your neural network and print them out
    print("-" * 50)
    for row in dataset:
        x = torch.from_numpy(np.expand_dims(row, axis = 1)).type(torch.FloatTensor)
        out = cnn(torch.autograd.Variable(x))
        pred_y = torch.max(out, 1)[1].data.numpy()
        row_pred = ""
        for pred in pred_y:
            row_pred += (letters[pred] + " ")
        print(row_pred)
print("-" * 50)