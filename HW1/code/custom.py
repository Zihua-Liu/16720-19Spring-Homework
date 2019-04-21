import torch
import numpy as np
import os
import skimage.io
from skimage import transform
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn as nn
from torch.autograd import Variable 
from progress.bar import Bar
import cv2



def load_data(train = True):
    if train:
        print("Loading training data...")
        data = np.load("../data/train_data.npz")
    else:
        print("Loading test data")
        data = np.load("../data/test_data.npz")
    files = data['files']
    labels = data['labels']

    data = []

    for i, (path, label) in enumerate(zip(files, labels)):
        image_path = os.path.join("../data/", path)
        image = cv2.imread(image_path)
        try:
            image = cv2.resize(image, (224, 224))
        except:
            continue
        data.append((image, label))
    print("Loading {} images".format(len(data)))
    print("Done!")
    print("-" * 50)
    return data



class MyDataset(Dataset):
    def __init__(self, data):
        self.images, self.labels = [], []
        for (image, label) in data:
            self.images.append(np.transpose(image, (2, 0, 1)))
            self.labels.append(label)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 7, stride = 1, padding = 3),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 3),
                nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3),
                nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2),
                nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2),
                )
        self.fc = nn.Linear(256, 8)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out





def train():
    training_data = load_data(train = True)
    training_set = MyDataset(training_data)
    test_data = load_data(train = False)
    test_set = MyDataset(test_data)
    # model = models.vgg16(pretrained = True)
    model = CNN()
    training_loader = DataLoader(training_set, batch_size = 16, shuffle = True)
    test_loader = DataLoader(test_set, batch_size = 32, shuffle = False)
    # for p in model.parameters():
        # p.requires_grad = False
    # model.classifier._modules['6'] = nn.Linear(in_features = 4096, out_features = 8, bias = True)
    # for p in model.classifier._modules['6'].parameters():
        # p.requires_grad = True

    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    epochs = 20
    for epoch in range(epochs):
        model.train()
        print("Epoch {}".format(epoch))
        correct = 0
        bar = Bar('Training', max = len(training_loader))
        for image, label in training_loader:
            image, label = Variable(image.float()), Variable(label)
            optimizer.zero_grad()
            y_pred = model(image)
            _, pred = torch.max(y_pred.data, 1)
            loss = cost(y_pred, label)
            loss.backward()
            optimizer.step()
            correct += torch.sum(pred == label.data)
            bar.next()
        print("\nTraining Accuracy: {}".format(float(correct) / len(training_set)))
        test(model, test_loader, len(test_data))
        print('-' * 50)
        bar.finish()

def test(model, test_loader, test_len):
    model.eval()
    correct = 0
    bar = Bar('Testing', max = len(test_loader))
    for image, label in test_loader:
        image, label = Variable(image.float()), Variable(label)
        y_pred = model(image)
        _, pred = torch.max(y_pred.data, 1)
        correct += torch.sum(pred == label.data)
        bar.next()
    print("\nTest Accuracy: {}".format(float(correct) / test_len))
    bar.finish()
    return



if __name__ == "__main__":
    train()
