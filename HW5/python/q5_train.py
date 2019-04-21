import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from progress.bar import Bar

epochs = 10
batch_size = 64
learning_rate = 0.001
download_dataset = False

if not os.path.exists("./emnist/"):
	download_dataset = True

train_data = torchvision.datasets.EMNIST(
		root = "./emnist/",
		split = "balanced",
		train = True,
		transform = torchvision.transforms.ToTensor(),
		download = download_dataset
	)

train_x = train_data.train_data
train_y = train_data.train_labels
print("Training Data Shape: {}".format(train_x.shape))
print("Training Label Shape: {}".format(train_y.shape))
# plt.imshow(train_x[0].numpy(), cmap='gray')
# plt.show()
train_loader = Data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)

test_data = torchvision.datasets.EMNIST(
		root = "./emnist/",
		split = "balanced",
		train = False,
		transform = torchvision.transforms.ToTensor(),
		download = download_dataset
	)

test_x = test_data.test_data
test_y = test_data.test_labels
print("Test Data Shape: {}".format(test_x.shape))
print("Test Label Shape: {}".format(test_y.shape))
test_loader = Data.DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False)

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
optimizer = torch.optim.Adam(cnn.parameters(), lr = learning_rate)
loss_func = nn.CrossEntropyLoss() 

for epoch in range(epochs):
	bar = Bar('Training epoch {}'.format(epoch), max = len(train_loader))
	for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
		out = cnn(batch_x)
		loss = loss_func(out, batch_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		bar.next()
	bar.finish()

	correct = 0.0
	for batch_idx, (x, y) in enumerate(test_loader):
		test_out = cnn(x)
		pred_y = torch.max(test_out, 1)[1].data.numpy()
		correct += float((pred_y == y.data.numpy()).astype(int).sum())
	print("Epoch: {} | Accuracy: {}".format(epoch, correct / float(test_x.shape[0])))
	torch.save(cnn.state_dict(), "trained_weights".format(epoch))


























