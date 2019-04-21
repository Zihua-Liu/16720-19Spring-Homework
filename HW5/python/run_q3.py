import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 32
learning_rate = 1e-2
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
n_input_layer = train_x.shape[1]
n_output_layer = train_y.shape[1]
initialize_weights(n_input_layer, hidden_size, params, 'layer1')
initialize_weights(hidden_size, n_output_layer, params, 'output')

n_examples = train_x.shape[0]

x = np.arange(max_iters)
train_acc_list = []
valid_acc_list = []
train_loss_list = []
valid_loss_list = []
# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # forward
        out = forward(xb, params, "layer1", sigmoid)
        probs = forward(out, params, "output", softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc

        # backward
        delta = probs - yb
        delta = backwards(delta, params, "output", linear_deriv)
        delta = backwards(delta, params, "layer1", sigmoid_deriv)

        # apply gradient
        params["Wlayer1"] -= learning_rate * params["grad_Wlayer1"]
        params["blayer1"] -= learning_rate * params["grad_blayer1"]
        params["Woutput"] -= learning_rate * params["grad_Woutput"]
        params["boutput"] -= learning_rate * params["grad_boutput"]
    
    total_acc /= batch_num
    total_loss /= n_examples   

    out = forward(valid_x, params, "layer1", sigmoid)
    probs = forward(out, params, "output", softmax)
    valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)
    valid_loss /= valid_x.shape[0]

    train_acc_list.append(total_acc)
    valid_acc_list.append(valid_acc)
    train_loss_list.append(total_loss)
    valid_loss_list.append(valid_loss)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
plt.plot(x, train_acc_list, linewidth = 3, label = "Training Accuracy")
plt.plot(x, valid_acc_list, linewidth = 3, label = "Validation Accuracy")
plt.legend()
plt.show()

plt.plot(x, train_loss_list, linewidth = 3, label = "Training Loss")
plt.plot(x, valid_loss_list, linewidth = 3, label = "Validation Loss")
plt.legend()
plt.show()

# run on validation set and report accuracy! should be above 75%
valid_acc = None
out = forward(valid_x, params, "layer1", sigmoid)
probs = forward(out, params, "output", softmax)
_, valid_acc = compute_loss_and_acc(valid_y, probs)

print('Validation accuracy: ',valid_acc)

test_acc = None
out = forward(test_x, params, "layer1", sigmoid)
probs = forward(out, params, "output", softmax)
_, test_acc = compute_loss_and_acc(test_y, probs)

print('Test accuracy: ',test_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
from mpl_toolkits.axes_grid1 import ImageGrid
def visualize_weights(W):
    assert W.shape == (32 * 32, 64)
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols = (8, 8))
    for i in range(64):
        grid[i].imshow(W[:, i].reshape(32, 32))
    plt.axis("off")
    plt.show()
visualize_weights(params["Wlayer1"])


# Q3.1.4
valid_acc = None
out = forward(valid_x, params, "layer1", sigmoid)
probs = forward(out, params, "output", softmax)
_, valid_acc = compute_loss_and_acc(valid_y, probs)
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
valid_pred_y = np.argmax(probs, axis = 1)
for i in range(valid_pred_y.shape[0]):
    pred = valid_pred_y[i]
    label = np.argmax(valid_y[i])
    confusion_matrix[label][pred] += 1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()