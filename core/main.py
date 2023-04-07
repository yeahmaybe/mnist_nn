import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from MnistNet import MnistNet

# %%
# define model parameters
NUM_EPOCHS = 20  # original paper
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 28  # pixels
NUM_CLASSES = 10  # 10 classes for mnist dataset
DEVICE_IDS = [0, 1, 2, 3]  # GPUs to use

# define pytorch device - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
# print the seed value
# seed = torch.initial_seed()
# print('Used seed : {}'.format(seed))
# %%
mnistnet = MnistNet(num_classes=NUM_CLASSES).to(device)
# train on multiple GPUs
mnistnet = torch.nn.parallel.DataParallel(mnistnet, device_ids=DEVICE_IDS)
print(mnistnet)
print('MnistNet created')
# %%
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
y = y.astype(int)
# %%
print(X.shape)
X = X.reshape(X.shape[0], 1, 28, 28)
print(X.shape)
# %%
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
train_dataset = TensorDataset(torch.from_numpy(X_train).float(),
                              torch.from_numpy(y_train).long())
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=8,
                          drop_last=True,
                          batch_size=BATCH_SIZE)

test_dataset = TensorDataset(torch.from_numpy(X_test).float(),
                             torch.from_numpy(y_test).long())
test_loader = DataLoader(test_dataset,
                         shuffle=True,
                         pin_memory=True,
                         num_workers=8,
                         drop_last=True,
                         batch_size=BATCH_SIZE)
# %%
# create optimizer
# the one that WORKS
optimizer = optim.Adam(params=mnistnet.parameters(), lr=0.0001)
### BELOW is the setting proposed by the original paper - which doesn't train....
# optimizer = optim.SGD(
#     params=alexnet.parameters(),
#     lr=LR_INIT,
#     momentum=MOMENTUM,
#     weight_decay=LR_DECAY)
print('Optimizer created')
# %%
# multiply LR by 1 / 10 after every 30 epochs
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
print('LR Scheduler created')

# start training!!
print('Starting training...')
total_steps = 1
for epoch in range(NUM_EPOCHS):
    lr_scheduler.step()
    for imgs, classes in train_loader:
        imgs, classes = imgs.to(device), classes.to(device)

        # calculate the loss
        output = mnistnet(imgs)
        loss = F.cross_entropy(output, classes)

        # update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log the information and add to tensorboard
        if total_steps % 10 == 0:
            with torch.no_grad():
                _, preds = torch.max(output, 1)
                accuracy = torch.sum(preds == classes) / len(preds)

                print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                      .format(epoch + 1, total_steps, loss.item(), accuracy.item()))

        # print out gradient values and parameter average values
        if total_steps % 100 == 0:
            with torch.no_grad():
                # print and save the grad of the parameters
                # also print and save parameter values
                print('*' * 10)
                for name, parameter in mnistnet.named_parameters():
                    if parameter.grad is not None:
                        avg_grad = torch.mean(parameter.grad)
                        print('\t{} - grad_avg: {}'.format(name, avg_grad))
                        # tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
                        # tbwriter.add_histogram('grad/{}'.format(name),
                        #                       parameter.grad.cpu().numpy(), total_steps)
                    if parameter.data is not None:
                        avg_weight = torch.mean(parameter.data)
                        print('\t{} - param_avg: {}'.format(name, avg_weight))
                        # tbwriter.add_histogram('weight/{}'.format(name),
                        #                       parameter.data.cpu().numpy(), total_steps)
                        # tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)

        total_steps += 1
# %%
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
# %%
mnistnet.eval()
with torch.no_grad():
    correct_out = 0
    total_out = 0
    for pics, lbls in test_loader:
        out = mnistnet(pics)
        pred = torch.argmax(out, dim=1)
        total_out += lbls.shape[0]
        correct_out += (pred == lbls).sum().item()

loss_current = criterion(out, lbls)
print(correct_out / total_out, loss_current)
# %%
import cv2
import numpy as np

data = []
classes = [2, 3, 6, 8, 9, 1, 0, 4, 7, 5]
for i in range(10):
    img = cv2.imread("../my_dataset/" + str(i + 1) + ".png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharp_filter = np.array([[-1, 0, -1],
                             [0, 7, 0],
                             [-1, 0, -1]])
    img = cv2.filter2D(img, ddepth=-1, kernel=sharp_filter)

    img = cv2.bitwise_not(img)
    img = img.astype('float32')
    data.append(img)

data = np.reshape(data, (len(data), 1, 28, 28))
# data = np.array(data)
classes = np.array(classes)

print(data.shape)
# %%
plt.figure(figsize=(10, 7))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(data[i].reshape(28, 28), cmap='Greys')
    plt.title("Цифра %d" % classes[i])
# %%
experiment_dataset = TensorDataset(torch.from_numpy(data).float(),
                                   torch.from_numpy(classes).long())
experiment_loader = DataLoader(experiment_dataset,
                               shuffle=False,
                               # pin_memory=True,
                               # num_workers=8,
                               # drop_last=True,

                               batch_size=BATCH_SIZE)

mnistnet.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in experiment_loader:
        outputs = mnistnet(images)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.shape[0]
        correct += (predicted == labels).sum().item()

    print("Test accuracy:",
          100 * correct / total, "%")

# %%
plt.figure(figsize=(10, 7))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(data[i].reshape(28, 28), cmap='Greys')
    plt.title("%d, pred: %d" % (classes[i], predicted[i]))