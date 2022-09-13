import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from SiameseNetwork import SiameseNetwork, ContrastiveLoss, SiameseNetworkNew
from data_preprocess import BreastCancerDataset

import numpy as np
import torch

from utils import EarlyStopping


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


epoch_num = 100
# early stopping
patience = 10
threshold = 0.25

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "./models/googlenet_SVMSMOTE"
print(model_path)
# Load the training dataset
dataset_path = "./dataset/train_SVMSMOTE.csv"

# Initialize the network
cancer_dataset = BreastCancerDataset(imageFolderDataset=dataset_path,
                                     transform=None)

train_size = int(0.8 * cancer_dataset.__len__())
test_size = cancer_dataset.__len__() - train_size
train_dataset, test_dataset = torch.utils.data.random_split(cancer_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset,
                              shuffle=True,
                              num_workers=0,
                              batch_size=10)
test_dataloader = DataLoader(test_dataset,
                             shuffle=True,
                             num_workers=0,
                             batch_size=10)

model = SiameseNetworkNew()
model.to(device)
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
# initialize the early_stopping object
early_stopping = EarlyStopping(patience=patience, verbose=True, save_path=model_path)

train_loss_plot, train_acc_plot = [], []
valid_loss_plot, valid_acc_plot = [], []

# Iterate throught the epochs
for epoch in range(1, epoch_num + 1):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the train and validation accuracy as the model trains
    train_pred_acc, train_true_acc = [], []
    valid_pred_acc, valid_true_acc = [], []

    # torch.save(model.state_dict(), model_path)
    ###################
    # train the model #
    ###################
    train_loss = 0
    train_acc = 0
    for batch in tqdm(train_dataloader):
        model.train()
        # Send the images and labels to CUDA
        left, right, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # Pass in the two images into the network and obtain two outputs
        output1, output2 = model(left, right)
        # Pass the outputs of the networks and label into the loss function
        loss_contrastive = criterion(output1, output2, label)
        # Calculate the backpropagation
        loss_contrastive.backward()
        # Optimize
        optimizer.step()

        euclidean_distance = F.pairwise_distance(output1, output2)
        threshold = torch.tensor([threshold]).cuda()
        predicted = ((euclidean_distance > threshold).float() * 1).int()
        train_pred_acc.extend(predicted.detach().cpu().numpy())
        train_true_acc.extend(label.detach().cpu().numpy())

        train_losses.append(loss_contrastive.item())


    ###################
    # validate the model #
    ###################
    eval_loss = 0
    eval_acc = 0
    for batch in tqdm(test_dataloader):
        model.eval()
        # Send the images and labels to CUDA
        left, right, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        output1, output2 = model(left, right)
        loss_contrastive = criterion(output1, output2, label)
        valid_losses.append(loss_contrastive.item())

        euclidean_distance = F.pairwise_distance(output1, output2)
        threshold = torch.tensor([threshold]).cuda()
        predicted = ((euclidean_distance > threshold).float() * 1).int()
        valid_pred_acc.extend(predicted.detach().cpu().numpy())
        valid_true_acc.extend(label.detach().cpu().numpy())

    # print training/validation statistics
    # calculate average loss over an epoch
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    train_loss_plot.append(train_loss)
    valid_loss_plot.append(valid_loss)

    # calculate accuracies over an epoch
    train_acc = accuracy_score(train_true_acc, train_pred_acc)
    valid_acc = accuracy_score(valid_true_acc, valid_pred_acc)
    train_acc_plot.append(train_acc)
    valid_acc_plot.append(valid_acc)

    epoch_len = len(str(epoch_num))

    print_msg = (f'[{epoch:>{epoch_len}}/{epoch_num:>{epoch_len}}] ' +
                 f'train_loss: {train_loss:.5f} ' +
                 f'valid_loss: {valid_loss:.5f}')
    print(print_msg)
    print_msg = (f'[{epoch:>{epoch_len}}/{epoch_num:>{epoch_len}}] ' +
                 f'train_acc: {train_acc:.5f} ' +
                 f'valid_acc: {valid_acc:.5f}')
    print(print_msg)

    # clear lists to track next epoch
    train_losses = []
    valid_losses = []

    # early_stopping needs the validation loss to check if it has decresed,
    # and if it has, it will make a checkpoint of the current model
    early_stopping(epoch, valid_acc, valid_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break


###################
# plot acc & loss #
###################

model_path = "./plot/googlenet_SVMSMOTE"

# Loss and Accuracy Curve
plt.style.use("ggplot")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(np.arange(len(train_loss_plot)), train_loss_plot, label="train_loss")
plt.plot(np.arange(len(valid_loss_plot)), valid_loss_plot, label="val_loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.xticks(np.arange(0, len(valid_loss_plot), 5))
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.arange(len(train_acc_plot)), train_acc_plot, label="train_acc")
plt.plot(np.arange(len(valid_acc_plot)), valid_acc_plot, label="valid_acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.xticks(np.arange(0, len(valid_acc_plot), 5))
plt.legend()

plt.savefig(model_path + '_epoch' + str(epoch) + '_valAcc' + str(round(valid_acc_plot[-1], 2)) + '_valLoss' + str(round(valid_loss_plot[-1], 2)) + '.pdf')

