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
from SiameseNetworkGoogle import ContrastiveLoss, SiameseNetworkGoogle
from data_preprocess import BreastCancerDataset

import numpy as np
import torch

from utils import EarlyStopping, FocalLoss, EarlyStoppingAcc, FocalLossNoAlpha


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


epoch_num = 2000
# early stopping
patience = 100

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_type = "GoogLeNet"
model_name = 'SiameseGoogle_rotate_lr10-5_BCE_best'
# model_name = 'SiameseDense_hidlayer6_128unite_lr10-5_BCE'
model_path = "./models/" + model_name
plot_path = "./plot/" + model_name
num_hidden_layers = 6
num_hidden_unit = 256
gamma = 2
alpha = 0.5
print('model name: ' + model_name)

# Load the training and validation dataset
# train_set_path = "./dataset/train_ADASYN_new_withoutoversample.csv"
train_set_path = "./dataset/train_ADASYN_new_rotate.csv"
valid_set_path = "./dataset/valid_ADASYN_new.csv"
# folder_dataset = datasets.ImageFolder(path)

# Resize the images and transform to tensors
# transformation = transforms.Compose([transforms.Resize((100,100)),
#                                      transforms.ToTensor()
#                                     ])

# Initialize the network
train_set = BreastCancerDataset(imageFolderDataset=train_set_path,
                                     transform=None, model=model_type)

valid_set = BreastCancerDataset(imageFolderDataset=valid_set_path,
                                     transform=None, model=model_type)
# print(siamese_dataset[1])
# Create a simple dataloader just for simple visualization
# cancer_dataloader = DataLoader(cancer_dataset,
#                                shuffle=True,
#                                num_workers=0,
#                                batch_size=8)

# Split training and testing set 0.8 / 0.2
# train_size = int(0.8 * cancer_dataset.__len__())
# test_size = cancer_dataset.__len__() - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(cancer_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_set,
                              shuffle=True,
                              num_workers=0,
                              batch_size=10)
valid_dataloader = DataLoader(valid_set,
                             shuffle=True,
                             num_workers=0,
                             batch_size=10)

# model = SiameseDenseNetwork(input_features=22,
#                             units=num_hidden_unit, embedding_size=num_hidden_unit,
#                             num_layers=num_hidden_layers,
#                             dropout_rate=0.1, use_batchnorm=True)

model = SiameseNetworkGoogle()
model.to(device)

# print(model)

criterion1 = ContrastiveLoss()
# criterion2 = FocalLoss(gamma=gamma, alpha=alpha)
criterion2 = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=5e-4)
# initialize the early_stopping object
# early_stopping = EarlyStoppingAcc(patience=patience, verbose=True, save_path=model_path)
early_stopping = EarlyStopping(patience=patience, verbose=True, save_path=model_path)
# 用数组保存每一轮迭代中，在测试数据上测试的损失值和精确度，也是为了通过画图展示出来。
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
    model.train()
    for batch in tqdm(train_dataloader):
        # Send the images and labels to CUDA
        left, right, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # Pass in the two images into the network and obtain two outputs
        output1, output2, output_classification = model(left, right)
        # Pass the outputs of the networks and label into the loss function
        loss_contrastive = criterion1(output1, output2, label)
        loss_classification = criterion2(output_classification, label)
        # Calculate the backpropagation
        loss = loss_contrastive + loss_classification
        loss.backward()
        # Optimize
        optimizer.step()

        predicted = output_classification.max(dim=1)[1]
        train_pred_acc.extend(predicted.detach().cpu().numpy())
        train_true_acc.extend(label.detach().cpu().numpy())

        train_losses.append(loss_contrastive.item())

    ###################
    # validate the model #
    ###################
    eval_loss = 0
    eval_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            # Send the images and labels to CUDA
            left, right, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            output1, output2, output_classification = model(left, right)

            loss_contrastive = criterion1(output1, output2, label)
            loss_classification = criterion2(output_classification, label)
            # Calculate the backpropagation
            loss = loss_contrastive + loss_classification
            valid_losses.append(loss.item())

            predicted = output_classification.max(dim=1)[1]
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

    ###################
    # plot acc & loss #
    ###################
    # Loss and Accuracy Curve
    interval = 50 if len(train_loss_plot) > 500 else 100
    plt.clf()
    plt.close()
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(train_loss_plot)), train_loss_plot, label="train_loss")
    plt.plot(np.arange(len(valid_loss_plot)), valid_loss_plot, label="val_loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.xticks(np.arange(0, len(valid_loss_plot), interval))
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(train_acc_plot)), train_acc_plot, label="train_acc")
    plt.plot(np.arange(len(valid_acc_plot)), valid_acc_plot, label="valid_acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(0, len(valid_acc_plot), interval))
    plt.legend()

    plt.savefig(plot_path + '.pdf')

    if early_stopping.early_stop:
        print("Early stopping")
        break

