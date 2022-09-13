import numpy as np
import os
from torch import nn
import torch
from torch.nn import functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path :
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, epoch, val_acc, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, val_acc, val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}. \n Best loss: {self.val_loss_min:.6f}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, val_acc, val_loss, model)
            self.counter = 0

    def save_checkpoint(self, epoch, val_acc, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = self.save_path + '_best.pt'
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


class EarlyStoppingAcc:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path :
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -1.0
        self.delta = delta

    def __call__(self, epoch, val_acc, val_loss, model):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, val_acc, val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}. \n Best accuracy: {self.val_acc_max:.6f}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, val_acc, val_loss, model)
            self.counter = 0

    def save_checkpoint(self, epoch, val_acc, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
        path = self.save_path + '_best.pt'
        torch.save(model.state_dict(), path)
        self.val_acc_max = val_acc



class FocalLossNoAlpha(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, alpha=1, eps=1e-7):
        super(FocalLossNoAlpha, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        logp = self.ce(input, target)
        target = target.type(torch.long)
        # at = self.alpha.gather(0, target.data.view(-1))
        p = torch.exp(-logp)
        # loss = at * (1 - p) ** self.gamma * logp
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        # logpt = F.log_softmax(input)
        logpt = F.log_softmax(input, -1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)


        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
