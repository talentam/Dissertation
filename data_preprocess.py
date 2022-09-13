import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
import matplotlib.pyplot as plt

import numpy as np
import random
from PIL import Image
import PIL.ImageOps

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

# data_path = "./dataset/dataset_clean.csv"
# df = pd.read_csv(data_path)
# for index, row in df.iterrows():
#     pixel = row

class Oversample():
    def __init__(self):
        pass


# Dataset
class BreastCancerDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, model="GoogLeNet"):
        self.imageFolderDataset = pd.read_csv(imageFolderDataset)
        self.transform = transform
        self.model = model

    def __getitem__(self, index):
        row = self.imageFolderDataset.iloc[index]
        L0, L1, L2, L3, L4 = row["L0"], row["L1"], row["L2"], row["L3"], row["L4"]
        L5, L6, L7, L8, L9 = row["L5"], row["L6"], row["L7"], row["L8"], row["L9"]

        LS0, LS1, LS2, LS3, LS4 = row["Skin L0"], row["Skin L1"], row["Skin L2"], row["Skin L3"], row["Skin L4"]
        LS5, LS6, LS7, LS8, LS9 = row["Skin L5"], row["Skin L6"], row["Skin L7"], row["Skin L8"], row["Skin L9"]

        R0, R1, R2, R3, R4 = row["R0"], row["R1"], row["R2"], row["R3"], row["R4"]
        R5, R6, R7, R8, R9 = row["R5"], row["R6"], row["R7"], row["R8"], row["R9"]

        RS0, RS1, RS2, RS3, RS4 = row["Skin R0"], row["Skin R1"], row["Skin R2"], row["Skin R3"], row["Skin R4"]
        RS5, RS6, RS7, RS8, RS9 = row["Skin R5"], row["Skin R6"], row["Skin R7"], row["Skin R8"], row["Skin R9"]

        LT0, LTS0 = row["T0"], row["Skin T0"]
        RT1, RTS1 = row["T1"], row["Skin T1"]

        if self.model == "GoogLeNet":
            # image format
            left = torch.tensor([[[L8, (L8+L1)/2, L1, (L1+L2)/2, L2],
                                  [(L7+L8)/2, (L0+L1+L7+L8)/4, (L0+L1)/2, (L0+L1+L2+L3)/4, (L2+L3)/2],
                                  [L7, (L0+L7)/2, L0, (L0+L3)/2, L3],
                                  [(L6+L7)/2, (L0+L5+L6+L7)/4, (L0+L5)/2, (L0+L3+L4+L5)/4, (L3+L4)/2],
                                  [L6, (L5+L6)/2, L5, (L4+L5)/2, L4]],
                                 [[LS8, (LS8 + LS1) / 2, LS1, (LS1 + LS2) / 2, LS2],
                                  [(LS7 + LS8) / 2, (LS0 + LS1 + LS7 + LS8) / 4, (LS0 + LS1) / 2, (LS0 + LS1 + LS2 + LS3) / 4, (LS2 + LS3) / 2],
                                  [LS7, (LS0 + LS7) / 2, LS0, (LS0 + LS3) / 2, LS3],
                                  [(LS6 + LS7) / 2, (LS0 + LS5 + LS6 + LS7) / 4, (LS0 + LS5) / 2, (LS0 + LS3 + LS4 + LS5) / 4, (LS3 + LS4) / 2],
                                  [LS6, (LS5 + LS6) / 2, LS5, (LS4 + LS5) / 2, LS4]]])

            right = torch.tensor([[[R8, (R8 + R1) / 2, R1, (R1 + R2) / 2, R2],
                                  [(R7 + R8) / 2, (R0 + R1 + R7 + R8) / 4, (R0 + R1) / 2, (R0 + R1 + R2 + R3) / 4, (R2 + R3) / 2],
                                  [R7, (R0 + R7) / 2, R0, (R0 + R3) / 2, R3],
                                  [(R6 + R7) / 2, (R0 + R5 + R6 + R7) / 4, (R0 + R5) / 2, (R0 + R3 + R4 + R5) / 4, (R3 + R4) / 2],
                                  [R6, (R5 + R6) / 2, R5, (R4 + R5) / 2, R4]],
                                 [[RS8, (RS8 + RS1) / 2, RS1, (RS1 + RS2) / 2, RS2],
                                  [(RS7 + RS8) / 2, (RS0 + RS1 + RS7 + RS8) / 4, (RS0 + RS1) / 2, (RS0 + RS1 + RS2 + RS3) / 4, (RS2 + RS3) / 2],
                                  [RS7, (RS0 + RS7) / 2, RS0, (RS0 + RS3) / 2, RS3],
                                  [(RS6 + RS7) / 2, (RS0 + RS5 + RS6 + RS7) / 4, (RS0 + RS5) / 2, (RS0 + RS3 + RS4 + RS5) / 4, (RS3 + RS4) / 2],
                                  [RS6, (RS5 + RS6) / 2, RS5, (RS4 + RS5) / 2, RS4]]])
        elif self.model == "FNN":
            # feature vector format
            left = torch.tensor([L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, LT0,
                                 LS0, LS1, LS2, LS3, LS4, LS5, LS6, LS7, LS8, LS9, LTS0])

            right = torch.tensor([R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, RT1,
                                 RS0, RS1, RS2, RS3, RS4, RS5, RS6, RS7, RS8, RS9, RTS1])

        label = int(row["Cancer"])
        return left, right, label

    def __len__(self):
        return self.imageFolderDataset.shape[0]


