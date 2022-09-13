import torch
import torch.nn as nn
import torch.nn.functional as F


# model structure of adjusted FNN
class SiameseDenseNetwork(nn.Module):

    def __init__(self, input_features, units=128, embedding_size=128, num_layers=6,
                 dropout_rate=0.1, use_batchnorm=False):
        super(SiameseDenseNetwork, self).__init__()

        self.linear = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        self.use_batchnorm = use_batchnorm
        self.num_layers = num_layers
        for i in range(num_layers):
            if i == 0:
                input_size = input_features
                output_size = units
            elif i == num_layers - 1:
                input_size = units
                output_size = embedding_size
            else:
                input_size = units
                output_size = units

            self.linear.append(nn.Linear(input_size, output_size))

            if self.use_batchnorm:
                self.batchnorm.append(nn.BatchNorm1d(output_size))

        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.SiLU()
        self.score = DenseLayer(1)


    def siamese_branch(self, x):
        x = x.float()

        for i in range(len(self.linear)):
            identity = x
            x = self.activation(self.linear[i](x))

            if self.use_batchnorm:
                x = self.batchnorm[i](x)

            if i != len(self.linear) - 1:
                x = self.dropout(x)

            if self.num_layers-1 > i > 0:
                x += identity
                identity = x

        return x

    def forward(self, input1, input2):
        output1 = self.siamese_branch(input1)
        output2 = self.siamese_branch(input2)
        euclidean_distance = torch.unsqueeze(F.pairwise_distance(output1, output2), 1)
        output_classification = self.score(euclidean_distance)

        return torch.unsqueeze(output1, 1), torch.unsqueeze(output2, 1), output_classification


class DenseLayer(nn.Module):
    def __init__(self, in_channels):
        super(DenseLayer, self).__init__()
        self.linear1 = nn.Linear(in_channels, 2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        return x
