from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitNet(nn.Module):
    def __init__(self):
        super(DigitNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1), F.log_softmax(x, dim=1)

class DeepEquationNet(nn.Module):
    def __init__(self):
        super(DeepEquationNet, self).__init__()
        # Handwritten digit NN layers
        self.digit_net = DigitNet()

        # Deep equation NN layers
        self.embed = nn.Embedding(4, 256)
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(10, 128)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 96)

    def forward(self, x_a, x_b, o):
        # Digit
        x_a, output_a = self.digit_net(x_a)
        x_b, output_b = self.digit_net(x_b)

        # Equation
        x_a = self.fc1(x_a)
        x_a = F.relu(x_a)
        x_b = self.fc2(x_b)
        x_b = F.relu(x_b)
        x = torch.cat((x_a, x_b), 1)
        o = self.embed(o).squeeze(1)
        x = torch.mul(x, o)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        output_eq = F.log_softmax(x, dim=1)
        return output_a, output_b, output_eq

