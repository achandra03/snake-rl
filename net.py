import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.input = nn.Linear(400, 500, bias = True)
        self.hidden1 = nn.Linear(500, 500, bias = True)
        self.hidden2 = nn.Linear(500, 300, bias = True)
        self.output = nn.Linear(300, 5, bias = True)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.softmax(self.output(x))
        return x

    def set_weights(self, net):
        net.input.weight.data = self.input.weight.data
        net.hidden1.weight.data = self.hidden1.weight.data
        net.hidden2.weight.data = self.hidden2.weight.data
        net.output.weight.data = self.output.weight.data



        