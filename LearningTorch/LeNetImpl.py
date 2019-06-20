# Implementation of a sample CNN
# Inspired by, and heavily referencing, the tutorial here:
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import threading


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def target(self, val):
        # goal: if a 'pixel' is negative, classify -1, else 1
        # [print(v.detach().numpy()[0], end='') for v in val]
        # print()
        return torch.tensor([[-1 if v.detach().numpy() < 0 else 1 for v in val[0]]], dtype=torch.float)


stop = False


def stop_thread():
    global stop
    input("Press enter to exit\n")
    stop = True


if __name__ == "__main__":
    net = Net()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=.01)

    loss_list = []

    input("Press any key to start")
    print("starting")
    threading.Thread(target=stop_thread).start()
    count = 0
    try:
        while not stop:
            optimizer.zero_grad()
            output = net(torch.randn(1, 1, 32, 32))  # Feed random input.
            # print(f"output: {output}")
            # also, apparently '__call__' is a thing in python

            target = net.target(output)  # calculate loss from given target function.
            # print(f"target: {target}")
            loss = criterion(output, target)

            loss_list.append(loss.detach().numpy())  # track all losses for eval at end

            loss.backward()
            optimizer.step()  # does the update, apparently.

            count += 1
            if count % 1000 == 0:
                print(count / 1000, end='')

    except KeyboardInterrupt:
        pass

    # print(loss_list)
    [print(v) for v in loss_list]
