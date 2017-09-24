import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class VGG(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1=L.Convolution2D(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2=L.Convolution2D(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_1=L.Convolution2D(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2=L.Convolution2D(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_1=L.Convolution2D(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2=L.Convolution2D(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3=L.Convolution2D(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_1=L.Convolution2D(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2=L.Convolution2D(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3=L.Convolution2D(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_1=L.Convolution2D(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2=L.Convolution2D(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3=L.Convolution2D(512, 512, kernel_size=3, stride=1, padding=1)
        self.fc6=L.Linear(25088, 4096)
        self.fc7=L.Linear(4096, 4096)
        self.fc8=L.Linear(4096, 1000)

    def forward(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), train=self.train, ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train, ratio=0.5)
        h = self.fc8(h)

        return h

model = Net()
model.eval()

data = np.zeros([1, 3, 224, 224], np.float32)
data = torch.from_numpy(data)

output = model(data)
