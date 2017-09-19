import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
test_set = datasets.MNIST(root='../', train=False, transform=trans, download=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=1,
                shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 1000)
        self.l2 = nn.Linear(1000, 1000)
        self.l3 = nn.Linear(1000, 10)

    def forward(self, x):
        x = x.view((-1, 784))
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

model = Net()

model.eval()
for data, target in tqdm.tqdm(test_loader):
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
