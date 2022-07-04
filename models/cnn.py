import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class basic_cnn(nn.Module):

    def __init__(self):
        super(basic_cnn, self).__init__()

        self.conv1 = nn.Conv2d(6, 16, kernel_size=5)
        nn.init.normal_(self.conv1.weight,std = 0.01)
        self.conv2 =  nn.Conv2d(16, 64, kernel_size=5)
        nn.init.normal_(self.conv2.weight,std = 0.01)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        nn.init.normal_(self.conv3.weight,std = 0.01)
        self.pool = nn.MaxPool2d(2,2)


        self.fc1 = nn.Linear(128 * 9 * 9, 640)
        self.fc2 = nn.Linear(640, 2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(x))

        x = self.conv2(x)
        x = self.pool(F.relu(x))

        x = self.conv3(x)
        x = F.relu(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc2(x)
        return x


class _cnn(nn.Module):

    def __init__(self):
        super(_cnn, self).__init__()

        self.conv1 = nn.Conv2d(6, 16, kernel_size=5)
        nn.init.normal_(self.conv1.weight,std = 0.01)
        self.conv2 =  nn.Conv2d(16, 64, kernel_size=5)
        nn.init.normal_(self.conv2.weight,std = 0.01)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        nn.init.normal_(self.conv3.weight,std = 0.01)

        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128 * 9 * 9, 640)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class cnn_points(nn.Module):

    def __init__(self):
        super(cnn_points, self).__init__()

        self.f1 = _cnn()
        self.f2 = _cnn()
        self.f3 = _cnn()
        self.f4 = _cnn()
        self.f5 = _cnn()
        self.f6 = _cnn()
        self.f7 = _cnn()
        self.f8 = _cnn()
        self.f9 = _cnn()
        self.f10= _cnn()

        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(640*10, 2)

    def forward(self, x1,x2,x3,x4,x5,x6,x7,x8,x9,x10):
        f1 = self.f1(x1)
        f2 = self.f2(x2)
        f3 = self.f3(x3)
        f4 = self.f4(x4)
        f5 = self.f5(x5)
        f6 = self.f6(x6)
        f7 = self.f7(x7)
        f8 = self.f8(x8)
        f9 = self.f9(x9)
        f10 = self.f10(x10)
        x = torch.cat((f1,f2,f3,f4,f5,f6,f7,f8,f9,f10),dim=1)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x