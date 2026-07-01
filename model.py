import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F #dropout这个操作就从这个库调入

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.Relu = nn.ReLU()
        self.c1 = nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=0)
        self.s2=nn.MaxPool2d(kernel_size=3, stride=2)
        self.c3 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.s4=nn.MaxPool2d(kernel_size=3, stride=2)
        self.c5 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.c6=nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.s8=nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self.Relu(self.c1(x))
        x = self.s2(x)
        x = self.Relu(self.c3(x))
        x = self.s4(x)
        x = self.Relu(self.c5(x))
        x = self.Relu(self.c6(x))
        x = self.Relu(self.c7(x))
        x = self.s8(x)
        x = self.flatten(x)
        x = self.Relu(self.fc1(x))
        x=F.dropout(x, p=0.5)
        x = self.fc2(x)
        x=F.dropout(x, p=0.5)
        x = self.fc3(x)
        return x

# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     #模型实例化，放到设备中
#     model = AlexNet().to(device)
#     print(summary(model, input_size=(1, 227, 227)))


