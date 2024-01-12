# 导入所需的库
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
import shutil
import time
# from torchvision.transforms import functional as F
from PIL import ImageFilter
from torchsummary import summary

#Super parameters
train_batch_size = 4
num_epochs = 30
RandomRotation = True
RandomColorJitter = True
scale = 150

class EdgeDetectionTransform:
    def __init__(self):
        self.sobel = ImageFilter.FIND_EDGES

    def __call__(self, img):
        return img.filter(self.sobel)

# 定义预处理
transform = transforms.Compose([
    transforms.RandomRotation(30),
    # EdgeDetectionTransform(),
    # transforms.RandomResizedCrop(150),
    # transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机改变亮度和对比度
    transforms.Resize(scale),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# # 加载数据集
# train_data = datasets.ImageFolder("./rps/rps", transform=transform)
# train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)

# for inputs, targets in train_loader:
#     print(inputs.shape)  # 打印输入数据的形状
#     print(targets.shape)  # 打印目标数据的形状

# 定义卷积神经网络模型
class Net(nn.Module):
    def __init__(self,device):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool1 = nn.MaxPool2d(4, 4)  # change pool size to 4x4
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(4, 4)  # change pool size to 4x4
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.pool3 = nn.MaxPool2d(4, 4)
        # self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = None
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)
        self.device = device

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.shape)
        x = self.pool3(F.relu(self.conv3(x)))
        # print(x.shape)
        x = x.view(x.size(0),-1)
        # print(x.shape)
        # x = F.relu(self.fc1(x))
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 256).to(self.device)  # dynamically define fc1
        # x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc1(x)))
        # print(x.shape)
        x = self.fc2(x)
        x = self.fc3(x)
        # print(x.shape)
        return x

# model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = Net(device).to(device)

# 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
summary(model, input_size=(3, 150, 150))