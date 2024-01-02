# 导入所需的库
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import shutil
import time
# from torchvision.transforms import functional as F
from PIL import ImageFilter

#Super parameters
train_batch_size = 4
num_epochs = 30
RandomRotation = False
RandomColorJitter = False
scale = 150

class EdgeDetectionTransform:
    def __init__(self):
        self.sobel = ImageFilter.FIND_EDGES

    def __call__(self, img):
        return img.filter(self.sobel)
# # 删除目录
# if os.path.exists("./runs/experiment_1"):
#     shutil.rmtree("./runs/experiment_1")

# # 创建一个 SummaryWriter 对象
# writer = SummaryWriter('runs/experiment_1')

# 获取 runs 目录下的子目录列表
subdirs = [d for d in os.listdir('./runs') if os.path.isdir(os.path.join('./runs', d))]

# 获取子目录的数量
num_dirs = len(subdirs)

# 创建一个新的 SummaryWriter 对象，其名称基于子目录的数量
writer = SummaryWriter(f'runs/experiment_{num_dirs + 1}')

# 定义预处理
transform = transforms.Compose([
    transforms.RandomRotation(30),
    EdgeDetectionTransform(),
    # transforms.RandomResizedCrop(150),
    # transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机改变亮度和对比度
    transforms.Resize(scale),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
train_data = datasets.ImageFolder("./rps/rps", transform=transform)
train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)

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
        x = F.relu(self.fc1(x))
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
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
writer.add_text('Training parameters', f'Batch size: {train_batch_size}, Epochs: {num_epochs}, RandomColorJitter: {RandomColorJitter}')
for epoch in range(num_epochs):  # loop over the dataset multiple times
    start_time = time.time()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)
        # print(inputs.shape)
        # print(labels.shape)
        # print(labels)
        # model_device = next(model.parameters()).device
        # print(model_device)
        # print(inputs.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    # 记录每个 epoch 的平均损失
    average_loss = running_loss / len(train_loader)
    writer.add_scalar('training loss', average_loss, epoch)
    end_time = time.time()
    print('epoch: {}, loss: {}'.format(epoch, average_loss))
    print('epoch time: {}'.format(end_time - start_time))

# 评估模型
# 注意：你需要提供一个测试数据集
path_to_your_test_data = "./rps-test-set/rps-test-set"
test_data = datasets.ImageFolder(path_to_your_test_data, transform=transform)
test_loader = DataLoader(test_data, batch_size=4, shuffle=True)

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        # images, labels = data
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
writer.add_text('Accuracy', f'Accuracy of the network on the test images: {100 * correct / total} %')