import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from keras.preprocessing.image import ImageDataGenerator
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

# 设置超参数
batch_size = 32
image_height = 600
image_width = 800
Lr = 0.001
num_epochs = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 定义转换操作
transform = transforms.ToTensor()

# 构造数据集
class CreateDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y


# 构建模型
class ConvolutionalNN(nn.Module):
    def __init__(self):
        super(ConvolutionalNN, self).__init__()

        self.model1 = nn.Sequential(
            # 池化层
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 卷积层
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True),
            # 激活函数
            nn.ReLU(),
            # 池化层
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 卷积层
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True),
            # 激活函数
            nn.ReLU(),
            # 池化层
            nn.MaxPool2d(kernel_size=5, stride=5),
            # 卷积层
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            # 激活函数
            nn.ReLU(),
            # 池化层
            nn.MaxPool2d(kernel_size=5, stride=5),
        )

        self.model2 = nn.Sequential(
            # 全连接层
            nn.Linear(32 * 6 * 8, 1024),
            # 激活函数
            nn.ReLU(),
            # 全连接层
            nn.Linear(1024, 256),
            # 激活函数
            nn.ReLU(),
            # 全连接层
            nn.Linear(256, 32),
            # 激活函数
            nn.ReLU(),
            # 全连接层
            nn.Linear(32, 5),
        )

        def forward(self, x):
            x = self.model1(x)
            x = x.view(-1, 32 * 6 * 8)
            x = self.model2(x)
            return x

# 创建实例
CNN = ConvolutionalNN()
CNN = CNN.to(device)
# 优化器和损失函数
optimizer_CNN = torch.optim.Adam(CNN.parameters(), lr=Lr)
Criterion = nn.CrossEntropyLoss()
Criterion = Criterion.to(device)

# 读取数据（阉割版）
train_data = []
train_labels = []
train_dir = '/kaggle/input/cassava-leaf-disease-classification/train_images/'
id_labels = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

label_counts = [0, 0, 0, 0, 0]  # 记录每个标签的数量
max_images_per_label = 500
for img_id, img_label in zip(id_labels['image_id'], id_labels['label']):
    if label_counts[img_label] < max_images_per_label:
        img = Image.open(os.path.join(train_dir, img_id))
        train_data.append(img)
        train_labels.append(img_label)
        label_counts[img_label] += 1

# 开始训练

# 训练过程
for epoch in range(num_epochs):
    for i in range(0, len(train_data), batch_size):
        Image_torch = []
        real_labels = []
        outputs_list = []
        for x in range(batch_size):
            Image_torch.append(transform(train_data[i + x]))
        dataset = CreateDataset(Image_torch, train_labels[i:i + batch_size])
        for batch_data, batch_labels in dataset:
            batch_data = batch_data.to(device)
            real_labels.append(torch.tensor(batch_labels).to(device))

            outputs_list.append(CNN(batch_data))

        print(real_labels)
        real_labels_tensor = torch.tensor(real_labels)
        print(real_labels_tensor.size())
        print(real_labels_tensor)

        print(outputs_list)
        outputs_list_tensor = torch.tensor(outputs_list[0])
        print(outputs_list_tensor)
        print(outputs_list_tensor.size())
        CNN_loss = Criterion(outputs_list_tensor, real_labels_tensor)

        optimizer_CNN.zero_grad()

        CNN_loss.backward()
        optimizer_CNN.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, CNN_loss.item()))


# 开始测试

# 读取数据
test_labels = []
file_list = os.listdir('/kaggle/input/cassava-leaf-disease-classification/test_images')
test_dir = '/kaggle/input/cassava-leaf-disease-classification/test_images'
for img_id in file_list:
    test_img = Image.open(os.path.join(test_dir, img_id))
    test_img = transform(test_img)
    test_img = test_img.to(device)
    test_outputs = CNN(test_img)
    test_labels.append(test_outputs)

df = pd.DataFrame({'image_id':file_list,'label':test_labels})
df.to_csv("/kaggle/working/submission.csv", index = False)