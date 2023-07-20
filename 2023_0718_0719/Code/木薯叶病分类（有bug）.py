import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from keras.preprocessing.image import ImageDataGenerator

#cuda = True if torch.cuda.is_available() else False

# 设置超参数
batch_size = 128
image_height = 600
image_width = 800
lr = 0.0002
num_epochs = 20

# 构建模型
class ConvolutionalNN(nn.Module):
    def __init__(self):
        super(ConvolutionalNN, self).__init__()

        self.model = nn.Sequential(
            # 卷积层
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            # 激活函数
            nn.ReLU(True),
            # 池化层
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 卷积层
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # 激活函数
            nn.ReLU(True),
            # 池化层
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 展平
            nn.Flatten(),

            # 全连接层
            nn.Linear(batch_size * image_width * image_height, 128),
            # 激活函数
            nn.ReLU(True),
            # 全连接层
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 创建实例
CNN = ConvolutionalNN()

# 优化器和损失函数
optimizer_CNN = torch.optim.Adam(CNN.parameters(), lr=lr)
Criterion = nn.CrossEntropyLoss()

'''
# 读取数据
train_data = []
train_labels = []
train_dir = '/kaggle/input/cassava-leaf-disease-classification/train_images/'
id_labels = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")
for img_id, img_label in zip(id_labels['image_id'], id_labels['label']):
    img = plt.imread(os.path.join(train_dir, img_id))
    train_data.append(img)
    train_labels.append(img_label)
'''



# 读取数据（阉割版）
label_counts = [0, 0, 0, 0, 0]  # 记录每个标签的数量
max_images_per_label = 1000

for img_id, img_label in zip(id_labels['image_id'], id_labels['label']):
    if label_counts[img_label] < max_images_per_label:
        img = plt.imread(os.path.join(train_dir, img_id))
        train_data.append(img)
        train_labels.append(img_label)
        label_counts[img_label] += 1


# 开始训练

# 转换数据类型
train_data = torch.tensor(train_data)
train_labels = torch.tensor(train_labels)

# 训练过程
for epoch in range(num_epochs):
    for i in range(0, len(train_data), batch_size):
        batch_data = train_data[i:i + batch_size]
        batch_labels = train_labels[i:i + batch_size]

        outputs = CNN(batch_data)

        CNN_loss = Criterion(outputs, batch_labels)

        optimizer_CNN.zero_grad()

        CNN_loss.backward()
        optimizer_CNN.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))

# 开始测试

# 读取数据
test_labels = []
file_list = os.listdir('/kaggle/input/cassava-leaf-disease-classification/test_images')
test_dir = '/kaggle/input/cassava-leaf-disease-classification/test_images'
for img_id in file_list:
    test_img = plt.imread(os.path.join(test_dir, img_id))
    test_img = torch.tensor(test_img)
    test_outputs = CNN(test_img)
    test_labels.append(test_outputs)

df = pd.DataFrame({'image_id':file_list,'label':test_labels})
df.to_csv("/kaggle/working/submission.csv", index = False)