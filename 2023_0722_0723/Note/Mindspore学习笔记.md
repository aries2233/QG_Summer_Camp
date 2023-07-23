# Mindspore学习笔记

## 张量 Tensor

```python
import numpy as np
import mindspore
from mindspore import ops
from mindspore import Tensor, CSRTensor, COOTensor

# 创建张量
x_tensor = Tensor(x)

# 继承张量
x_ones = ops.ones_like(x_data)
x_zeros = ops.zeros_like(x_data)
'''
Ones Tensor:
 [1 1 1 1]
Zeros Tensor:
 [0 0 0 0]
 '''

# 张量属性
'''
形状（shape）：Tensor的shape，是一个tuple。

数据类型（dtype）：Tensor的dtype，是MindSpore的一个数据类型。

单个元素大小（itemsize）： Tensor中每一个元素占用字节数，是一个整数。

占用字节数量（nbytes）： Tensor占用的总字节数，是一个整数。

维数（ndim）： Tensor的秩，也就是len(tensor.shape)，是一个整数。

元素个数（size）： Tensor中所有元素的个数，是一个整数。

每一维步长（strides）： Tensor每一维所需要的字节数，是一个tuple。

'''

# 张量索引
tensor = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))

print("First row: {}".format(tensor[0]))
print("value of bottom right corner: {}".format(tensor[1, 1]))
print("Last column: {}".format(tensor[:, -1]))
print("First column: {}".format(tensor[..., 0]))
'''
First row: [0. 1.]
value of bottom right corner: 3.0
Last column: [1. 3.]
First column: [0. 2.]
'''

# 张量运算
# Concat 连接
data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
output = ops.concat((data1, data2), axis=0)
'''
[[0. 1.]
 [2. 3.]
 [4. 5.]
 [6. 7.]]
shape:
 (4, 2)
 '''

#Stack 合并
data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
output = ops.stack([data1, data2])
'''
[[[0. 1.]
  [2. 3.]]

 [[4. 5.]
  [6. 7.]]]
shape:
 (2, 2, 2)
'''

# 稀疏张量？？？
```

## 数据集 Dataset

### mindspore.dataset

注：提供的接口**仅支持解压后的数据文件**

```python
import numpy as np
from mindspore.dataset import vision
from mindspore.dataset import MnistDataset, GeneratorDataset
import matplotlib.pyplot as plt

# create_tuple_iterator或create_dict_iterator接口创建数据迭代器，迭代访问数据。
# 访问的数据类型默认为Tensor；若设置output_numpy=True，访问的数据类型为Numpy。

# 常用操作

# shuffle随机排序（buffer_size为缓冲区大小）
train_dataset = train_dataset.shuffle(buffer_size=64)

# map
'''
map操作是数据预处理的关键操作，可以针对数据集指定列（column）添加数据变换（Transforms），将数据变换应用于该列数据的每个元素，并返回包含变换后元素的新数据集。
'''
train_dataset = train_dataset.map(vision.Rescale(1.0 / 255.0, 0), input_columns='image')
# (28, 28, 1) Float32

# batch 分批
train_dataset = train_dataset.batch(batch_size=32)
# batch后的数据增加一维，大小为batch_size。
# (32, 28, 28, 1) Float32

GeneratorDataset()
#主要用途是将生成器函数转换为 TensorFlow 数据集对象，以便在模型训练过程中使用。生成器函数可以用来生成数据样本，这样可以动态地生成数据，而不是一次性加载整个数据集到内存中?
```

## 数据变换 Transforms

```python
import numpy as np
from PIL import Image
from download import download
from mindspore.dataset import transforms, vision, text
from mindspore.dataset import GeneratorDataset, MnistDataset

# Rescale
'''
Rescale变换用于调整图像像素值的大小，包括两个参数：
rescale：缩放因子。
shift：平移因子。
图像的每个像素将根据这两个参数进行调整，输出的像素值为outputi=inputi∗rescale+shift。
'''
random_np = np.random.randint(0, 255, (48, 48), np.uint8)
random_image = Image.fromarray(random_np)

# Normalize
'''
Normalize变换用于对输入图像的归一化，包括三个参数：
mean：图像每个通道的均值。
std：图像每个通道的标准差。
is_hwc：输入图像格式为(height, width, channel)还是(channel, height, width)。
图像的每个通道将根据mean和std进行调整，计算公式为output=(input−mean)/std，其中 c代表通道索引
'''
normalize = vision.Normalize(mean=(0.1307,), std=(0.3081,))
normalized_image = normalize(rescaled_image)

# HWC2CHW
'''
HWC2CHW变换用于转换图像格式。在不同的硬件设备中可能会对(height, width, channel)或(channel, height, width)两种不同格式有针对性优化。
'''
hwc_image = np.expand_dims(normalized_image, -1)# 在末尾插入一个维度
hwc2chw = vision.HWC2CHW()
chw_image = hwc2chw(hwc_image)
'''
arr = np.array([1, 2, 3]) # 一维数组
expanded_arr = np.expand_dims(arr, axis=0) 
# [[1 2 3]]
# expanded_arr.shape = (1, 3)
expanded_arr = np.expand_dims(arr, -1)
# [[1]
 [2]
 [3]]
 # expanded_arr.shape = (3, 1)
'''
```

### Text Transforms

```python
texts = ['Welcome to Beijing']
test_dataset = GeneratorDataset(texts, 'text')

# PythonTokenizer
# 分词（Tokenize）操作是文本数据的基础处理方法
def my_tokenizer(content):
    return content.split()

test_dataset = test_dataset.map(text.PythonTokenizer(my_tokenizer))
print(next(test_dataset.create_tuple_iterator()))
# [Tensor(shape=[3], dtype=String, value= ['Welcome', 'to', 'Beijing'])]

# Lookup
# Lookup为词表映射变换，用来将Token转换为Index。在使用Lookup前，需要构造词表，一般可以加载已有的词表，或使用Vocab生成词表。这里我们选择使用Vocab.from_dataset方法从数据集中生成词表。
vocab = text.Vocab.from_dataset(test_dataset)
# 获得词表后我们可以使用vocab方法查看词表。
print(vocab.vocab())
# {'to': 2, 'Beijing': 0, 'Welcome': 1}
#生成词表后，可以配合map方法进行词表映射变换，将Token转为Index。
test_dataset = test_dataset.map(text.Lookup(vocab))
print(next(test_dataset.create_tuple_iterator()))
# [Tensor(shape=[3], dtype=Int32, value= [1, 2, 0])]
```

## Lambda Transforms

Lambda函数是一种不需要名字、由一个单独表达式组成的匿名函数，表达式会在调用时被求值。Lambda Transforms可以加载任意定义的Lambda函数，提供足够的灵活度。

```python
test_dataset = GeneratorDataset([1, 2, 3], 'data', shuffle=False)

test_dataset = test_dataset.map(lambda x: x * 2)
print(list(test_dataset.create_tuple_iterator()))
# [[Tensor(shape=[], dtype=Int64, value= 2)], [Tensor(shape=[], dtype=Int64, value= 4)], [Tensor(shape=[], dtype=Int64, value= 6)]]

def func(x):
    return x * x + 2
test_dataset = test_dataset.map(lambda x: func(x))
print(list(test_dataset.create_tuple_iterator()))
# [[Tensor(shape=[], dtype=Int64, value= 6)], [Tensor(shape=[], dtype=Int64, value= 18)], [Tensor(shape=[], dtype=Int64, value= 38)]]
```

## 网络构建

神经网络模型是由神经网络层和Tensor操作构成的，`mindspore.nn`提供了常见神经网络层的实现，在MindSpore中，`Cell`类是构建所有网络的基类，也是网络的基本单元。一个神经网络模型表示为一个`Cell`，它由不同的子`Cell`构成。使用这样的嵌套结构，可以简单地使用面向对象编程的思维，对神经网络结构进行构建和管理。

```python
import mindspore
from mindspore import nn, ops

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits
    
model = Network()

X = ops.ones((1, 28, 28), mindspore.float32)
logits = model(X)
print(logits)

#Tensor(shape=[1, 10], dtype=Float32, value=[[-5.08734025e-04,  3.39190010e-04,  4.62840870e-03 ... -1.20305456e-03, -5.05689112e-03,  3.99264274e-03]])

pred_probab = nn.Softmax(axis=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

#Predicted class: [4]
```

## 自动微分

```python
神经网络梯度计算
import numpy as np
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import Tensor, Parameter

# Define model
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.w = w
        self.b = b

    def construct(self, x):
        z = ops.matmul(x, self.w) + self.b
        return z

# Instantiate model
model = Network()
# Instantiate loss function
loss_fn = nn.BCEWithLogitsLoss()

# Define forward function
def forward_fn(x, y):
    z = model(x)
    loss = loss_fn(z, y)
    return loss

#value_and_grad接口获得微分函数，用于计算梯度
grad_fn=mindspore.value_and_grad(forward_fn,None,weights=model.trainable_params())
#使用Cell封装神经网络模型，模型参数为Cell的内部属性，此时我们不需要使用grad_position指定对函数输入求导，因此将其配置为None。对模型参数求导时，我们使用weights参数，使用model.trainable_params()方法从Cell中取出可以求导的参数。

loss, grads = grad_fn(x, y)
print(grads)
```

### 训练模型

```python
import mindspore
from mindspore import nn
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset

def datapipe(path, batch_size):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(mindspore.int32)

    dataset = MnistDataset(path)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

train_dataset = datapipe('MNIST_Data/train', batch_size=64)
test_dataset = datapipe('MNIST_Data/test', batch_size=64)

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

model = Network()

# 超参数
epochs = 3
batch_size = 64
learning_rate = 1e-2

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
optimizer = nn.SGD(model.trainable_params(), learning_rate=learning_rate)

# Define forward function
def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits

# Get gradient function
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

# Define function of one-step training
def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss

def train_loop(model, dataset):
    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator(num_epochs=1)):
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")


def test_loop(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator(num_epochs=1):
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(f"Test: \n Accuracy:{(100*correct):>0.1f}%,Avgloss{test_loss:>8f}\n")
    
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(model, train_dataset)
    test_loop(model, test_dataset, loss_fn)
print("Done!")
```

### 保存与加载模型

```python
# 保存
model = network()
mindspore.save_checkpoint(model, "model.ckpt")

# 加载
model = network()
param_dict = mindspore.load_checkpoint("model.ckpt")
param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
# param_not_load是未被加载的参数列表，为空时代表所有参数均加载成功。

# 保存和加载MindIR？
```

