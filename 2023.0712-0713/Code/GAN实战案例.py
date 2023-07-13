import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)
np.random.seed(1)

LR_G = 0.0001
LR_D = 0.0001
BATCH_SIZE = 64

N_IDEAS = 5  # 输入的噪声维度，可以自己设定（经过神经网络后会把维度调整）

ART_COMPONETS = 15  # 噪声输入后的输出维度
PAINT_POINTS = np.stack([np.linspace(-1, 1, ART_COMPONETS) for _ in range(BATCH_SIZE)], 0)  # 我们原始数据的x坐标-1~1均匀分布


def artist_work():
    a = np.ones((BATCH_SIZE, 1)) * 2
    paints = a * np.power(PAINT_POINTS, 2) + (a - 1)  # y = 2x^2 + 1
    paints = torch.from_numpy(paints).float()
    return paints


# 网络结构
G = nn.Sequential(
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONETS)
)
D = nn.Sequential(
    nn.Linear(ART_COMPONETS, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

# 优化器与损失函数
optimizer_G = torch.optim.Adam(G.parameters(), lr=LR_G)
optimizer_D = torch.optim.Adam(D.parameters(), lr=LR_D)
Criterion = torch.nn.BCELoss()

# 开始训练
plt.ion()
G_losses = []  # 储存了损失方便自己画图可视化
D_losses = []

for step in range(10000):
    artist_painting = artist_work()
    G_idea = torch.randn(BATCH_SIZE, N_IDEAS)

    G_paintings = G(G_idea)

    pro_atrist0 = D(artist_painting)
    pro_atrist1 = D(G_paintings)

    G_loss = -1 / torch.mean(torch.log(1. - pro_atrist1))
    G_losses.append(G_loss.item())
    D_loss = Criterion(pro_atrist0, torch.ones_like(pro_atrist0)) + Criterion(pro_atrist1,
                                                                              torch.zeros_like(pro_atrist1))
    D_losses.append(D_loss.item())
    optimizer_G.zero_grad()
    G_loss.backward(retain_graph=True)  # 因为D的反向传播需要用到G，所以设置为True

    optimizer_D.zero_grad()
    D_loss.backward()
    optimizer_G.step()
    optimizer_D.step()

    if step % 200 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='original data')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % pro_atrist0.data.numpy().mean(),
                 fontdict={'size': 13})
        # plt.text(-.5, 2, 'G_loss= %.2f ' % G_loss.data.numpy(), fontdict={'size': 13})

        plt.ylim((0, 3));
        plt.legend(loc='upper right', fontsize=10);
        plt.draw();
        plt.pause(0.1)

print('训练结束')
plt.ioff()
plt.show()