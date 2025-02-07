#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 假设我们有一些股票日线数据
# 这里我们生成一些随机数据作为示例
num_samples = 1000
input_dim = 5  # 例如：开盘价、收盘价、最高价、最低价、成交量
output_dim = 1  # 预测明天的收盘价

# 生成随机数据
X = np.random.randn(num_samples, input_dim).astype(np.float32)
y = np.random.randn(num_samples, output_dim).astype(np.float32)

# 转换为Tensor
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# 创建DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义一个简单的线性模型
class StockPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StockPredictor, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

model = StockPredictor(input_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 对抗训练参数
epsilon = 0.01  # 对抗扰动的强度

# 对抗训练函数
def fgsm_attack(image, epsilon, data_grad):
    # 这里的 sign_data_grad 是输入数据梯度的符号（即每个元素的正负号）。乘以 sign_data_grad 的原因如下：
    # 梯度方向：梯度表示损失函数变化最快的方向。通过沿着梯度方向添加扰动，可以使损失函数最大化，从而生成对抗样本。
    # 符号函数：sign_data_grad 是梯度的符号函数，它只取梯度值的正负号（+1 或 -1）。这样做的好处是：
    # 简化计算：符号函数简化了梯度的计算，因为只需要知道梯度的方向，而不需要具体的梯度值。
    # 控制扰动大小：通过乘以 epsilon，可以控制扰动的大小。epsilon 是一个很小的正数，用于限制扰动的强度，避免扰动过大导致生成的对抗样本与原始样本差异过大。
    # 最大化损失：通过沿着梯度的符号方向添加扰动，可以最大化损失函数。这是因为梯度的符号方向表示损失函数增加最快的方向。
    # 总结来说，乘以 sign_data_grad 是为了确保扰动沿着梯度的方向添加，从而使损失函数最大化，生成有效的对抗样本。

    # 计算梯度的符号（即每个元素的正负号）
    sign_data_grad = data_grad.sign()
    # 通过在原始输入数据上添加一个小的扰动（由 epsilon 和梯度符号决定）来生成对抗样本。
    perturbed_image = image + epsilon * sign_data_grad
    return perturbed_image

# 训练模型
for epoch in range(1):  # 假设训练10个epoch
    for data, target in dataloader:
        model.train()
        data.requires_grad = True

        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        model.zero_grad()
        loss.backward()
        #  获取输入数据的梯度。
        data_grad = data.grad.data

        # 对抗样本
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # 对抗训练
        output = model(perturbed_data)
        adv_loss = criterion(output, target)

        # 反向传播对抗损失
        optimizer.zero_grad()
        adv_loss.backward()
        optimizer.step()

        break


    print(f'Epoch {epoch+1}, Loss: {loss.item()}, Adv Loss: {adv_loss.item()}')

print("训练完成")