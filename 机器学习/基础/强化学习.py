#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# 超参数
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
MEMORY_SIZE = 10000

# 生成随机股票数据
def generate_stock_data(num_days=1000):
    return np.random.normal(0, 1, num_days)

# 定义环境
class StockTradingEnvironment:
    def __init__(self, data):
        self.data = data
        self.reset()

    def reset(self):
        self.index = 0
        self.position = 0
        self.profit = 0
        return self._get_state()

    def step(self, action):
        if action == 0:  # 买入
            self.position += 1
        elif action == 1:  # 卖出
            self.position -= 1
        reward = self._calculate_reward()
        self.index += 1
        done = self.index >= len(self.data) - 1
        return self._get_state(), reward, done

    def _get_state(self):
        return np.array([self.data[self.index], self.position])

    def _calculate_reward(self):
        if self.index + 1 < len(self.data):
            return self.position * (self.data[self.index + 1] - self.data[self.index])
        return 0

# 定义Q网络
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 定义经验回放
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 定义训练过程
def train(q_network, target_network, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))

    state_batch = torch.tensor(batch[0], dtype=torch.float32)
    action_batch = torch.tensor(batch[1], dtype=torch.long).unsqueeze(1)
    reward_batch = torch.tensor(batch[2], dtype=torch.float32)
    next_state_batch = torch.tensor(batch[3], dtype=torch.float32)
    done_batch = torch.tensor(batch[4], dtype=torch.float32)

    q_values = q_network(state_batch).gather(1, action_batch)
    next_q_values = target_network(next_state_batch).max(1)[0].detach()
    expected_q_values = reward_batch + (GAMMA * next_q_values * (1 - done_batch))

    loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 初始化
data = generate_stock_data()
env = StockTradingEnvironment(data)
q_network = QNetwork()
target_network = QNetwork()
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters())
memory = ReplayMemory(MEMORY_SIZE)

steps_done = 0

# 训练循环
for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
        if random.random() > eps_threshold:
            action = q_network(torch.tensor(state, dtype=torch.float32)).max(0)[1].item()
        else:
            action = random.randrange(2)

        next_state, reward, done = env.step(action)
        memory.push((state, action, reward, next_state, done))
        state = next_state

        train(q_network, target_network, memory, optimizer)
        steps_done += 1

    if episode % TARGET_UPDATE == 0:
        target_network.load_state_dict(q_network.state_dict())

print("训练完成")