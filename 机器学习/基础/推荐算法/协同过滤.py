#%%
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 示例用户-物品评分矩阵
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

# 计算用户之间的余弦相似度
user_similarity = cosine_similarity(ratings)
user_similarity=(user_similarity+1)/2
# 目标用户索引
target_user_index = 0

# 找到与目标用户最相似的K个用户
K = 2
similar_users_indices = np.argsort(user_similarity[target_user_index])[::-1][1:K+1]

# 预测目标用户对未评分物品的评分
predicted_ratings = np.zeros(ratings.shape[1])
for item_index in range(ratings.shape[1]):
    if ratings[target_user_index, item_index] == 0:  # 未评分的物品
        weighted_sum = 0
        similarity_sum = 0
        for user_index in similar_users_indices:
            if ratings[user_index, item_index] != 0:
                weighted_sum += user_similarity[target_user_index, user_index] * ratings[user_index, item_index]
                similarity_sum += user_similarity[target_user_index, user_index]
        if similarity_sum != 0:
            predicted_ratings[item_index] = weighted_sum / similarity_sum

# 生成推荐列表
recommendations = np.argsort(predicted_ratings)[::-1]

print("推荐列表:", recommendations)

#%%
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 示例用户-物品评分矩阵
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

# 计算物品之间的余弦相似度
item_similarity = cosine_similarity(ratings.T)
item_similarity = (item_similarity + 1) / 2  # 归一化到0-1范围

# 目标用户索引
target_user_index = 0

# 选择最相似的K个物品
K = 2

# 预测目标用户对未评分物品的评分
predicted_ratings = np.zeros(ratings.shape[1])
for item_index in range(ratings.shape[1]):
    if ratings[target_user_index, item_index] == 0:  # 未评分的物品
        # 找到与当前物品最相似的K个物品
        similar_items = np.argsort(item_similarity[item_index])[::-1][1:K+1]
        
        weighted_sum = 0
        similarity_sum = 0
        for similar_item_index in similar_items:
            if ratings[target_user_index, similar_item_index] != 0:
                weighted_sum += item_similarity[item_index, similar_item_index] * ratings[target_user_index, similar_item_index]
                similarity_sum += item_similarity[item_index, similar_item_index]
        if similarity_sum != 0:
            predicted_ratings[item_index] = weighted_sum / similarity_sum

# 生成推荐列表
recommendations = np.argsort(predicted_ratings)[::-1]

print("推荐列表:", recommendations)


#%%

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 示例数据：用户-物品评分矩阵
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

num_users, num_items = ratings.shape

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_size):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc = nn.Linear(embedding_size * 2, 1)

    def forward(self, user_indices, item_indices):
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)
        concat_embed = torch.cat((user_embed, item_embed), dim=1)
        output = self.fc(concat_embed)
        return output.squeeze()
    

# 将评分矩阵转换为训练数据和标签
user_indices = []
item_indices = []
ratings_labels = []

for user in range(num_users):
    for item in range(num_items):
        if ratings[user, item] > 0:
            user_indices.append(user)
            item_indices.append(item)
            ratings_labels.append(ratings[user, item])

user_indices = torch.tensor(user_indices, dtype=torch.long)
item_indices = torch.tensor(item_indices, dtype=torch.long)
ratings_labels = torch.tensor(ratings_labels, dtype=torch.float32)
user_indices,item_indices
#%%
# 初始化模型
embedding_size = 10
model = NeuralCollaborativeFiltering(num_users, num_items, embedding_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(user_indices, item_indices)
    loss = criterion(predictions, ratings_labels)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


#%%
# 评估模型
model.eval()
with torch.no_grad():
    test_user_indices = torch.tensor([0, 0, 2], dtype=torch.long)
    test_item_indices = torch.tensor([0, 0, 1], dtype=torch.long)
    predictions = model(test_user_indices, test_item_indices)
    print("Predictions:", predictions)

# Predictions: tensor([3.4203, 3.4203, 1.3188])
# 第一/二个预测值 3.4203 是模型对用户 0 和物品 0 的评分预测。
# 第三个预测值 1.3188 是模型对用户 2 和物品 1 的评分预测。

