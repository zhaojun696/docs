#%%
import torch
import torch.nn as nn
import torch.optim as optim

# 数据准备
# 这里我们使用随机生成的整数数据作为示例
batch_size = 1
seq_length = 5
input_dim = 4
num_heads = 2

# 随机生成输入数据
input_data = torch.randint(0, 10, (batch_size, seq_length, input_dim)).float()
target_data = torch.randint(0, 10, (batch_size, seq_length, input_dim)).float()  # 随机生成目标数据
target_data
#%%

# 模型定义
class SimpleAttentionModel(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(SimpleAttentionModel, self).__init__()
        # embed_dim：输入张量的嵌入维度。
        # num_heads：注意力头的数量。
        # dropout：注意力权重上的dropout概率。
        # bias：是否在嵌入层中添加偏置。
        # add_bias_kv：是否在键和值的嵌入层中添加偏置。
        # add_zero_attn：是否在注意力机制中添加零向量。
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, q,k,v):
        
        # 注意：nn.MultiheadAttention的输入形状为(seq_length, batch_size, input_dim)
        attn_output, _ = self.multihead_attn(q,k,v)
        output = self.linear(attn_output)
        return output

model = SimpleAttentionModel(input_dim, num_heads)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # 调整目标数据的形状为(seq_length, batch_size, input_dim)
    target_data_transposed = target_data.transpose(0, 1)

    # 调整输入数据的形状为(seq_length, batch_size, input_dim)
    input_data_transposed = input_data.transpose(0, 1)
    # 查询（Query）：查询向量用于表示当前位置或时间步的信息，它会被用来与所有的键进行比较，以确定哪些键与当前查询最相关。
    # 键（Key）：键向量用于表示所有位置或时间步的信息，它们会被用来与查询进行比较，以确定每个位置或时间步与当前查询的相关性。
    # 值（Value）：值向量也用于表示所有位置或时间步的信息，但它们会被用来在确定相关性后，根据相关性权重对信息进行加权求和。
    # 在自注意力机制中，查询、键和值通常来自同一个输入序列，因此在这种情况下，我们会传入相同的input_data_transposed作为查询、键和值。
    # 这样做的目的是让模型能够学习到输入序列中不同位置之间的依赖关系。
    outputs = model(input_data_transposed,input_data_transposed,input_data_transposed)
    # outputs = model(target_data_transposed,input_data_transposed,input_data_transposed)
    
    loss = criterion(outputs, target_data_transposed)
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 预测
model.eval()
with torch.no_grad():
    test_input_transposed = input_data.transpose(0, 1)
    predicted_output = model(test_input_transposed,test_input_transposed,test_input_transposed)

#%%
predicted_output,target_data.transpose(0, 1)

