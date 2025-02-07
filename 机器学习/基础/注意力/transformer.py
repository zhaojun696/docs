
# nn.Transformer 的输入包括以下几个部分：
# src：形状为 (S, N, E) 的张量，表示源序列（即编码器的输入），其中：
# S 是源序列的长度。
# N 是批量大小（batch size）。
# E 是嵌入维度（embedding dimension）或特征维度。
# tgt：形状为 (T, N, E) 的张量，表示目标序列（即解码器的输入），其中：
# T 是目标序列的长度。
# N 是批量大小（batch size）。
# E 是嵌入维度（embedding dimension）或特征维度。
# src_mask（可选）：形状为 (S, S) 的张量，用于指示哪些位置的注意力分数应该被屏蔽（即不参与注意力计算）。如果 src_mask[i, j] 为 True，则第 i 个源位置和第 j 个源位置之间的注意力分数将被屏蔽。
# tgt_mask（可选）：形状为 (T, T) 的张量，用于指示哪些位置的注意力分数应该被屏蔽（即不参与注意力计算）。如果 tgt_mask[i, j] 为 True，则第 i 个目标位置和第 j 个目标位置之间的注意力分数将被屏蔽。通常，tgt_mask 会包含一个下三角矩阵（即未来信息屏蔽）。
# memory_mask（可选）：形状为 (T, S) 的张量，用于指示哪些位置的注意力分数应该被屏蔽（即不参与注意力计算）。如果 memory_mask[i, j] 为 True，则第 i 个目标位置和第 j 个源位置之间的注意力分数将被屏蔽。
# src_key_padding_mask（可选）：形状为 (N, S) 的张量，用于指示哪些源位置是填充的（即不参与注意力计算）。如果 src_key_padding_mask[i, j] 为 True，则第 i 个批次的第 j 个源位置将被忽略。
# tgt_key_padding_mask（可选）：形状为 (N, T) 的张量，用于指示哪些目标位置是填充的（即不参与注意力计算）。如果 tgt_key_padding_mask[i, j] 为 True，则第 i 个批次的第 j 个目标位置将被忽略。
# memory_key_padding_mask（可选）：形状为 (N, S) 的张量，用于指示哪些源位置是填充的（即不参与注意力计算）。如果 memory_key_padding_mask[i, j] 为 True，则第 i 个批次的第 j 个源位置将被忽略。
# 输出格式
# nn.Transformer 的输出是一个形状为 (T, N, E) 的张量，表示解码器的输出，其中：
# T 是目标序列的长度
# N 是批量大小（batch size）。
# E 是嵌入维度（embedding dimension）或特征维度。

#%%
import torch
import torch.nn as nn
import torch.optim as optim

# 定义超参数
src_vocab_size = 10  # 源词汇表大小
tgt_vocab_size = 10  # 目标词汇表大小
d_model = 512  # 模型维度
nhead = 8  # 多头注意力头数
num_encoder_layers = 6  # 编码器层数
num_decoder_layers = 6  # 解码器层数
dim_feedforward = 2048  # 前馈网络维度
dropout = 0.1  # dropout比例
hidden_size = 512  # LSTM隐藏层大小
num_layers = 2  # LSTM层数

# 定义输入数据
src = torch.randint(1, src_vocab_size, (5, 2))  # 源序列 (seq_len, batch_size)
tgt_seq_len=10
tgt = torch.randint(1, tgt_vocab_size, (tgt_seq_len, 2))  # 目标序列 (seq_len, batch_size)

tgt
#%%
# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.embedding_src = nn.Embedding(src_vocab_size, d_model)
        self.embedding_tgt = nn.Embedding(tgt_vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    # src_mask:
    # 意义: src_mask 是一个掩码张量，用于在编码器（encoder）的自注意力机制中屏蔽某些位置。它通常用于防止模型在处理源序列时注意到某些不相关的部分，例如填充的标记。
    # 用法: src_mask 的形状应该是 (src_len, src_len)，其中 src_len 是源序列的长度。掩码中的值为 True 的位置表示在计算注意力时不应该考虑这些位置。
    # tgt_mask:
    # 意义: tgt_mask 是一个掩码张量，用于在解码器（decoder）的自注意力机制中屏蔽某些位置。它通常用于防止解码器在生成目标序列时注意到未来的位置。
    # 用法: tgt_mask 的形状应该是 (tgt_len, tgt_len)，其中 tgt_len 是目标序列的长度。掩码中的值为 True 的位置表示在计算注意力时不应该考虑这些位置。
    # memory_mask:
    # 意义: memory_mask 是一个掩码张量，用于在解码器的编码器-解码器注意力机制中屏蔽某些位置。它通常用于防止解码器在生成目标序列时注意到编码器输出中的某些不相关的部分。
    # 用法: memory_mask 的形状应该是 (tgt_len, src_len)，其中 tgt_len 是目标序列的长度，src_len 是源序列的长度。掩码中的值为 True 的位置表示在计算注意力时不应该考虑这些位置。
    # src_key_padding_mask:
    # 意义: src_key_padding_mask 是一个掩码张量，用于在编码器的自注意力机制中屏蔽填充标记。它通常用于防止模型在处理源序列时注意到填充的标记。
    # 用法: src_key_padding_mask 的形状应该是 (batch_size, src_len)，其中 batch_size 是批次大小，src_len 是源序列的长度。掩码中的值为 True 的位置表示在计算注意力时不应该考虑这些位置。
    # tgt_key_padding_mask:
    # 意义: tgt_key_padding_mask 是一个掩码张量，用于在解码器的自注意力机制中屏蔽填充标记。它通常用于防止解码器在生成目标序列时注意到填充的标记。
    # 用法: tgt_key_padding_mask 的形状应该是 (batch_size, tgt_len)，其中 batch_size 是批次大小，tgt_len 是目标序列的长度。掩码中的值为 True 的位置表示在计算注意力时不应该考虑这些位置。
    # memory_key_padding_mask:
    # 意义: memory_key_padding_mask 是一个掩码张量，用于在解码器的编码器-解码器注意力机制中屏蔽填充标记。它通常用于防止解码器在生成目标序列时注意到编码器输出中的填充标记。
    # 用法: memory_key_padding_mask 的形状应该是 (batch_size, src_len)，其中 batch_size 是批次大小，src_len 是源序列的长度。掩码中的值为 True 的位置表示在计算注意力时不应该考虑这些位置。
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src = self.embedding_src(src) * (d_model ** 0.5)
        tgt = self.embedding_tgt(tgt) * (d_model ** 0.5)
        output = self.transformer(src, tgt, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        output = self.fc_out(output)
        return output


# 初始化模型、损失函数和优化器
transformer_model = TransformerModel(src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)

criterion = nn.CrossEntropyLoss(ignore_index=0)
transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=0.0001)


# 训练Transformer模型
transformer_model.train()
for epoch in range(100):
    transformer_optimizer.zero_grad()
    output = transformer_model(src, tgt[:-1])
    loss = criterion(output.view(-1, tgt_vocab_size), tgt[1:].view(-1))
    loss.backward()
    transformer_optimizer.step()
    print(f'Transformer Epoch {epoch+1}, Loss: {loss.item()}')
#%%
# 设置模型为评估模式
transformer_model.eval()

# 定义一个函数来进行预测
def predict(model, src, max_len=tgt_seq_len):
    with torch.no_grad():
        # 获取源序列的嵌入
        src_embedded = model.embedding_src(src) * (d_model ** 0.5)
        
        # 初始化目标序列的开始标记
        tgt_input = torch.full((1, src.size(1)), 1, dtype=torch.long)  # 使用1作为开始标记
        tgt_embedded = model.embedding_tgt(tgt_input) * (d_model ** 0.5)
        
        for _ in range(max_len):
            # 通过模型进行前向传播
            output = model.transformer(src_embedded, tgt_embedded)
            output = model.fc_out(output)
            
            # 获取预测的下一个词
            next_word_probs = output[-1, :, :]
            next_word = torch.argmax(next_word_probs, dim=1)
            
            # 将预测的词添加到目标序列中
            tgt_input = torch.cat((tgt_input, next_word.unsqueeze(0)), dim=0)
            tgt_embedded = model.embedding_tgt(tgt_input) * (d_model ** 0.5)
        
        return tgt_input

# 使用模型进行预测
predicted_sequence = predict(transformer_model, src)
print("Predicted sequence:", predicted_sequence)

