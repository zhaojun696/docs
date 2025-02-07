
# GRU 层，其参数如下：
# input_size：输入特征的维度
# hidden_size：隐藏状态的维度
# num_layers：GRU 的层数
# batch_first：是否将批次维度放在第一个位置（即 (batch_size, seq_len, input_size)）
# bidirectional：是否是双向 GRU
# 那么，调用 forward 方法时，输入和输出的格式如下：
# 输入：
# input：形状为 (seq_len, batch_size, input_size) 或 (batch_size, seq_len, input_size)（如果 batch_first=True）的张量
# h_0（可选）：形状为 (num_layers * num_directions, batch_size, hidden_size) 的张量，表示初始隐藏状态
# 输出：
# output：形状为 (seq_len, batch_size, num_directions * hidden_size) 或 (batch_size, seq_len, num_directions * hidden_size)（如果 batch_first=True）的张量
# h_n：形状为 (num_layers * num_directions, batch_size, hidden_size) 的张量，表示最后一个时间步的隐藏状态
# 总结一下，nn.GRU 的 forward 方法返回的格式是 (output, h_n)，其中 output 包含了所有时间步的隐藏状态，而 h_n 包含了最后一个时间步的隐藏状态。




# nn.MultiheadAttention 的输入包括三个张量：查询（query）、键（key）和值（value）。这些张量的形状必须相同，具体格式如下：
# 查询（query）：形状为 (L, N, E) 的张量，其中：
# L 是目标序列的长度（即查询序列的长度）。
# N 是批量大小（batch size）。
# E 是嵌入维度（embedding dimension）或特征维度。
# 键（key）：形状为 (S, N, E) 的张量，其中：
# S 是源序列的长度（即键序列的长度）。
# N 是批量大小（batch size）。
# E 是嵌入维度（embedding dimension）或特征维度。
# 值（value）：形状为 (S, N, E) 的张量，其中：
# S 是源序列的长度（即值序列的长度）。
# N 是批量大小（batch size）。
# E 是嵌入维度（embedding dimension）或特征维度。
# 此外，还可以提供一个可选的布尔张量 key_padding_mask 和一个布尔张量 attn_mask：
# key_padding_mask：形状为 (N, S) 的张量，用于指示哪些键是填充的（即不参与注意力计算）。如果 key_padding_mask[i, j] 为 True，则第 i 个批次的第 j 个键将被忽略。
# attn_mask：形状为 (L, S) 的张量，用于指示哪些位置的注意力分数应该被屏蔽（即不参与注意力计算）。如果 attn_mask[i, j] 为 True，则第 i 个查询和第 j 个键之间的注意力分数将被屏蔽。

# 输出格式:
# nn.MultiheadAttention 的输出包括两个张量：
# 输出（output）：形状为 (L, N, E) 的张量，其中：
# L 是目标序列的长度（即查询序列的长度）。
# N 是批量大小（batch size）。
# E 是嵌入维度（embedding dimension）或特征维度。
# 注意力权重（attention weights）：形状为 (N, L, S) 的张量，其中：
# N 是批量大小（batch size）。
# L 是目标序列的长度（即查询序列的长度）。
# S 是源序列的长度（即键序列的长度）。

#%%
import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei:ei+1], encoder_hidden.detach())
        # encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # print(encoder_output.shape)

    decoder_input = torch.tensor([[SOS_token]])

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        loss += criterion(decoder_output, target_tensor[di].view(1))
        if decoder_input.item() == EOS_token:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

SOS_token = 0
EOS_token = 1

input_lang = {"hello": 2, "world": 3, "good": 4, "morning": 5, "how": 6, "are": 7, "you": 8}
output_lang = {"bonjour": 2, "monde": 3, "bon": 4, "matin": 5, "comment": 6, "va": 7, "tu": 8}

encoder = Encoder(len(input_lang) + 2, 256)
decoder = Decoder(256, len(output_lang) + 2)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)

criterion = nn.NLLLoss()

pairs = [
    ("hello world", "bonjour monde"),
    ("good morning", "bon matin"),
    ("how are you", "comment va tu")
]

for epoch in range(1000):
    for input_sentence, target_sentence in pairs:
        input_tensor = torch.tensor([input_lang[word] for word in input_sentence.split()] + [EOS_token])
        target_tensor = torch.tensor([output_lang[word] for word in target_sentence.split()] + [EOS_token])
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

def translate(input_sentence):
    with torch.no_grad():
        input_tensor = torch.tensor([input_lang[word] for word in input_sentence.split()])
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei:ei+1], encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]])
        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(10):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(list(output_lang.keys())[list(output_lang.values()).index(topi.item())])

            decoder_input = topi.squeeze().detach()

        return ' '.join(decoded_words)

for input_sentence, _ in pairs:
    print(f"Translating '{input_sentence}': {translate(input_sentence)}")


#%%
import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        attn_output, _ = self.attention(output, output, output)
        output = output + attn_output
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_heads):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        attn_output, _ = self.attention(output, output, output)
        output = output + attn_output
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden.detach())

    decoder_input = torch.tensor([[SOS_token]])

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        loss += criterion(decoder_output, target_tensor[di].view(1))
        if decoder_input.item() == EOS_token:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

SOS_token = 0
EOS_token = 1

input_lang = {"hello": 2, "world": 3, "good": 4, "morning": 5, "how": 6, "are": 7, "you": 8}
output_lang = {"bonjour": 2, "monde": 3, "bon": 4, "matin": 5, "comment": 6, "va": 7, "tu": 8}

encoder = Encoder(len(input_lang) + 2, 256, num_heads=8)
decoder = Decoder(256, len(output_lang) + 2, num_heads=8)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)

criterion = nn.NLLLoss()

pairs = [
    ("hello world", "bonjour monde"),
    ("good morning", "bon matin"),
    ("how are you", "comment va tu")
]

for epoch in range(1000):
    for input_sentence, target_sentence in pairs:
        input_tensor = torch.tensor([input_lang[word] for word in input_sentence.split()] + [EOS_token])
        target_tensor = torch.tensor([output_lang[word] for word in target_sentence.split()] + [EOS_token])
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

def translate(input_sentence):
    with torch.no_grad():
        input_tensor = torch.tensor([input_lang[word] for word in input_sentence.split()])
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]])
        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(10):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(list(output_lang.keys())[list(output_lang.values()).index(topi.item())])

            decoder_input = topi.squeeze().detach()

        return ' '.join(decoded_words)

for input_sentence, _ in pairs:
    print(f"Translating '{input_sentence}': {translate(input_sentence)}")



#%%
# 并行
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DataParallel

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs

# 设置参数
input_dim = 1000
output_dim = 1000
emb_dim = 256
hid_dim = 512
n_layers = 2
dropout = 0.5

# 创建模型实例
encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout)
decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seq2seq = Seq2Seq(encoder, decoder, device).to(device)

# 并行化模型
seq2seq = DataParallel(seq2seq)

# 示例输入
src = torch.randint(1, input_dim, (20, 32)).to(device)  # (src_len, batch_size)
trg = torch.randint(1, output_dim, (20, 32)).to(device)  # (trg_len, batch_size)

# 前向传播
output = seq2seq(src, trg)
output

#%%
nn.Embedding(3, 5)(torch.randint(1, 2,(1,))[0]),nn.Embedding(3, 5)(torch.randint(1, 2,(1,)))
