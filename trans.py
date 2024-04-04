import argparse
from thop import profile
import torch
from torchinfo import summary
from torch import nn
import torch.nn.functional as F

from self_supervised.Transformer.DotProductAttention import DotProductAttention

'''
        TO DO 修钱embedding层的操作
                理解多头注意力机制的输入，并且修改输入
'''


def get_hyperparameters():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--data_len', type=int, default=4800)
    parser.add_argument('--data_chan', type=int, default=2)
    parser.add_argument('--data_batch-size', type=int, default=32)
    parser.add_argument('--divide_batch_size', type=int, default=16)
    parser.add_argument('--divide_batch_size_added', type=int, default=32)
    parser.add_argument('--divide_group_len', type=int, default=301)  # data_len//divide_batch_size

    return parser.parse_args()


# 实现词表的编辑 32,4800,2_32 301 32
class Embedding(nn.Module):
    def __init__(self, divide_batch_size=16, output_size=32, data_chan=2):
        super(Embedding, self).__init__()
        self.divide_size = divide_batch_size
        self.Flatten = nn.Flatten(2)
        self.Linear1 = nn.Linear(self.divide_size * data_chan, output_size)
        self.dropout = nn.Dropout(0)

        self.classify_head = nn.Parameter(torch.randn(1, 1, self.divide_size * data_chan))
        self.Linear2 = nn.Linear(self.divide_size * data_chan, output_size)

    def forward(self, x):
        a, b, c = x.shape
        x = x.view(a, b // self.divide_size, self.divide_size, c)
        x = self.Flatten(x)
        x = self.Linear1(x)
        x = self.dropout(x)
        y = self.Linear2(self.classify_head)
        x = torch.cat((y.expand(a, 1, c * self.divide_size), x), dim=1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, divide_batch_added=32, divide_group_len=301, dropout=0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, divide_group_len, divide_batch_added))
        X = torch.arange(divide_group_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, divide_batch_added, 2, dtype=torch.float32) / divide_batch_added)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class Embedding_layer(nn.Module):
    def __init__(self):
        super(Embedding_layer, self).__init__()
        self.Embedding = Embedding()
        self.PositionalEncoding = PositionalEncoding()

    def forward(self, x):
        x = self.Embedding(x)
        a, b, c = x.shape
        y = torch.zeros(a, b, c).to(x.device)
        y = self.PositionalEncoding(y)
        y += x
        return y


class Norm(nn.Module):
    def __init__(self, normalized_shape, dropout=0):
        super(Norm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        x = self.dropout(x)
        x = self.LayerNorm(x)
        return x


def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 heads, dropout=0, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class MLP(nn.Module):
    def __init__(self, input_chan, middle_chan, output_chan):
        super().__init__()
        self.Linear1 = nn.Linear(input_chan, middle_chan)
        self.relu = nn.ReLU()
        self.Linear2 = nn.Linear(middle_chan, output_chan)

    def forward(self, x):
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.Linear2(x)
        return x


class Trans_Encoder_Block(nn.Module):  # 32 301 32
    def __init__(self, batch_size=32, heads=16, middle_chan=128, output_chan=32):
        super().__init__()
        self.Norm1 = Norm(normalized_shape=batch_size, dropout=0)
        self.MultiHeadAttention = MultiHeadAttention(key_size=batch_size, query_size=batch_size, value_size=batch_size,
                                                     num_hiddens=batch_size, heads=heads, dropout=0, bias=True)
        self.Norm2 = Norm(normalized_shape=batch_size, dropout=0)
        self.MLP = MLP(input_chan=batch_size, middle_chan=middle_chan, output_chan=output_chan)

    def forward(self, x):
        y = x
        x = self.Norm1(x)
        x = self.MultiHeadAttention(x, x, x, None)
        x += y
        y = x
        x = self.Norm2(x)
        x = self.MLP(x)
        x += y
        return x


class MLP_head(nn.Module):
    def __init__(self, input_chan, middle_chan, output_chan):
        super().__init__()
        self.Linear1 = nn.Linear(input_chan, middle_chan)
        self.relu = nn.ReLU()
        self.Linear2 = nn.Linear(middle_chan, output_chan)

    def forward(self, x):
        a, _, _ = x.shape
        x = x.view(a, -1)
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.Linear2(x)
        return x


class VIT(nn.Module):
    def __init__(self):
        super().__init__()
        self.Embedding_layer = Embedding_layer()
        self.Trans_Encoder_Block1 = Trans_Encoder_Block()
        self.MLP_head = MLP_head(9632, 300, 10)

    def forward(self, x):
        x = self.Embedding_layer(x)
        x = self.Trans_Encoder_Block1(x)
        x = self.MLP_head(x)
        return x


# 符合全覆盖标准
def test():
    x = torch.rand(128, 4800, 2)
    net = VIT()
    x = net(x)
    print(x.shape)
    return 0


def count_parameters():
    model = VIT()
    summary(model, (32, 4800, 2))
    input = torch.randn(32, 4800, 2)
    Flops, params = profile(model, inputs=(input,))
    print('Flops: % .4fG' % (Flops / 1000000000))
    return 0


if __name__ == '__main__':
    count_parameters()
    test()
