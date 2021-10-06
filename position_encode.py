import torch.nn as nn
import numpy as np
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        """初始化
        Args:
            d_model: 一个标量。模型的维度，论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        super(PositionalEncoding, self).__init__()
        # 根据论文给的公式，构造出PE矩阵
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)]).astype(np.float32)
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        position_encoding = torch.from_numpy(position_encoding)  # [max_seq_len, model_dim]
        # print('==position_encoding.shape:', position_encoding.shape)
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))  # [max_seq_len+1, model_dim]
        # print('==position_encoding.shape:', position_encoding.shape)
        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)

    def forward(self, input_len):
        """神经网络的前向传播。
        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。
          每一个张量的值代表这一批文本序列中对应的长度。
        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """
        # todo 找出这一批序列的最大长度，找出这批数据中，长度最长的句子
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos = tensor(
            [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        # print('==input_pos:', input_pos)#pad补齐
        # print('==input_pos.shape:', input_pos.shape)#[bs, max_len]
        return self.position_encoding(input_pos)


def debug_position():
    """d_model:模型的维度"""
    bs = 16
    x_scalar = np.random.randint(1, 30, bs).reshape(bs, 1)
    model = PositionalEncoding(d_model=512, max_seq_len=int(max(x_scalar)))
    x = torch.from_numpy(x_scalar)  # [bs, 1]
    print('===x:', x)
    print('====x.shape', x.shape)
    out = model(x)
    # [16, 27, 512]
    print('==out.shape:', out.shape)  # [bs, max_seq_len, model_dim]


if __name__ == '__main__':
    debug_position()