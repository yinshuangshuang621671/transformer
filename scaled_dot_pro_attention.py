import torch.nn as nn
import torch
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.5):
        super(ScaledDotProductAttention, self).__init__()

        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.
        Args:
          q: Queries张量，形状为[B, L_q, D_q]
          k: Keys张量，形状为[B, L_k, D_k]
          v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
          scale: 缩放因子，一个浮点标量
          attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
          上下文张量和attention张量
        """

        # todo sequence:是一个句子的长度，就是一个句子包含几个单词
        # todo 计算attention的时候，应该是进行了词嵌入吧？？？
        attention = torch.bmm(q, k.transpose(1, 2))  # [B, sequence, sequence]
        # print('===attention.shape', attention.size())

        if scale:
            attention = attention * scale

        # TODO attn_mask矩阵的作用，就是
        if attn_mask is not None:
            # 给需要mask的地方设置一个负无穷
            attention = attention.masked_fill_(attn_mask, -np.inf)

        # 输出纬度：[2,4,4]
        # print('===attention.shape', attention.size())

        attention = self.softmax(attention)  # [B, sequence, sequence]
        # print('===attention.shape', attention.shape)
        attention = self.dropout(attention)  # [B, sequence, sequence]
        # print('===attention.shape', attention.shape)
        context = torch.bmm(attention, v)  # [B, sequence, dim]
        return context, attention


def debug_scale_attention():
    model = ScaledDotProductAttention()

    # B, L_q, D_q 分别表示 batch_size， seq_len:句子的长度
    B, L_q, D_q = 2, 4, 10

    # todo False的地方是我们要进行处理的地方
    # pading_mask的维度：[2,4,4]
    pading_mask = torch.tensor([[[False, False, False, False],
                                 [False, False, False, False],
                                 [False, False, False, False],
                                 [False, False, False, False]],

                                [[False, False, True, True],
                                 [False, False, True, True],
                                 [False, False, True, True],
                                 [False, False, True, True]]])

    q, k, v = torch.rand(B, L_q, D_q), torch.rand(B, L_q, D_q), torch.rand(B, L_q, D_q)

    print('==q.shape:', q.shape)
    print('====k.shape', k.shape)
    print('==v.shape:', v.shape)

    out = model(q, k, v, attn_mask=pading_mask)


if __name__ == '__main__':
    debug_scale_attention()


"""
每一个单词都有QKV这三个向量，这里运用了注意力机制，
也有是会去求其他单词和该单词的匹配度，
那Q表示的就是与我这个单词相匹配的单词的属性，
K就表示我这个单词的本身的属性，
V表示的是我这个单词的包含的信息本身。
"""