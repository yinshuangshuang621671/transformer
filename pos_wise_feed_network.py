import torch.nn as nn
import torch
import torch.nn.functional as F


# Position-wise Feed Forward Networks
class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        """model_dim:词向量的维度
            ffn_dim:卷积输出的维度 ？？？？
        """
        super(PositionalWiseFeedForward, self).__init__()

        # todo 直接这种Conv1d函数，就是初始化一个类，创建一个网络结构而已，是在下面传值
        self.w1 = nn.Conv1d(model_dim, ffn_dim, (1,))
        self.w2 = nn.Conv1d(ffn_dim, model_dim, (1,))
        # print('==w1:', self.w1)
        # print('==w2:', self.w2)

        self.dropout = nn.Dropout(dropout)
        # 层归一化
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):  # [B, sequence, model_dim]
        output = x.transpose(1, 2)  # [B, model_dim, sequence]
        print('===pos_wise_feed_network_output.shape:', output.shape)

        output = self.w2(F.relu(self.w1(output)))  # [B, model_dim, sequence]
        output = self.dropout(output.transpose(1, 2))  # [B, sequence, model_dim]

        # add residual and norm layer  添加残差连接和层归一化，这两个操作不改变数据维度
        output = self.layer_norm(x + output)

        print('===pos_wise_feed_network最终的输出维度:', output.shape)
        return output


def debug_PositionalWiseFeedForward():
    B, L_q, D_q = 32, 100, 512
    # 输入
    x = torch.rand(B, L_q, D_q)

    # 初始化网络
    model = PositionalWiseFeedForward()

    out = model(x)
    print('==out.shape:', out.shape)


if __name__ == '__main__':
    debug_PositionalWiseFeedForward()