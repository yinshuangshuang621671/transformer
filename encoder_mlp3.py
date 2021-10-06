import torch
import torch.nn as nn
import numpy as np
from .multi_head_attention import MultiHeadAttention
from .pos_wise_feed_network import PositionalWiseFeedForward
# from position_encode import PositionalEncoding


def sequence_mask(seq):
    """
    sequence mask是为了使得decoder不能看见未来的信息。
    """
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                      diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask


def padding_mask(seq_k, seq_q):
    """
    维度扩充，标识出补0的位置
    """
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


def change_obs(data):
    """
    包含两部分内容，一部分是进行数据mask预处理，另一部分是进行mask操作
    return:
    返回mask矩阵，等于0的位置是true，其余是false
    """
    aa = data
    print("aa的维度", aa.size())

    # aa = aa.unsqueeze(0)
    # print("aa的维度", aa.size())

    # [2,3] 两行三列
    bb = torch.tensor([[0] * aa.size()[1]] * aa.size()[0])
    print("bb", bb)

    for i in range(aa.size()[0]):
        for j in range(aa.size()[1]):
            for k in aa[i][j]:
                if k == 0:
                    bb[i][j] += 1
    """
    bb tensor([[3, 3, 7],
        [3, 3, 7]])
    """
    print("bb", bb)

    for i in range(len(bb)):
        for j in range(len(bb[0])):
            if bb[i][j] == aa.size()[2]:
                bb[i][j] = 0
            else:
                bb[i][j] = 1
    # [3,3,7]
    print("bb", bb)
    # 最终输出的结果是，数据中全为0的行是0，其他行是1

    attention_mask = padding_mask(bb, bb)
    return attention_mask


class EncoderLayer(nn.Module):
    """
    编码器encoder结构中，其中一层网络结构，共六层
    """
    def __init__(self, model_dim=14, num_heads=4, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

        # TODO 新增
        # self.attn_mask = attn_mask

    def forward(self, inputs, attn_mask=None):
        """
         Args:
         ----
         inputs: 输入数据 ，维度[batch_size, sequence, model_dim]

         Return:
         ------
         经过全连接层、残差网络层及层归一化操作后的输出
         """
        print("哈哈哈哈哈哈哈", inputs.size())
        print(inputs.type())

        # todo
        attn_mask = change_obs(inputs)

        context, attention = self.attention.forward(inputs, inputs, inputs, attn_mask)
        output = self.feed_forward.forward(context)  # [B, sequence, model_dim]

        return output


# model_dim 需要重新固定数字，暂定40
def encoder(num_layers=6, model_dim=14, num_heads=4, ffn_dim=2048, dropout=0.0):
    """
    构建transformer中编码器网络结构

    Args:
    ----
    attn_mask: 标识出批次数据中，为0的数据的位置，是个Bool值矩阵
    num_layers: 编码器网络层数量
    model_dim: 数据编码维度
    num_heads:多头注意力机制中头的数量
    ffn_dim: 网络隐藏层神经元个数
    dropout:dropout网络层系数

    Return:
    ------
    编码器网络结构
    """
    encoder_layers = []

    for i in range(num_layers):
        # 改动：将attn_mask作为初始化参数传入，不然会报错
        encoder_layer_sig = EncoderLayer(model_dim, num_heads, ffn_dim, dropout)
        encoder_layers.append(encoder_layer_sig)

    # 另一种方法：在encoder编码层后面连接全连接层
    # sizes = [512, 64, 64, 8]
    # for j in range(3):
    #     encoder_layers += [nn.Linear(sizes[j], sizes[j + 1])]

    # todo 创建策略网络
    encoder_net = nn.Sequential(*encoder_layers)

    return encoder_net


class Sensor_Action1(nn.Module):
    def __init__(self, inputs, sensor_num=7, action_dim=7):
        super(Sensor_Action1, self).__init__()
        self.batch_size = inputs.size()[0]
        self.max_len = inputs.size()[1]
        self.model_dim = inputs.size()[2]
        self.sensor_num = sensor_num
        self.action_dim = action_dim

        self.linear_1 = nn.Linear(self.model_dim, self.action_dim)

    def forward(self, inputs):
        pos_output = torch.sum(inputs, dim=1, keepdim=True) / self.max_len
        print("定位体制输出维度", pos_output.shape)

        # todo 按照传感器数量截取
        inputs = inputs[:, :self.sensor_num, :]
        print("截取后维度", inputs.size())

        # 将定位体制输出和传感器输出连接在一起 [batch_size, sensor_num + 1, model_dim]
        output_cat = torch.cat((inputs, pos_output), dim=1)
        print("连接后输出的维度", output_cat.shape)

        # 对连接后的数据做全连接，输出维度是[batch_size, sensor_num+1, action_dim]
        output_action = self.linear_1(output_cat)
        print("最终的输出维度", output_action.shape)

        return output_action


def debug_encoder():
    """
    测试
    """
    # 构建测试数据
    Bs = 16
    vocab_size = 1000

    # inputs_len是一个列向量，里面的每个值表示一个批次中每个obs包含的对象个数，
    # 如果某个obs中的对象数量不够，应该得按照固定长度补，例如每个obs归一化后都包含16个对象
    inputs_len = np.random.randint(1, 30, Bs).reshape(Bs, 1)
    print("inputs_len", inputs_len)

    # max_seq_len表示一个批次中，每条数据中包含的对象个数，
    # 因为不同step中，获取到的传感器数据个数啥的和信号目标个数啥的可能并不相同!
    max_seq_len = int(max(inputs_len))
    print("max_seq_len", max_seq_len)

    x = np.zeros((Bs, max_seq_len), dtype=np.int64)
    for s in range(Bs):
        for j in range(inputs_len[s][0]):
            x[s][j] = j + 1
    x = torch.from_numpy(x)

    self_attention_mask = padding_mask(x, x)
    print("mask维度", self_attention_mask[0])
    print("mask的维度", self_attention_mask.shape)

    # 词嵌入与位置嵌入
    seq_embedding = nn.Embedding(vocab_size + 1, 40, padding_idx=0)
    # pos_embedding = PositionalEncoding(512, max_seq_len)

    input = seq_embedding(x)  # [bs, max_seq_len, model_dim]
    print('========input.shape', input.shape)

    # todo 做法：将attn_mask作为参数传递进去
    encoder_net = encoder(self_attention_mask)

    # todo 注：在这边直接将encoder作为参数传入encoder会报错
    output = encoder_net(input)
    print("最终的维度", output.size())

    """
    第一种：函数方式
    """
    # sensor_action(output, sensor_num=7)

    """
    第二种：模型方式
    """
    sensor_ac = Sensor_Action1(output)
    # print(list(sensor_ac.named_parameters()))

    oo = sensor_ac(output)
    print(oo.shape)


if __name__ == '__main__':
    debug_encoder()
