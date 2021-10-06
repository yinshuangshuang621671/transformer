import torch
import numpy as np

# todo 在整体trans代码中，两个参数是不相同的呢，padding_mask是根据解码和编码的输入，共同输出padding_mask
def padding_mask(seq_k, seq_q):
    print("seq_k的维度", seq_k.size())
    print("seq_q的维度", seq_q.size())
    # todo 为什么 第二个维度使用的是第二个参数的维度呢？？？
    len_q = seq_q.size(1)
    print('=len_q:', len_q)
    # `PAD` is 0
    pad_mask_ = seq_k.eq(0)  # 每句话的pad mask
    # print('==pad_mask_:', pad_mask_)

    # 维度是 [2,4,4]，第二个维度一开始不存在，是先扩充为1的维度，然后变为和第三个维度数据相同
    pad_mask = pad_mask_.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k] #作用于attention的mask
    # print('==pad_mask', pad_mask)
    return pad_mask


# 测试
def debug_padding_mask():
    # batch_size
    Bs = 2
    inputs_len = np.random.randint(1, 5, Bs).reshape(Bs, 1)
    print('==inputs_len:', inputs_len)

    max_seq_len = int(max(inputs_len))
    print("max_len", max_seq_len)

    x = np.zeros((Bs, max_seq_len), dtype=np.int64)
    print("x", x)
    for s in range(Bs):
        for j in range(inputs_len[s][0]):
            x[s][j] = j + 1

    print("x", x)
    x = torch.from_numpy(x)
    print('x的维度', x.shape)

    #
    mask = padding_mask(seq_k=x, seq_q=x)
    print('==mask:', mask.shape)


if __name__ == '__main__':
    debug_padding_mask()


# 这个函数为什么要进行维度扩充呢？？？？
# 原文链接：https: // blog.csdn.net / fanzonghao / article / details / 109240938