import torch.nn as nn
import torch
from scaled_dot_pro_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        """model_dim:词向量维度，也是网络输入维度
           num_heads:头个数
        """
        super(MultiHeadAttention, self).__init__()
        # split个数也就是每个head要处理维度:512 / 8
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads

        # q,k,v三个网络的 输出纬度是 *，521
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        # 输出纬度是512
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        # 层归一化
        self.layer_norm = nn.LayerNorm(model_dim)


    def forward(self, key, value, query, attn_mask=None):
        # [B, sequence, model_dim]
        residual = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # [B, sequence, model_dim]
        key = self.linear_k(key)
        # [B, sequence, model_dim]
        value = self.linear_v(value)
        # [B, sequence, model_dim]
        query = self.linear_q(query)

        # print('===key.shape:', key.shape)
        # print('===value.shape:', value.shape)
        # print('==query.shape:', query.shape)

        # [B* num_heads, sequence, model_dim//*num_heads]
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        # [B* num_heads, sequence, model_dim//*num_heads]
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        # [B* num_heads, sequence, model_dim//*num_heads]
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        # print('===key.shape:', key.shape)
        # print('===value.shape:', value.shape)
        # print('==query.shape:', query.shape)

        # todo 修改
        print("重复前的维度", attn_mask.shape)
        if attn_mask.size():
            # 重复八次，每个head都进行mask
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        print("重复后的维度", attn_mask.size)

        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
          query, key, value, scale, attn_mask)

        print('===context.shape', context.shape)        # [B* num_heads, sequence, model_dim//*num_heads]
        print('===attention.shape', attention.shape)        # [B* num_heads, sequence, sequence]

        # # [B, sequence, model_dim]
        context = context.view(batch_size, -1, dim_per_head * num_heads)
        print('===context.shape', context.shape)

        # final linear projection
        output = self.linear_final(context)     # [B, sequence, model_dim]
        # print('===context.shape', context.shape)
        # dropout
        output = self.dropout(output)
        # add residual and norm layer
        output = self.layer_norm(residual + output)     # [B, sequence, model_dim]
        # print('==output.shape:', output.shape)

        # return output, attention

        # todo ************ 修改，只返回一个值 ************
        return output, attention


def debug_multi_head_attention():
    model = MultiHeadAttention()

    B, L_q, D_q = 32, 100, 512  # 批次数据量，句子序列长度， 网络纬度
    q, k, v = torch.rand(B, L_q, D_q), torch.rand(B, L_q, D_q), torch.rand(B, L_q, D_q)

    print('==q.shape:', q.shape)    # [B, sequence, model_dim]
    print('====k.shape', k.shape)   # [B, sequence, model_dim]
    print('==v.shape:', v.shape)    # [B, sequence, model_dim]

    out, _ = model(q, k, v)     # [B, sequence, model_dim]   [32, 100, 512]
    print('==out.shape:', out.shape)


if __name__ == '__main__':
    debug_multi_head_attention()

    """
    就是通过自注意力操作后，得到的数据维度还是和输入的一样。。。。。。
    就是一通数据转换
    """
