import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SoftAttention(nn.Module):
    def __init__(self, input_size, num_attention_heads, hidden_size, hidden_dropout_prob=0, attention_probs_dropout_prob=0):
        super(SoftAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Sequential(
            nn.Linear(input_size, self.all_head_size),
            nn.ReLU()
        )

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 3, 2, 4)

    def forward(self, h, message, mask=None):
        """
        :param h: [batch_size, n_agents, hidden_size]
        :param message: [batch_size, n_agents, n_neighbors, hidden_size]
        :param mask: [batch_size, n_agents, n_neighbors]
        :return: [batch_size, n_agents, hidden_size]
        """ 

        bz, n, na, _ = message.shape
        h_in = h.view(bz, n, 1, self.all_head_size)

        mixed_query_layer = self.query(h_in)
        mixed_key_layer = self.key(message)
        mixed_value_layer = self.value(message)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        mask = mask.permute(0, 1, 3, 2).contiguous().repeat(1, self.num_attention_heads, 1, 1).view(bz, n, self.num_attention_heads, 1, na)

        attention_scores = (attention_scores / math.sqrt(self.attention_head_size)) * mask
        attention_scores_replace = torch.where(attention_scores == 0, torch.tensor(-1e6).to(attention_scores.device), attention_scores)

        attention_probs = F.softmax(attention_scores_replace, dim=-1) * mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 1, 3, 2, 4).contiguous()

        context_layer = context_layer.view(bz, n, self.all_head_size)

        hidden_states = self.dense(context_layer)
        out = self.LayerNorm(hidden_states + h_in.view(bz, n, self.all_head_size))

        return out

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias