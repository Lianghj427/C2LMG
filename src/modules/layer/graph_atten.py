import torch
import torch.nn as nn
import torch.nn.functional as F
    
class GraphAttention(nn.Module):

    def __init__(self, input_dim, output_dim, dropout=0, num_heads=1, negative_slope=0.2, bias=True, average=False, normalize=False):
        super(GraphAttention, self).__init__()
        """
        初始化图注意力层

        Arguments:
            input_dim (int): 输入层的维度
            output_dim (int): 输出层的维度
            dropout (int/float): 参数失活的概率
            negative_slope (int/float): 负范围中的倾斜率
            number_heads (int): 注意力头的个数
            bias (bool): 是否在输出中添加偏置项
            average (bool): 是否对所有的注意力头进行平均
            normalize (bool): 是否利用通信图将权重归零后的系数归一化
        """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.num_heads = num_heads
        self.average = average
        self.normalize = normalize

        self.W = nn.Parameter(torch.zeros(size=(input_dim, num_heads * output_dim)))
        self.a_i = nn.Parameter(torch.zeros(size=(num_heads, output_dim, 1)))
        self.a_j = nn.Parameter(torch.zeros(size=(num_heads, output_dim, 1)))
        if bias:
            if average:
                self.bias = nn.Parameter(torch.DoubleTensor(output_dim))
            else:
                self.bias = nn.Parameter(torch.DoubleTensor(num_heads * output_dim))
        else:
            self.register_parameter('bias', None)
        self.leakyrelu = nn.LeakyReLU(self.negative_slope)
        
        self.reset_parameters()
        
    def reset_parameters(self):

        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W.data, gain=gain)
        nn.init.xavier_normal_(self.a_i.data, gain=gain)
        nn.init.xavier_normal_(self.a_j.data, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, input, adj):
        """
        前向传播函数

        Arguments:
            input (tensor): 图注意力层的输入，[N, input_dim]
            adj (tensor): 由schedule生成的邻接矩阵, [N, N]

        Return:
            聚合之后的特征
        """

        # 对input实现线性变化，并产生多头
        # self.W: [input_dim, (num_heads*output_dim)]
        # h (tensor): 线性变化之后的矩阵 [N, num_heads, output_dim]

        h = torch.mm(input, self.W).view(-1, self.num_heads, self.output_dim)
        N = h.size()[0]
        
        e = []

        # 计算非归一化系数
        # a_i, a_j (tensors): [num_heads, output_dim, 1]
        for head in range(self.num_heads):
            # coeff_i, coeff_j (tensors): intermediate matrices to calculate unnormalized coefficients [N, 1]
            coeff_i = torch.mm(h[:, head, :], self.a_i[head, :, :])
            coeff_j = torch.mm(h[:, head, :], self.a_j[head, :, :])
            # coeff (tensor): the matrix of unnormalized coefficients for each head [N, N, 1]
            coeff = coeff_i.expand(N, N) + coeff_j.transpose(0, 1).expand(N, N)
            coeff = coeff.unsqueeze(-1)
            
            e.append(coeff)
            
        # e (tensor): the matrix of unnormalized coefficients for all heads [N * N * num_heads]
        # sometimes the unnormalized coefficients can be large, so regularization might be used 
        # to limit the large unnormalized coefficient values (TODO)
        e = self.leakyrelu(torch.cat(e, dim=-1))
            
        # adj: [N * N * num_heads]
        adj = adj.unsqueeze(-1).expand(N, N, self.num_heads)
        # attention (tensor): the matrix of coefficients used for the message aggregation [N * N * num_heads]
        attention = e * adj
        attention = F.softmax(attention, dim=1)
        # the weights from agents that should not communicate (send messages) will be 0, the gradients from 
        # the communication graph will be preserved in this way
        attention = attention * adj
        # normalize: make the some of weights from all agents be 1
        if self.normalize:
            attention += 1e-15
            attention = attention / attention.sum(dim=1).unsqueeze(dim=1).expand(N, N, self.num_heads)
            attention = attention * adj
        # dropout on the coefficients  
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # output (tensor): the matrix of output of the gat layer [N * (num_heads*output_dim)]
        output = []
        for head in range(self.num_heads):
            h_prime = torch.matmul(attention[:, :, head], h[:, head, :])
            output.append(h_prime)
        if self.average:
            output = torch.mean(torch.stack(output, dim=-1), dim=-1)
        else:
            output = torch.cat(output, dim=-1)
        
        if self.bias is not None:
            output += self.bias

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(input_dim={}, output_dim={})'.format(self.input_dim, self.output_dim)
