import torch.nn as nn
import torch.nn.functional as f
import torch as th

from modules.layer.graph_atten import GraphAttention
from modules.layer.soft_atten import SoftAttention

class C2LMGAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(C2LMGAgent, self).__init__()

        self.n_agents = args.n_agents
        self.input_shape = input_shape

        self.args = args
        
        # encoder
        self.encoder = nn.Linear(self.input_shape, self.args.rnn_hidden_dim)
        self.gru = nn.GRUCell(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)

        self.others_encoder = nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)

        # concat the agent_obs and neighbnour_obs
        self.message_encoder = nn.Sequential(
            nn.Linear(self.args.rnn_hidden_dim*2, self.args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim),
            nn.ReLU()
        )

        # initialize the gat encoder
        if self.args.use_gat_encoder:
            if self.args.share_weight_gat:
                self.gat_encoder = GraphAttention(input_dim=self.args.rnn_hidden_dim, output_dim=self.args.rnn_hidden_dim)
                self.gat_encoders = nn.ModuleList([self.gat_encoder for _ in range(self.args.comm_rounds)])
            else:
                self.gat_encoders = nn.ModuleList([GraphAttention(input_dim=self.args.rnn_hidden_dim, output_dim=self.args.rnn_hidden_dim) 
                                                   for _ in range(self.args.comm_rounds)])
        
        # initial the mlp for obtaining the hard adj
        self.sub_schedulers = nn.ModuleList(nn.Sequential(nn.Linear(self.args.rnn_hidden_dim*2, self.args.rnn_hidden_dim//2),
                                                              nn.ReLU(),
                                                              nn.Linear(self.args.rnn_hidden_dim//2, 2))
                                                for _ in range(self.args.comm_rounds))
        
        # initial the soft attention module
        self.self_atten = nn.ModuleList([SoftAttention(self.args.rnn_hidden_dim, self.args.att_head_nums, self.args.rnn_hidden_dim)
                                             for _ in range(self.args.comm_rounds)])

        self.output_q = nn.Sequential(nn.Linear(args.rnn_hidden_dim * 2, args.rnn_hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(args.rnn_hidden_dim, args.n_actions))

        print('Init CommModule')

    def forward(self, obs, hidden_state):

        bz, na, eb = obs.size()
        n = self.n_agents

        obs_encode = self.encoder(obs.view(-1, eb))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h_out = self.gru(obs_encode, h_in)

        self_h_out = h_out.view(bz, na, self.args.rnn_hidden_dim)
        others_h_out = self.others_encoder(h_out.clone().detach()).view(bz, na, self.args.rnn_hidden_dim)

        self_info = self_h_out.repeat(1, 1, n).reshape(bz, na, n, self.args.rnn_hidden_dim)
        others_info = others_h_out.repeat(1, na, 1).reshape(bz, na, n, self.args.rnn_hidden_dim)
        message = th.cat([others_info, self_info], dim=-1)
        message = self.message_encoder(message)

        message_list = [message]
        for _ in range(self.args.comm_rounds-1):
            message_list.append(message)

        result = []
        for i in range(self.args.comm_rounds):
            if self.args.hard_att:
                adj = self.schedule_proc(mlp=self.sub_schedulers[i], h=self_h_out, encode_state=message_list[i], directed=self.args.directed)
            else:
                adj = th.ones(size=(bz, na, n, 1), device=message.device)
            result.append(f.elu(self.self_atten[i](self_h_out, message, adj)))

        aggre_message = th.mean(th.stack(result), dim=0)

        q = self.output_q(th.cat([self_h_out, aggre_message], dim=-1))

        return q.view(bz, na, -1), h_out.view(bz, na, -1)
    
    def train_forward(self, obs, hidden_state):

        bz, na, eb = obs.size()
        n = self.n_agents

        obs_encode = self.encoder(obs.view(-1, eb))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h_out = self.gru(obs_encode, h_in)

        con_h_out = h_out.clone()

        self_h_out = h_out.view(bz, na, self.args.rnn_hidden_dim)
        others_h_out = self.others_encoder(h_out.clone().detach()).view(bz, na, self.args.rnn_hidden_dim)

        self_info = self_h_out.repeat(1, 1, n).reshape(bz, na, n, self.args.rnn_hidden_dim)
        others_info = others_h_out.repeat(1, na, 1).reshape(bz, na, n, self.args.rnn_hidden_dim)

        message = th.cat([others_info, self_info], dim=-1)
        message = self.message_encoder(message)

        # message for contrastive learning
        con_message = message.clone()

        message_list = [message]
        for _ in range(self.args.comm_rounds-1):
            message_list.append(message)

        result = []
        for i in range(self.args.comm_rounds):
            if self.args.hard_att:
                adj = self.schedule_proc(mlp=self.sub_schedulers[i], h=self_h_out, encode_state=message_list[i], directed=self.args.directed)
            else:
                adj = th.ones(size=(bz, na, n, 1), device=message.device)
            result.append(f.elu(self.self_atten[i](self_h_out, message, adj)))

        aggre_message = th.mean(th.stack(result), dim=0)

        q = self.output_q(th.cat([self_h_out, aggre_message], dim=-1))

        return q.view(bz, na, -1), h_out.view(bz, na, -1), con_h_out.view(bz, na, -1), con_message

    def schedule_proc(self, mlp, h, encode_state, directed=True):

        n = self.args.n_agents
        bz, na, eb = h.size()

        h_in = h.repeat(1, 1, n).view(bz, na, n, eb)

        hard_attn_input = th.cat([h_in, encode_state], dim=-1)

        if directed is False:
            hard_attn_output = f.gumbel_softmax(mlp(hard_attn_input), hard=True, dim=-1)
        else:
            hard_attn_output = f.gumbel_softmax(0.5*mlp(hard_attn_input)+0.5*mlp(hard_attn_input.permute(0, 2, 1, 3)), hard=True, dim=-1)

        # [bz, n, na, 1]
        hard_attn_output = th.narrow(hard_attn_output, 3, 1, 1)
        
        return hard_attn_output
    
    def init_hidden(self):
        # make hidden states on same device as model
        return self.encoder.weight.new(1, self.args.rnn_hidden_dim).zero_()
