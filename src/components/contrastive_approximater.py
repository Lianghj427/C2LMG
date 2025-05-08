import torch
import torch.nn as nn

class ContrastiveApproximater(nn.Module):
    def __init__(self, args):
        super(ContrastiveApproximater, self).__init__()
        self.temperature = args.temperature
        self.base_temperature = args.base_temperature

        self.args = args

    def forward(self, agent_traj, emb_messages, mask, arange_mat):

        views = self._data_proc(agent_traj, emb_messages)
        loss = self._get_loss(views, mask, arange_mat)

        return loss

    def _get_loss(self, views, mask, arange_mat):

        views = nn.functional.normalize(views, dim=self.args.norm_type)

        batch_size, contrast_count = views.shape[0], views.shape[1]

        contrast_views = torch.cat(torch.unbind(views, dim=1), dim=0)
        anchor_views, anchor_count = views.clone()[:, 0], 1

        # 根据Eq. (2) from Ref. "Supervised Contrastive Learning" 进行loss计算
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_views, contrast_views.T),
            self.temperature
        )

        # 数值稳定
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 获取正样本的mask，并去除自身连接
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, arange_mat, 0)
        mask = mask * logits_mask
        logits_mask = logits_mask * (-mask+1)
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss
    
    def _data_proc(self, agent_traj, emb_messages):

        shape = emb_messages.shape
        bacth_num, trajectory_length, n_agents, n_neighbors = shape[0], shape[1], shape[2], shape[3]

        agent_traj = agent_traj.view(bacth_num, trajectory_length, n_agents, 1, self.args.rnn_hidden_dim)

        result = torch.cat([agent_traj, emb_messages], dim=3).view(bacth_num*trajectory_length*n_agents, n_neighbors+1, -1)

        return result