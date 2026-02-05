import torch
import torch.nn as nn
import torch.nn.functional as F


def l_g(n, pos_pairs, neg_samples):
    '''
    动态图优化，使用每个时间步随机游走生成的序列作为正样本，随机生成的序列作为负样本
    
    :param n_i: 训练得到的节点嵌入，按时间片切分为列表
    :param pos_pair: 正样本节点对，按时间片切分为列表
    :param neg_samples: 对应的负样本节点集，按时间片切分为列表
    '''

    loss = 0.0
    for i in range(len(n)):

        i_idx = pos_pairs[i][:, 0]
        j_idx = pos_pairs[i][:, 1]

        n_i = torch.index_select(n[i], 0, i_idx)
        n_j = torch.index_select(n[i], 0, j_idx)

        pos_score = torch.einsum('pd,pd->p', n_i, n_j)
        pos_loss = -F.logsigmoid(pos_score)

        n_k = n[i][neg_samples[i]].detach()


        neg_score = torch.einsum('pkd,pd->pk', n_k, n_i)
        # neg_score = torch.sum(n_k * n_i.unsqueeze(1), dim=-1)


        neg_loss = -F.logsigmoid(-neg_score)

        loss_t = pos_loss + neg_loss.sum(dim=1)
        loss += loss_t.mean()

    
    return loss / len(n)

def l_ER(y_hat_t, y_true_t):
    '''
    ERM Loss
    
    :param y_hat_t: 单个时间步的预测值
    :param y_true_t: 单个时间步的真实值
    '''
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(y_hat_t, y_true_t)

def l_IR(y_hat, y_true):
    '''
    IRM Loss
    
    :param y_hat: 所有时间步的预测值，以列表表示
    :param y_true: 所有时间步的真实值，以列表表示
    '''
    num_timesteps = len(y_hat)
    y_hat_all = torch.cat(y_hat, dim=0)
    y_true_all = torch.cat(y_true, dim=0)

    labels = torch.unique(torch.cat(y_true))
    l_ir = 0.0

    for y in labels:
        mask = (y_true_all == y)  # [T, N]
        masked_losses = []
        for t in range(num_timesteps):
            if mask[t].sum() == 0:
                continue
            loss_t = l_ER(y_hat_all[t][mask[t]], y_true_all[t][mask[t]])
            masked_losses.append(loss_t)
        if len(masked_losses) > 1:
            stacked = torch.stack(masked_losses)
            l_ir += torch.var(stacked)
    l_ir = l_ir / len(labels)
    return l_ir


def l_F(y_hat, y_true, mode, lambda_=0.5):
    '''
    不变任务优化，使用标签
    
    :param y_hat: 预测结果
    :param y_true: 真实标签
    :param lambda_: 控制l_er和l_ir贡献的超参数
    '''
    if mode == 'node_classification':
        for i in range(len(y_hat)):
            y_hat[i] = y_hat[i][y_true[i][0]]
            y_true[i] = y_true[i][1]
    
    list_l_er = []
    for y_hat_t, y_true_t in zip(y_hat, y_true):
        list_l_er.append(l_ER(y_hat_t, y_true_t))
    l_er = torch.stack(list_l_er).mean()
    l_ir = l_IR(y_hat, y_true)
    l_F = l_er + lambda_ * l_ir
    return l_F

def l_F_ablation_F(y_hat, y_true, mode):
    
    if mode == 'node_classification':
        for i in range(len(y_hat)):
            y_hat[i] = y_hat[i][y_true[i][0]]
            y_true[i] = y_true[i][1]
    
    list_loss = []
    for y_hat_t, y_true_t in zip(y_hat, y_true):
        loss_fn = nn.CrossEntropyLoss()
        list_loss.append(loss_fn(y_hat_t, y_true_t))

    loss = torch.stack(list_loss).mean()
    return loss
    