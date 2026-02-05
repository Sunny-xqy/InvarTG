import torch
import torch_scatter
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import loss
import evaluate



class Initializer(nn.Module):
    def __init__(self, node_num, in_dim):
        super().__init__()
        self.emb = nn.Embedding(node_num, in_dim)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self):
        return self.emb.weight


class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x_t, edge_index_list_t):
        x_t = self.conv1(x_t, edge_index_list_t)
        x_t = F.relu(x_t)
        x_t = self.conv2(x_t, edge_index_list_t)
        return x_t
    

class TemporalGNNLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.gnn = GNNEncoder(in_dim, hidden_dim)

    def forward(self, x_t, edge_index_list_t):
        x_t = self.gnn(x_t, edge_index_list_t)

        return x_t
    

class TimeEncoder(nn.Module):
    def __init__(self, d_time, device):
        super().__init__()

        self.d_time = d_time
        self.freq = nn.Parameter(torch.randn(int(d_time // 2), device=device))
    
    def forward(self, delta_t):
        delta_t = delta_t.unsqueeze(-1)
        omega_t = delta_t * self.freq
        return torch.cat([torch.sin(omega_t), torch.cos(omega_t)], dim=-1)


class FeatEncoder(nn.Module):
    def __init__(self, ):
        super().__init__()


class MLPMixerLayer(nn.Module):
    """MLP混合器"""
    def __init__(self, num_tokens, hidden_dim):
        super().__init__()

        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, num_tokens),
            nn.GELU(),
            nn.Linear(num_tokens, num_tokens)
        )

        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        y = self.token_norm(x)
        y = y.transpose(1, 2)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)
        x = x + y

        z = self.channel_norm(x)
        z = self.channel_mlp(z)
        x = x + z

        return x


class FinalProcessLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()

        self.final = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LeakyReLU(),
        )
        
    def forward(self, x, graph_type):
        if graph_type == 'discrete':
            return x.mean(dim=1)
        elif graph_type == 'continuous':
            return self.final(x.mean(dim=1))


class GraphContextForecaster(nn.Module):
    def __init__(self, node_num, in_dim, hidden_dim, m, graph_type, device, edge_feat):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_feat = edge_feat
        self.initializer = Initializer(node_num, in_dim)
        if graph_type == 'discrete':
            self.gnn_layer = TemporalGNNLayer(in_dim, hidden_dim)
            self.mixer_layer = MLPMixerLayer(m, hidden_dim)
            self.final_process_layer = FinalProcessLayer(hidden_dim, hidden_dim)
        elif graph_type == 'continuous':
            self.time_encoder = TimeEncoder(hidden_dim // 2, device)
            self.mixer_layer = MLPMixerLayer(m + 1, int(hidden_dim * 1.5) + edge_feat)
            self.final_process_layer = FinalProcessLayer(int(hidden_dim * 1.5) + edge_feat, hidden_dim)

    def forward(self, node_t, edge_index_t, m, graph_type, device, i_m_t=None):
        '''
        Docstring for forward
        
        :param node_t: 每个时间步的节点id
        :param edge_index_t: 每个时间步以列表表示的边
        :param i_m_t: 连续图输入时每个节点前m个交互的节点id和时间戳，当m=10时，[{'time': torch[N, 11], 'node_id': torch[N, 11]}, ...]，如果没有，则time为0
        :param m: 时间窗口大小（离散时间图中m表示前m个时间片，连续时间图中m表示前m个交互节点）
        :param mode: 离散时间图模式discrete或连续时间图模式continuous
        :param device: cuda或cpu等
        '''
        x_t = [self.initializer() for _ in range(len(edge_index_t))]
        x_t_new = []
        if graph_type == 'discrete':
            x_m = torch.zeros(x_t[0].size(0), m, self.hidden_dim, device=device)
            for i, x in enumerate(x_t):
                x_m_new = self.mixer_layer(x_m)
                x_new = self.final_process_layer(x_m_new, graph_type)
                x_t_new.append(x_new)

                x = self.gnn_layer(x, edge_index_t[i])
                padding = torch.zeros(x.size(0), dtype=torch.bool, device=device)
                padding[node_t[i]] = True
                x_padding = x
                x_padding[padding] = 0
                x_m = torch.cat([x_m[:, 1:, :], x_padding.unsqueeze(1)], dim=1)
        elif graph_type == 'continuous':
            for i, x in enumerate(x_t):
                x_m = x[i_m_t[i]['node_id']]
                t = i_m_t[i]['time'][:, :1] - i_m_t[i]['time']
                t_encoded = self.time_encoder(t)

                if self.edge_feat != 0:
                    edge_feat = i_m_t[i]['edge_feature']
                    x_m = torch.cat([x_m, t_encoded, edge_feat], dim=-1)
                else:
                    x_m = torch.cat([x_m, t_encoded], dim=-1)
                x_m = self.mixer_layer(x_m)

                x_new = self.final_process_layer(x_m, graph_type)
                x_new = x_new * (i_m_t[i]['time'].sum(dim=1) != 0).float().unsqueeze(-1)
                x_t_new.append(x_new)
        return x_t_new
    

class InvariantPredictor(nn.Module):
    '''
    不变任务预测器
    '''
    def __init__(self, hidden_dim, out_dim, mode, data):
        super().__init__()
        self.mode = mode
        self.data = data
        self.w = nn.Linear(hidden_dim, hidden_dim)
        if mode == 'node_classification':
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, out_dim),
                nn.ReLU()
            )
        if mode == 'link_prediction':
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )

    def forward(self, v):
        h = []
        y_hat = []
        for v_t in v:
            h_t = torch.tanh(self.w(v_t))
            y_hat_t = self.classifier(h_t)
            h.append(h_t)
            y_hat.append(y_hat_t)
        return h, y_hat
    

class InvarTG(nn.Module):
    def __init__(self, node_num, in_dim, hidden_dim, m, graph_type, out_dim, lr, device, mode, data=0, edge_feat=0, ablation='none'):
        super().__init__()
        self.G = GraphContextForecaster(node_num, in_dim, hidden_dim, m, graph_type, device, edge_feat)
        self.F = InvariantPredictor(hidden_dim, out_dim, mode, data)

        self.optimizer_g = optim.Adam(self.G.parameters(), lr=lr)

        if mode == 'node_classification':
            self.optimizer_F = optim.Adam(self.F.parameters(), lr=lr)
            self.optimizer_G = optim.Adam(list(self.G.parameters()) + list(self.F.parameters()), lr=lr)
        elif mode == 'link_prediction':
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.Linear(hidden_dim * 2, 2),
                nn.ReLU()
            )
            self.optimizer_F = optim.Adam(list(self.F.parameters()) + list(self.predictor.parameters()), lr=lr)
            self.optimizer_G = optim.Adam(list(self.G.parameters()) + list(self.F.parameters()) + list(self.predictor.parameters()), lr=lr)

        self.v = None

        self.m = m
        self.graph_type = graph_type
        self.mode = mode
        self.device = device
        if ablation == 'no_g':
            self.TGAT = TGATLayer(node_num, hidden_dim, hidden_dim)
            self.optimizer_ablation_g_g = optim.Adam(self.TGAT.parameters(), lr=lr)
            self.optimizer_ablation_g_G = optim.Adam(list(self.TGAT.parameters()) + list(self.F.parameters()), lr=lr)
        
        if ablation == 'no_F':
            self.optimizer_ablation_F_F = optim.Adam(self.F.parameters(), lr=lr)

    def forward(self, node_t, edge_index_t, i_m_t=None):
        v = self.G(node_t, edge_index_t, self.m, self.graph_type, self.device, i_m_t)
        h, y_hat = self.F(v)
        z = [v_t + h_t for v_t, h_t in zip(v, h)]
        return z, y_hat, h

    def train_g(self, node_t, edge_index_t, sample, i_m_t=None):
        l_g_batch = 0.0
        index = 0
        batch_num = len(sample)

        for batch in tqdm(sample):
            index += 1
            self.optimizer_g.zero_grad()
            pos_pairs = batch['pos_pairs']
            neg_samples = batch['neg_samples']
            try:
                v = self.G(node_t, edge_index_t, self.m, self.graph_type, self.device, i_m_t)
            except RuntimeError:
                raise RuntimeError('您似乎在变更参数m后未删除并重新生成dataset_name_final.pkl。It seems that you havnt regenerate dataset_name_final.pkl after changing parameter --m.')

            # for t in range(len(v)):
            #     v[t] = F.normalize(v[t], p=2, dim=-1)

            l_g = loss.l_g(v, pos_pairs, neg_samples)
            l_g.backward()
            self.optimizer_g.step()

            if (index == batch_num):
                self.v = [v_t.detach() for v_t in v]

            l_g_batch += l_g.item()            

        l_g_batch /= index
        return l_g_batch
    
    def train_F(self, node_t, edge_index_t, sample, i_m_t=None, lambda_=0.5):
        if self.v is None:
            v = self.G(node_t, edge_index_t, self.m, self.graph_type, self.device, i_m_t)
        else:
            v = self.v
        self.optimizer_F.zero_grad()
        if self.mode == 'node_classification':
            _, y_hat = self.F(v)
            sample_f = self.sample([[item[0], item[1]] for item in sample])
            y_true = [[item[0], item[1]] for item in sample_f]
            l_F = loss.l_F(y_hat, y_true, self.mode, lambda_)
        elif self.mode == 'link_prediction':
            _, v = self.F(v)

            y_true = []
            y_hat = []
            for i, item in enumerate(sample[0]):
                y_true_t = torch.tensor([1 for _ in range(len(item[0]))] + [0 for _ in range(len(sample[1][i][0]))], dtype=torch.long, device=self.device)
                y_true.append(y_true_t)
                src = sample[0][i][0] + sample[1][i][0]
                dst = sample[0][i][1] + sample[1][i][1]
                src_emb = v[i][src]
                dst_emb = v[i][dst]
                y_hat_t = torch.cat([src_emb, dst_emb], dim=1)
                y_hat_t = self.predictor(y_hat_t)
                # print(y_hat_t, y_true_t)
                y_hat.append(y_hat_t)
            l_F = loss.l_F(y_hat, y_true, self.mode, lambda_)
        l_F.backward()
        self.optimizer_F.step()
        return l_F.item()
    
    def sample(self, sample):
        index = 0
        for sample_t in sample:
            pos_mask = sample_t[1] == 1
            neg_mask = sample_t[1] == 0

            pos_indices = sample_t[0][pos_mask]
            neg_indices = sample_t[0][neg_mask]

            num_pos = pos_indices.size(0)

            perm = torch.randperm(neg_indices.size(0), device=neg_indices.device)
            sampled_neg = neg_indices[perm[:num_pos]]

            sampled_indices = torch.cat([pos_indices, sampled_neg], dim=0)
            sampled_labels = torch.cat([
                sample_t[1][pos_mask],
                sample_t[1][neg_mask][perm[:num_pos]]
            ], dim=0)

            shuffle = torch.randperm(sampled_indices.size(0), device=sampled_indices.device)
            sampled_indices = sampled_indices[shuffle]
            sampled_labels = sampled_labels[shuffle]

            sample[index] = [sampled_indices, sampled_labels]
            index += 1
        
        return sample

    def eval_F(self, node_t, edge_index_t, sample, i_m_t=None):
        with torch.no_grad():
            if self.v is None:
                v = self.G(node_t, edge_index_t, self.m, self.graph_type, self.device, i_m_t)
            else:
                v = self.v
            if self.mode == 'node_classification':
                _, y_hat = self.F(v)
                sample_f = self.sample([[item[0], item[1]] for item in sample])
                precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, precision_micro_all, recall_micro_all, f1_micro_all, precision_macro_all, recall_macro_all, f1_macro_all = evaluate.evaluate_node_classification(y_hat, [[item[0], item[1]] for item in sample_f])
                return precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, precision_micro_all, recall_micro_all, f1_micro_all, precision_macro_all, recall_macro_all, f1_macro_all
            elif self.mode == 'link_prediction':
                _, v = self.F(v)

                y_true = []
                y_hat = []
                for i, item in enumerate(sample[0]):
                    y_true_t = torch.tensor([1 for _ in range(len(item[0]))] + [0 for _ in range(len(sample[1][i][0]))], dtype=torch.long, device=self.device)
                    y_true.append(y_true_t)
                    src = sample[0][i][0] + sample[1][i][0]
                    dst = sample[0][i][1] + sample[1][i][1]
                    src_emb = v[i][src]
                    dst_emb = v[i][dst]
                    y_hat_t = torch.cat([src_emb, dst_emb], dim=1)
                    y_hat_t = self.predictor(y_hat_t)
                    y_hat.append(y_hat_t)
                auc_roc, ap, roc_all, auc_roc_all, ap_all = evaluate.evaluate_link_prediction(y_hat, y_true)    
                return auc_roc, ap, roc_all, auc_roc_all, ap_all

    def train_G(self, node_t, edge_index_t, sample, i_m_t=None):
        l_g_batch = 0.0
        index = 0
        for batch in sample:
            index += 1
            self.optimizer_G.zero_grad()
            pos_pairs = batch['pos_pairs']
            neg_samples = batch['neg_samples']

            z, _, _ = self.forward(node_t, edge_index_t, i_m_t)

            # for t in range(len(z)):
            #     z[t] = F.normalize(z[t], p=2, dim=-1)

            l_G = loss.l_g(z, pos_pairs, neg_samples)
            l_G.backward()
            self.optimizer_G.step()
            l_g_batch += l_G.item()

        l_g_batch /= index
        return l_g_batch
    
    def ablation_g_g(self, node_t, edge_index_t, sample):
        l_g_batch = 0.0
        index = 0
        batch_num = len(sample)

        for batch in sample:
            index += 1
            print(index, end='\r')
            self.optimizer_ablation_g_g.zero_grad()
            pos_pairs = batch['pos_pairs']
            neg_samples = batch['neg_samples']

            v = []

            for t in range(len(node_t)):
                v_t = self.TGAT(edge_index_t[t])
                v.append(F.normalize(v_t, p=2, dim=-1))
            l_g = loss.l_g(v, pos_pairs, neg_samples)
            l_g.backward()
            self.optimizer_ablation_g_g.step()

            if (index == batch_num):
                self.v = [v_t.detach() for v_t in v]

            l_g_batch += l_g.item()

        l_g_batch /= index
        return l_g_batch
    
    def ablation_g_G(self, node_t, edge_index_t, sample):
        l_g_batch = 0.0
        index = 0
        for batch in sample:
            index += 1
            self.optimizer_ablation_g_G.zero_grad()
            pos_pairs = batch['pos_pairs']
            neg_samples = batch['neg_samples']

            v = []
            for t in range(len(node_t)):
                v_t = self.TGAT(edge_index_t[t])
                v.append(v_t)
            h, _ = self.F(v)
            z = [v_t + h_t for v_t, h_t in zip(v, h)]
            l_G = loss.l_g(z, pos_pairs, neg_samples)
            l_G.backward()
            self.optimizer_ablation_g_G.step()
            l_g_batch += l_G.item()
        l_g_batch /= index
        return l_g_batch
    
    def eval_ablation_g(self, node_t, edge_index_t, sample):
        with torch.no_grad():
            v = []
            for t in range(len(node_t)):
                v_t = self.TGAT(edge_index_t[t])
                v.append(v_t)
            if self.mode == 'node_classification':
                _, y_hat = self.F(v)
                # print(y_hat[0])
                sample_f = self.sample([[item[0], item[1]] for item in sample])
                precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, precision_micro_all, recall_micro_all, f1_micro_all, precision_macro_all, recall_macro_all, f1_macro_all = evaluate.evaluate_node_classification(y_hat, [[item[0], item[1]] for item in sample_f])
                return precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, precision_micro_all, recall_micro_all, f1_micro_all, precision_macro_all, recall_macro_all, f1_macro_all
            elif self.mode == 'link_prediction':
                _, v = self.F(v)

                y_true = []
                y_hat = []
                for i, item in enumerate(sample[0]):
                    y_true_t = torch.tensor([1 for _ in range(len(item[0]))] + [0 for _ in range(len(sample[1][i][0]))], dtype=torch.long, device=self.device)
                    y_true.append(y_true_t)
                    src = sample[0][i][0] + sample[1][i][0]
                    dst = sample[0][i][1] + sample[1][i][1]
                    src_emb = v[i][src]
                    dst_emb = v[i][dst]
                    y_hat_t = torch.cat([src_emb, dst_emb], dim=1)
                    y_hat_t = self.predictor(y_hat_t)
                    y_hat.append(y_hat_t)
                auc_roc, ap, roc_all, auc_roc_all, ap_all = evaluate.evaluate_link_prediction(y_hat, y_true)    
                return auc_roc, ap, roc_all, auc_roc_all, ap_all
    
    def ablation_F_F(self, node_t, edge_index_t, sample, i_m_t=None):
        if self.v is None:
            v = self.G(node_t, edge_index_t, self.m, self.graph_type, self.device, i_m_t)
        else:
            v = self.v
        self.optimizer_ablation_F_F.zero_grad()
        if self.mode == 'node_classification':
            _, y_hat = self.F(v)
            sample_f = self.sample([[item[0], item[1]] for item in sample])
            y_true = [[item[0], item[1]] for item in sample_f]
            l_F = loss.l_F_ablation_F(y_hat, y_true, self.mode)
        elif self.mode == 'link_prediction':
            _, v = self.F(v)

            y_true = []
            y_hat = []
            for i, item in enumerate(sample[0]):
                y_true_t = torch.tensor([1 for _ in range(len(item[0]))] + [0 for _ in range(len(sample[1][i][0]))], dtype=torch.long, device=self.device)
                y_true.append(y_true_t)
                src = sample[0][i][0] + sample[1][i][0]
                dst = sample[0][i][1] + sample[1][i][1]
                src_emb = v[i][src]
                dst_emb = v[i][dst]
                y_hat_t = torch.cat([src_emb, dst_emb], dim=1)
                y_hat_t = self.predictor(y_hat_t)
                y_hat.append(y_hat_t)
            l_F = loss.l_F_ablation_F(y_hat, y_true, self.mode)
        l_F.backward()
        self.optimizer_ablation_F_F.step()
        return l_F.item()

class TimeEncoderTGAT(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
        )

    def forward(self, time_diffs):
        return self.time_embed(time_diffs)
       

class TGATLayer(nn.Module):
    def __init__(self, node_num, in_dim, out_dim, n_heads=4):
        super(TGATLayer, self).__init__()

        assert out_dim % n_heads == 0

        self.initializer = Initializer(node_num, in_dim)
        self.time_encoder = TimeEncoderTGAT(out_dim)

        self.node_num = node_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)

        self.attn_dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(out_dim, out_dim)

        self.residual_proj = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, edge_index, edge_time=None):
        device = edge_index.device

        x = self.initializer()
        x_residual = self.residual_proj(x)

        if edge_time is None: edge_time = torch.arange( edge_index.shape[1], device=device, dtype=torch.float32).unsqueeze(-1)

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(self.node_num, self.n_heads, self.head_dim)
        K = K.view(self.node_num, self.n_heads, self.head_dim)
        V = V.view(self.node_num, self.n_heads, self.head_dim)

        src, dst = edge_index[0], edge_index[1]
        q_dst = Q[dst]
        k_src = K[src]
        v_src = V[src]

        if edge_time is not None:
            t_emb = self.time_encoder(edge_time)
            t_emb = t_emb.view(-1, self.n_heads, self.head_dim)
            k_src = k_src + t_emb

        with torch.amp.autocast('cuda', enabled=False):
            attn_score = (q_dst.float() * k_src.float()).sum(dim=-1)
            attn_score = attn_score / (self.head_dim ** 0.5)
            attn_score = F.leaky_relu(attn_score)

        _, inverse = torch.unique(dst, return_inverse=True)

        attn_score = torch_scatter.scatter_softmax(attn_score, inverse, dim=0)

        attn_score = self.attn_dropout(attn_score)

        msg = v_src * attn_score.unsqueeze(-1)
        out = torch_scatter.scatter_add( msg, dst, dim=0, dim_size=self.node_num)

        out = out.reshape(self.node_num, self.out_dim)
        out = self.out_proj(out)

        out = out + x_residual
        return F.relu(out)


class InvarTGT(nn.Module):
    def __init__(self, node_num, in_dim, hidden_dim, m, graph_type, out_dim, lr, device, mode, data=0, edge_feat=0, ablation='none'):
        super().__init__()
        self.G = GraphContextForecaster(node_num, in_dim, hidden_dim, m, graph_type, device, edge_feat)
        self.F = InvariantPredictor(hidden_dim, out_dim, mode, data)

        self.optimizer_G = optim.Adam(self.G.parameters(), lr=lr)

        if mode == 'node_classification':
            self.optimizer_F = optim.Adam(self.F.parameters(), lr=lr)
        elif mode == 'link_prediction':
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.Linear(hidden_dim * 2, 2),
                nn.ReLU()
            )
            self.optimizer_F = optim.Adam(list(self.F.parameters()) + list(self.predictor.parameters()), lr=lr)
        
        self.m = m
        self.graph_type = graph_type
        self.mode = mode
        self.device = device



    def train_link_prediction(self, node_t, edge_index_t, sample, train_label, i_m_t=None, lambda_=2):
        index = 0
        for batch in sample:
            index += 1
            print(index, end='\r')

            pos_pairs = batch['pos_pairs']
            neg_samples = batch['neg_samples']

            for p in self.F.parameters():
                p.requires_grad = False
            for p in self.predictor.parameters():
                p.requires_grad = False
            self.optimizer_G.zero_grad()

            try:
                v = self.G(node_t, edge_index_t, self.m, self.graph_type, self.device, i_m_t)
            except RuntimeError:
                raise RuntimeError('您似乎在变更参数m后未删除并重新生成dataset_name_final.pkl。It seems that you havnt regenerate dataset_name_final.pkl after changing parameter --m.')

            for t in range(len(v)):
                v[t] = F.normalize(v[t], p=2, dim=-1)

            l_g = loss.l_g(v, pos_pairs, neg_samples)
            l_g.backward()
            self.optimizer_G.step()

            for p in self.G.parameters():
                p.requires_grad = False
            for p in self.F.parameters():
                p.requires_grad = True
            for p in self.predictor.parameters():
                p.requires_grad = True
            self.optimizer_F.zero_grad()

            _, v = self.F([v_t.detach() for v_t in v])

            y_true = []
            y_hat = []

            for i, item in enumerate(train_label[0]):
                y_true_t = torch.tensor([1 for _ in range(len(item[0]))] + [0 for _ in range(len(train_label[1][i][0]))], dtype=torch.long, device=self.device)
                y_true.append(y_true_t)
                src = train_label[0][i][0] + train_label[1][i][0]
                dst = train_label[0][i][1] + train_label[1][i][1]
                src_emb = v[i][src]
                dst_emb = v[i][dst]
                y_hat_t = torch.cat([src_emb, dst_emb], dim=1)
                y_hat_t = self.predictor(y_hat_t)
                y_hat.append(y_hat_t)
            l_F = loss.l_F(y_hat, y_true, self.mode, lambda_)
            l_F.backward()
            self.optimizer_F.step()

            for p in self.G.parameters():
                p.requires_grad = True
            self.optimizer_G.zero_grad()
            self.optimizer_F.zero_grad()
            
            v = self.G(node_t, edge_index_t, self.m, self.graph_type, self.device, i_m_t)
            h, _ = self.F(v)
            z = [v_t + h_t for v_t, h_t in zip(v, h)]

            for t in range(len(z)):
                z[t] = F.normalize(z[t], p=2, dim=-1)

            l_g = loss.l_g(z, pos_pairs, neg_samples)
            l_g.backward()
            self.optimizer_G.step()
            self.optimizer_F.step()
    
    def test_link_prediction(self, node_t, edge_index_t, test_label, i_m_t=None):
        with torch.no_grad():
            v = self.G(node_t, edge_index_t, self.m, self.graph_type, self.device, i_m_t)
            _, v = self.F(v)

            y_true = []
            y_hat = []
            for i, item in enumerate(test_label[0]):
                y_true_t = torch.tensor([1 for _ in range(len(item[0]))] + [0 for _ in range(len(test_label[1][i][0]))], dtype=torch.long, device=self.device)
                y_true.append(y_true_t)
                src = test_label[0][i][0] + test_label[1][i][0]
                dst = test_label[0][i][1] + test_label[1][i][1]
                src_emb = v[i][src]
                dst_emb = v[i][dst]
                y_hat_t = torch.cat([src_emb, dst_emb], dim=1)
                y_hat_t = self.predictor(y_hat_t)
                y_hat.append(y_hat_t)
            auc_roc, ap, roc_all, auc_roc_all, ap_all = evaluate.evaluate_link_prediction(y_hat, y_true)    
            return auc_roc, ap, roc_all, auc_roc_all, ap_all




            

