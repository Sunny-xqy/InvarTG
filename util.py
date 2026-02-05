from collections import defaultdict
import random
import torch
import pickle
import math
from tqdm import tqdm
from torch.utils.data import Dataset

class DatasetSample(Dataset):
    def __init__(self, data, sample_percent):
        self.data = data
        self.sample_psercent = sample_percent
        self.preprocess()

    def preprocess(self):
        self.samples = []
        for _, item in enumerate(self.data):
            pos_pairs = item['pos_pairs']
            neg_samples = item['neg_samples']
            pos_pairs_slice = []
            neg_samples_slice = []
            sample_size = math.floor(len(pos_pairs) / self.sample_psercent)
            for i in range(self.sample_psercent):
                pos_pairs_slice.append(pos_pairs[i*sample_size:(i+1)*sample_size])
                neg_samples_slice.append(neg_samples[i*sample_size:(i+1)*sample_size])
            self.samples.append({
                'pos_pairs': pos_pairs_slice,
                'neg_samples': neg_samples_slice
            })
            # [{'pos_pairs': [torch, ...],'neg_samples': []}]

    def __len__(self):
        return len(self.samples[0]['pos_pairs'])
    
    def __getitem__(self, idx):
        sample = {'pos_pairs': [], 'neg_samples': []}
        for item in self.samples:
            sample['pos_pairs'].append(item['pos_pairs'][idx])
            sample['neg_samples'].append(item['neg_samples'][idx])
        return sample
    
def collate_fn(batch):
    '''
    DataLoader 的 collate_fn，合并多个样本为一个 batch
    返回：
        pos_pairs: [B, S, 2]
        neg_samples: [B, S, K]
        t: [B]
    '''
    batch_data = {'pos_pairs': [], 'neg_samples': []}
    for i in range(len(batch[0]['pos_pairs'])):
        pos_pairs = []
        neg_samples = []
        for item in batch:
            pos_pairs.append(item['pos_pairs'][i])
            neg_samples.append(item['neg_samples'][i])
        batch_data['pos_pairs'].append(torch.cat(pos_pairs, dim=0))
        batch_data['neg_samples'].append(torch.cat(neg_samples, dim=0))

    return batch_data


def build_neighbor_dict(node_list, edge_index_list):
    '''
    建立邻居字典
    
    :param node_list: 节点列表
    :param edge_index_list: 边的节点索引列表[[1, 2], [1, 3]]
    '''
    neighbors = defaultdict(list)
    for src, dst in edge_index_list.t().tolist():
        neighbors[src].append(dst)
        neighbors[dst].append(src)

    for node in node_list:
        if node not in neighbors:
            neighbors[node] = []
    return neighbors

def random_walk_sequence(node, neighbors_dict, walk_length=10):
    '''
    生成随机游走序列
    
    :param node: 单个节点
    :param neighbors_dict: 邻居字典
    :param walk_length: 游走长度
    '''
    walk = [node]
    current = node
    for _ in range(walk_length - 1):
        neigh = neighbors_dict[current]
        if len(neigh) == 0:
            break
        current = random.choice(neigh)
        walk.append(current)
    return walk

def generate_positive_pairs_from_edges(node_list, edge_index_list, walk_length=10, window_size=2, num_walks=5):
    '''
    正样本生成
    
    :param node_list: 节点列表
    :param edge_index_list: 边的节点索引列表
    :param walk_length: 随机游走距离
    :param window_size: 采样窗口
    :param num_walks: 随机游走次数
    '''
    neighbors_dict = build_neighbor_dict(node_list, edge_index_list)
    positive_pairs = []

    for node in node_list:
        for _ in range(num_walks):
            walk = random_walk_sequence(node, neighbors_dict, walk_length)
            for i, vi in enumerate(walk):
                start = max(i - window_size, 0)
                end = min(i + window_size + 1, len(walk))
                for j in range(start, end):
                    if i != j:
                        vj = walk[j]
                        positive_pairs.append([vj, vi])
    return positive_pairs

# def generate_negative_samples(positive_pairs, node_all_list, num_neg=5):
#     node_all_list = list(node_all_list)  # 确保是 list
#     node_set = set(node_all_list)         # O(1) 查询

#     negative_samples = []
#     for vi, _ in tqdm(positive_pairs):
#         candidates = node_set - {vi}
#         neg_nodes = random.sample(candidates, num_neg)
#         negative_samples.append(neg_nodes)

#     return negative_samples

# def generate_negative_samples(positive_pairs, node_all_list, num_neg=5):
#     ''' 
#     生成num_neg个负样本 
#     :param positive_pairs: 正样本对 
#     :param all_nodes: 当前时间步所有节点的列表 
#     :param num_neg: 负样本对数量 每 正样本对 
#     '''
#     negative_samples = []
#     for vi, _ in tqdm(positive_pairs):
#         neg_nodes = random.sample([n for n in node_all_list if n != vi], num_neg)
#         negative_samples.append(neg_nodes)
#     return negative_samples

def generate_negative_samples(positive_pairs, node_all_list, num_neg=5, device='cuda:0'):
    '''
    生成num_neg个负样本
    
    :param positive_pairs: 正样本对
    :param all_nodes: 当前时间步所有节点的列表
    :param num_neg: 负样本对数量 每 正样本对
    '''
    node_all = torch.tensor(node_all_list, device=device)
    node_num = node_all.size(0)

    vi_list = torch.tensor([vi for vi, _ in positive_pairs], device=device)

    neg_samples = torch.randint(
        0, node_num,
        (len(positive_pairs), num_neg),
        device=device
    )

    mask = node_all[neg_samples] == vi_list.unsqueeze(1)
    while mask.any():
        neg_samples[mask] = torch.randint(0, node_num, (mask.sum(),), device=device)
        mask = node_all[neg_samples] == vi_list.unsqueeze(1)

    return node_all[neg_samples]

# def generate_negative_samples(positive_pairs, node_all_list, num_neg=5, device='cuda:0'):
#     '''
#     生成num_neg个负样本
    
#     :param positive_pairs: 正样本对
#     :param all_nodes: 当前时间步所有节点的列表
#     :param num_neg: 负样本对数量 每 正样本对
#     '''
#     node_all_list = list(node_all_list)
#     node_num = len(node_all_list)

#     negative_samples = []
#     for vi, _ in tqdm(positive_pairs):
#         neg_nodes = set()
#         while len(neg_nodes) < num_neg:
#             vj = node_all_list[random.randint(0, node_num - 1)]
#             if vj != vi:
#                 neg_nodes.add(vj)
#         negative_samples.append(list(neg_nodes))

#     return negative_samples

def generate_samples(data_name, node_all_list, node_list, edge_index_list, device, walk_length=10, window_size=5, num_walks=10, num_neg=5):
    '''
    生成训练样本

    :param node_all_list: 当前时间步所有节点的列表
    :param node_list: 当前时间步所有节点的列表
    :param edge_index_list: 当前时间步所有边的列表
    :param walk_length: 随机游走长度
    :param window_size: 窗口大小
    :param num_walks: 随机游走数量
    :param num_neg: 负样本对数量
    '''
    try:
        with open(f'data/{data_name}/{data_name}_samples.pkl', 'rb') as f:
            samples = pickle.load(f)
    except FileNotFoundError:
        samples = []
        for t in range(len(node_list)):
            print(f'正在生成第 {t + 1} 个时间步的正样本...')
            pos_pairs = generate_positive_pairs_from_edges(node_all_list, edge_index_list[t], walk_length, window_size, num_walks)
            print(f'正在生成第 {t + 1} 个时间步的负样本...')
            neg_samples = generate_negative_samples(pos_pairs, node_list[t], num_neg)
            samples.append({'pos_pairs': torch.tensor(pos_pairs, dtype=torch.long, device=device), 'neg_samples': torch.tensor(neg_samples, dtype=torch.long, device=device)})
        
        with open(f'data/{data_name}/{data_name}_samples.pkl', 'wb') as f:
            pickle.dump(samples, f)
    return samples