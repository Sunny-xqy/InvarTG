import json
import torch
from sklearn.model_selection import train_test_split


def read_label(data_name, device):
    read_label_by_data_name = globals()[f'read_label_{data_name}']
    return read_label_by_data_name(device)

def read_label_wikipedia_continuous(device):
    with open('data/wikipedia_continuous/wikipedia.json') as f:
        data = json.load(f)
    node_label_train = []
    node_label_test = []
    node_label_t = [0 for _ in range(data['num_users'])]
    node_label_index = [i for i in range(data['num_users'])]
    for i in range(len(data['graph'])):
        for edge in data['graph'][i]:
            node_label_t[edge['user_id']] = edge['label']
        node_label_t_train, node_label_t_test, node_label_index_train, node_label_index_test = train_test_split(node_label_t, node_label_index, test_size=0.25)
        node_label_train.append([torch.tensor(node_label_index_train, dtype=torch.long, device=device), torch.tensor(node_label_t_train, dtype=torch.long, device=device)])
        node_label_test.append([torch.tensor(node_label_index_test, dtype=torch.long, device=device), torch.tensor(node_label_t_test, dtype=torch.long, device=device)])
    return node_label_train, node_label_test

def read_label_wikipedia_discrete(device):
    return read_label_wikipedia_continuous(device)

def read_label_reddit_continuous(device):
    with open('data/reddit_continuous/reddit.json') as f:
        data = json.load(f)
    node_label_train = []
    node_label_test = []
    node_label_t = [0 for _ in range(data['num_users'])]
    node_label_index = [i for i in range(data['num_users'])]
    for i in range(len(data['graph'])):
        for edge in data['graph'][i]:
            node_label_t[edge['user_id']] = edge['label']
        node_label_t_train, node_label_t_test, node_label_index_train, node_label_index_test = train_test_split(node_label_t, node_label_index, test_size=0.25)
        node_label_train.append([torch.tensor(node_label_index_train, dtype=torch.long, device=device), torch.tensor(node_label_t_train, dtype=torch.long, device=device)])
        node_label_test.append([torch.tensor(node_label_index_test, dtype=torch.long, device=device), torch.tensor(node_label_t_test, dtype=torch.long, device=device)])
    return node_label_train, node_label_test

def read_label_reddit_discrete(device):
    return read_label_reddit_continuous(device)