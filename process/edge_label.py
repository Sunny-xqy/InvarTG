import random
from tqdm import tqdm
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import os


def read_label(data_name):
    with open(f'data/{data_name}/{data_name}_final.pkl', 'rb') as f:
        data = pickle.load(f)

    # os.remove(f'data/{data_name}/{data_name}_final.pkl')

    edge_index_train_list = data['edge_index_list']
    edge_index_test_list = data['edge_index_test_list']
    node_list = [data['node_all_list'] for _ in range(len(edge_index_train_list))]
    # node_list = data['node_list']

    neg_edge_index_train_list = []
    neg_edge_index_test_list = []

    for t in tqdm(range(len(edge_index_train_list)), desc='加载中：'):
        edge_index_train_list_t = [i + j for i, j in zip(edge_index_train_list[t], edge_index_test_list[t])]
        src, dst = edge_index_train_list_t
        edge_set = set(zip(src, dst))

        num_neg_edges = len(edge_set)
        neg_edges = set()
        while len(neg_edges) < num_neg_edges:
            u = random.choice(node_list[t])
            v = random.choice(node_list[t])

            if u == v:
                continue

            if (u, v) in edge_set:
                continue

            neg_edges.add((u, v))

        neg_src, neg_dst = zip(*neg_edges)
        
        neg_edge_index_list_t = [list(neg_src), list(neg_dst)]
        neg_edge_index_train_list_t, neg_edge_index_test_list_t = train_test_split(np.array(neg_edge_index_list_t).T.tolist(), test_size=0.3, shuffle=True)
        neg_edge_index_train_list_t = np.array(neg_edge_index_train_list_t).T.tolist()
        neg_edge_index_test_list_t = np.array(neg_edge_index_test_list_t).T.tolist()

        neg_edge_index_train_list.append(neg_edge_index_train_list_t)
        neg_edge_index_test_list.append(neg_edge_index_test_list_t)

        # pos_train_src, pos_train_dst = zip(*edge_set)
        # edge_index_train_list[t] = [list(pos_train_src), list(pos_train_dst)]

        # src, dst = edge_index_test_list[t]
        # edge_set = set(zip(src, dst))
        # pos_test_src, pos_test_dst = zip(*edge_set)
        # edge_index_test_list[t] = [list(pos_test_src), list(pos_test_dst)]


    return [edge_index_train_list, neg_edge_index_train_list], [edge_index_test_list, neg_edge_index_test_list]
        
