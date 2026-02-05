import json
import pickle
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np

def read_data(data_name, graph_type, m=10, mode='node_classification', edge_feat=0, noise=0.0):
    '''
    读取data文件
    
    :param data_name: 文件名称
    :param graph_type: 图类型
    :param m: 每个节点保留的交互次数
    :return:
    '''
    print(noise)
    if graph_type == 'continuous':
        try:
            with open(f'data/{data_name}/{data_name}_final.pkl', 'rb') as f:
                data = pickle.load(f)
            if data['mode'] != mode:
                os.remove(f'data/{data_name}/{data_name}_samples.pkl')
                read_data_by_data_name = globals()[f'read_data_{data_name}']
                return read_data_by_data_name(m, mode, edge_feat, noise)
            else:
                return data['node_all_list'], data['node_list'], data['edge_index_list'], data['node_interaction_list']
        except FileNotFoundError:
            read_data_by_data_name = globals()[f'read_data_{data_name}']
            return read_data_by_data_name(m, mode, edge_feat, noise)
    elif graph_type == 'discrete':
        try:
            with open(f'data/{data_name}/{data_name}_final.pkl', 'rb')as f:
                data = pickle.load(f)
            if data['mode'] != mode:
                os.remove(f'data/{data_name}/{data_name}_samples.pkl')
                read_data_by_data_name = globals()[f'read_data_{data_name}']
                return read_data_by_data_name(mode, noise)
            else:
                return data['node_all_list'], data['node_list'], data['edge_index_list']
        except FileNotFoundError:
            read_data_by_data_name = globals()[f'read_data_{data_name}']
            return read_data_by_data_name(mode, noise)

def get_history_interaction(interaction_history, num_nodes, m, edge_dim=0):
    '''
    m次历史交互
    
    :param interaction_history: DefaultDict存储的交互历史
    :param num_nodes: 节点数
    :param m: 历史交互次数
    '''
    if edge_dim != 0:
        edge_feat = [[[0 for _ in range(edge_dim)] for _ in range(m + 1)] for _ in range(num_nodes)]
    else:
        edge_feat = []
    node_id = [[0 for _ in range(m + 1)] for _ in range(num_nodes)]
    time = [[0 for _ in range(m + 1)] for _ in range(num_nodes)]
    for node in range(num_nodes):
        history = interaction_history.get(node, [])
        if len(history) == 0:
            continue
        history = sorted(history, key=lambda x: x[0], reverse=True)
        recent = history[:m+1]

        for i, (t, nbr, feat) in enumerate(recent):
            if i == 0:
                node_id[node][i] = node
            else:
                node_id[node][i] = nbr
            time[node][i] = t + 1
            if edge_dim != 0:
                edge_feat[node][i] = feat[:edge_dim]
    
    return node_id, time, edge_feat


def read_data_wikipedia_continuous(m, mode, edge_feat=0, noise=0.0):
    with open('data/wikipedia_continuous/wikipedia.json') as f:
        data = json.load(f)
    node_all_list = [i for i in range(data['num_nodes'])]
    edge_index_list = []
    edge_index_test_list = []
    node_list = []
    node_interaction_list = []

    for t, i in enumerate(data['graph']):
        edge_index_list_t = [[] for _ in range(4)]
        node_list_t = []

        for edge in i:
            edge_index_list_t[0].append(edge['user_id'])
            edge_index_list_t[1].append(edge['item_id'])
            edge_index_list_t[2].append(edge['timestamp'])
            if edge_feat != 0:
                edge_index_list_t[3].append(edge['feature'])
            else:
                edge_index_list_t[3].append(0)

            node_list_t.append(edge['user_id'])
            node_list_t.append(edge['item_id'])

        if mode == 'link_prediction':
            edge_index_list_t, edge_index_test_list_t = train_test_split(list(map(list, zip(*edge_index_list_t))), test_size=0.3, shuffle=True)
            edge_index_list_t, _ = train_test_split(edge_index_list_t, test_size=noise) if float(noise) != 0.0 else (edge_index_list_t, [])
            edge_index_list_t = list(map(list, zip(*edge_index_list_t)))
            edge_index_test_list_t = list(map(list, zip(*edge_index_test_list_t)))
            edge_index_test_list.append([edge_index_test_list_t[0], edge_index_test_list_t[1]])
        elif mode == 'node_classification':
            edge_index_list_t, _ = train_test_split(list(map(list, zip(*edge_index_list_t))), test_size=noise) if float(noise) != 0.0 else (edge_index_list_t, [])
            edge_index_list_t = list(map(list, zip(*edge_index_list_t)))
        
        edge_index_list.append(edge_index_list_t)

        interaction_history = defaultdict(list)
        src, dst, timestamp, feature = edge_index_list_t
        for u, v, t_n, f_n in zip(src, dst, timestamp, feature):
            interaction_history[u].append((t_n, v, f_n))
            interaction_history[v].append((t_n, u, f_n))


        node_list_t = list(set(node_list_t))
        node_list.append(node_list_t)

        if t != 0:
            for item in edge_index_list[:t]:
                src, dst, timestamp, feature = item
                for u, v, t_n, f_n in zip(src, dst, timestamp, feature):
                    interaction_history[u].append((t_n, v, f_n))
                    interaction_history[v].append((t_n, u, f_n))
        if t == len(data['graph']) - 1:
            edge_index_list[t] = [edge_index_list_t[0], edge_index_list_t[1]]

        node_id_t, time_t, edge_feature_t = get_history_interaction(interaction_history, data['num_nodes'], m, edge_feat)
        if edge_feat != 0:
            node_interaction_list.append({'node_id': node_id_t, 'time': time_t, 'edge_feature': edge_feature_t})
        else:
            node_interaction_list.append({'node_id': node_id_t, 'time': time_t})

    edge_index_list = [[i[0], i[1]] for i in edge_index_list]

    with open(f'data/wikipedia_continuous/wikipedia_continuous_final.pkl', 'wb')as f:
        pickle.dump({'mode': mode, 'node_all_list': node_all_list, 'node_list': node_list, 'edge_index_list': edge_index_list, 'edge_index_test_list': edge_index_test_list, 'node_interaction_list': node_interaction_list}, f)
    return node_all_list, node_list, edge_index_list, node_interaction_list
        

def read_data_wikipedia_discrete(mode, noise=0.0):
    with open('data/wikipedia_discrete/wikipedia.json') as f:
        data = json.load(f)
    node_all_list = [i for i in range(data['num_nodes'])]
    edge_index_list = []
    edge_index_test_list = []
    node_list = []
    for i in data['graph']:
        edge_index_list_t = [[] for _ in range(2)]
        node_list_t = []
        for edge in i:
            edge_index_list_t[0].append(edge['user_id'])
            edge_index_list_t[1].append(edge['item_id'])

            node_list_t.append(edge['user_id'])
            node_list_t.append(edge['item_id'])
        
        if mode == 'link_prediction':
            edge_index_list_t, edge_index_test_list_t = train_test_split(list(map(list, zip(*edge_index_list_t))), test_size=0.3, shuffle=True)
            edge_index_list_t, _ = train_test_split(edge_index_list_t, test_size=noise) if float(noise) != 0.0 else (edge_index_list_t, [])
            edge_index_list_t = list(map(list, zip(*edge_index_list_t)))
            edge_index_test_list_t = list(map(list, zip(*edge_index_test_list_t)))
            edge_index_test_list.append([edge_index_test_list_t[0], edge_index_test_list_t[1]])
        elif mode == 'node_classification':
            edge_index_list_t, _ = train_test_split(list(map(list, zip(*edge_index_list_t))), test_size=noise) if float(noise) != 0.0 else (edge_index_list_t, [])
            edge_index_list_t = list(map(list, zip(*edge_index_list_t)))
        
        edge_index_list.append(edge_index_list_t)
        node_list_t = list(set(node_list_t))
        node_list.append(node_list_t)

    edge_index_list = [[i[0], i[1]] for i in edge_index_list]

    with open(f'data/wikipedia_discrete/wikipedia_discrete_final.pkl', 'wb')as f:
        pickle.dump({'mode': mode, 'node_all_list': node_all_list, 'node_list': node_list, 'edge_index_list': edge_index_list, 'edge_index_test_list': edge_index_test_list}, f)
    return node_all_list, node_list, edge_index_list

def read_data_reddit_continuous(m, mode, edge_feat=0, noise=0.0):
    with open('data/reddit_continuous/reddit.json') as f:
        data = json.load(f)
    node_all_list = [i for i in range(data['num_nodes'])]
    edge_index_list = []
    edge_index_test_list = []
    node_list = []
    node_interaction_list = []

    for t, i in enumerate(data['graph']):
        edge_index_list_t = [[] for _ in range(4)]
        node_list_t = []
        for edge in i:
            edge_index_list_t[0].append(edge['user_id'])
            edge_index_list_t[1].append(edge['item_id'])
            edge_index_list_t[2].append(edge['timestamp'])
            if edge_feat != 0:
                edge_index_list_t[3].append(edge['feature'])
            else:
                edge_index_list_t[3].append(0)

            node_list_t.append(edge['user_id'])
            node_list_t.append(edge['item_id'])

        if mode == 'link_prediction':
            edge_index_list_t, edge_index_test_list_t = train_test_split(list(map(list, zip(*edge_index_list_t))), test_size=0.3, shuffle=True)
            edge_index_list_t, _ = train_test_split(edge_index_list_t, test_size=noise) if float(noise) != 0.0 else (edge_index_list_t, [])
            edge_index_list_t = list(map(list, zip(*edge_index_list_t)))
            edge_index_test_list_t = list(map(list, zip(*edge_index_test_list_t)))
            edge_index_test_list.append([edge_index_test_list_t[0], edge_index_test_list_t[1]])
        elif mode == 'node_classification':
            edge_index_list_t, _ = train_test_split(list(map(list, zip(*edge_index_list_t))), test_size=noise) if float(noise) != 0.0 else (edge_index_list_t, [])
            edge_index_list_t = list(map(list, zip(*edge_index_list_t)))
        
        edge_index_list.append(edge_index_list_t)

        interaction_history = defaultdict(list)
        src, dst, timestamp, feature = edge_index_list_t
        for u, v, t_n, f_n in zip(src, dst, timestamp, feature):
            interaction_history[u].append((t_n, v, f_n))
            interaction_history[v].append((t_n, u, f_n))


        node_list_t = list(set(node_list_t))
        node_list.append(node_list_t)

        if t != 0:
            for item in edge_index_list[:t]:
                src, dst, timestamp, feature = item
                for u, v, t_n, f_n in zip(src, dst, timestamp, feature):
                    interaction_history[u].append((t_n, v, f_n))
                    interaction_history[v].append((t_n, u, f_n))
        if t == len(data['graph']) - 1:
            edge_index_list[t] = [edge_index_list_t[0], edge_index_list_t[1]]

        node_id_t, time_t, edge_feature_t = get_history_interaction(interaction_history, data['num_nodes'], m, edge_feat)
        if edge_feat != 0:
            node_interaction_list.append({'node_id': node_id_t, 'time': time_t, 'edge_feature': edge_feature_t})
        else:
            node_interaction_list.append({'node_id': node_id_t, 'time': time_t})

    edge_index_list = [[i[0], i[1]] for i in edge_index_list]

    with open(f'data/reddit_continuous/reddit_continuous_final.pkl', 'wb')as f:
        pickle.dump({'mode': mode, 'node_all_list': node_all_list, 'node_list': node_list, 'edge_index_list': edge_index_list, 'edge_index_test_list': edge_index_test_list, 'node_interaction_list': node_interaction_list}, f)
    return node_all_list, node_list, edge_index_list, node_interaction_list

def read_data_reddit_discrete(mode, noise=0.0):
    with open('data/reddit_discrete/reddit.json') as f:
        data = json.load(f)
    node_all_list = [i for i in range(data['num_nodes'])]
    edge_index_list = []
    edge_index_test_list = []
    node_list = []
    for i in data['graph']:
        edge_index_list_t = [[] for _ in range(2)]
        node_list_t = []
        for edge in i:
            edge_index_list_t[0].append(edge['user_id'])
            edge_index_list_t[1].append(edge['item_id'])

            node_list_t.append(edge['user_id'])
            node_list_t.append(edge['item_id'])
        
        if mode == 'link_prediction':
            edge_index_list_t, edge_index_test_list_t = train_test_split(list(map(list, zip(*edge_index_list_t))), test_size=0.3, shuffle=True)
            edge_index_list_t, _ = train_test_split(edge_index_list_t, test_size=noise) if float(noise) != 0.0 else (edge_index_list_t, [])
            edge_index_list_t = list(map(list, zip(*edge_index_list_t)))
            edge_index_test_list_t = list(map(list, zip(*edge_index_test_list_t)))
            edge_index_test_list.append([edge_index_test_list_t[0], edge_index_test_list_t[1]])
        elif mode == 'node_classification':
            edge_index_list_t, _ = train_test_split(list(map(list, zip(*edge_index_list_t))), test_size=noise) if float(noise) != 0.0 else (edge_index_list_t, [])
            edge_index_list_t = list(map(list, zip(*edge_index_list_t)))
        
        edge_index_list.append(edge_index_list_t)
        node_list_t = list(set(node_list_t))
        node_list.append(node_list_t)

    edge_index_list = [[i[0], i[1]] for i in edge_index_list]

    with open(f'data/reddit_discrete/reddit_discrete_final.pkl', 'wb')as f:
        pickle.dump({'mode': mode, 'node_all_list': node_all_list, 'node_list': node_list, 'edge_index_list': edge_index_list, 'edge_index_test_list': edge_index_test_list}, f)
    return node_all_list, node_list, edge_index_list

def read_data_enron_continuous(m, mode, edge_feat=0, noise=0.0):
    with open('data/enron_continuous/enron.json') as f:
        data = json.load(f)
    node_all_list = [i for i in range(data['num_nodes'])]
    edge_index_list = []
    edge_index_test_list = []
    node_list = []
    node_interaction_list = []

    for t, i in enumerate(data['graph']):
        edge_index_list_t = [[] for _ in range(4)]
        node_list_t = []
        for edge in i:
            edge_index_list_t[0].append(edge['src'])
            edge_index_list_t[1].append(edge['dst'])
            edge_index_list_t[2].append(edge['timestamp'])
            if edge_feat != 0:
                edge_index_list_t[3].append(edge['feature'])
            else:
                edge_index_list_t[3].append(0)

            node_list_t.append(edge['src'])
            node_list_t.append(edge['dst'])

        edge_index_list_t, edge_index_test_list_t = train_test_split(list(map(list, zip(*edge_index_list_t))), test_size=0.3, shuffle=True)
        edge_index_list_t, _ = train_test_split(edge_index_list_t, test_size=noise) if float(noise) != 0.0 else (edge_index_list_t, [])
        edge_index_list_t = list(map(list, zip(*edge_index_list_t)))
        edge_index_test_list_t = list(map(list, zip(*edge_index_test_list_t)))
        edge_index_test_list.append([edge_index_test_list_t[0], edge_index_test_list_t[1]])
        
        edge_index_list.append(edge_index_list_t)

        interaction_history = defaultdict(list)
        src, dst, timestamp, feature = edge_index_list_t
        for u, v, t_n, f_n in zip(src, dst, timestamp, feature):
            interaction_history[u].append((t_n, v, f_n))
            interaction_history[v].append((t_n, u, f_n))


        node_list_t = list(set(node_list_t))
        node_list.append(node_list_t)

        if t != 0:
            for item in edge_index_list[:t]:
                src, dst, timestamp, feature = item
                for u, v, t_n, f_n in zip(src, dst, timestamp, feature):
                    interaction_history[u].append((t_n, v, f_n))
                    interaction_history[v].append((t_n, u, f_n))
        if t == len(data['graph']) - 1:
            edge_index_list[t] = [edge_index_list_t[0], edge_index_list_t[1]]

        node_id_t, time_t, edge_feature_t = get_history_interaction(interaction_history, data['num_nodes'], m)
        if edge_feat != 0:
            node_interaction_list.append({'node_id': node_id_t, 'time': time_t, 'edge_feature': edge_feature_t})
        else:
            node_interaction_list.append({'node_id': node_id_t, 'time': time_t})

    edge_index_list = [[i[0], i[1]] for i in edge_index_list]

    with open(f'data/enron_continuous/enron_continuous_final.pkl', 'wb')as f:
        pickle.dump({'mode': mode, 'node_all_list': node_all_list, 'node_list': node_list, 'edge_index_list': edge_index_list, 'edge_index_test_list': edge_index_test_list, 'node_interaction_list': node_interaction_list}, f)
    return node_all_list, node_list, edge_index_list, node_interaction_list

def read_data_enron_discrete(mode, noise=0.0):
    with open('data/enron_discrete/enron.json') as f:
        data = json.load(f)
    node_all_list = [i for i in range(data['num_nodes'])]
    edge_index_list = []
    edge_index_test_list = []
    node_list = []
    for i in data['graph']:
        
        edge_index_list_t = [[] for _ in range(2)]
        node_list_t = []
        for edge in i:
            edge_index_list_t[0].append(edge['src'])
            edge_index_list_t[1].append(edge['dst'])

            node_list_t.append(edge['src'])
            node_list_t.append(edge['dst'])

        edge_index_list_t, edge_index_test_list_t = train_test_split(list(map(list, zip(*edge_index_list_t))), test_size=0.3, shuffle=True)
        edge_index_list_t, _ = train_test_split(edge_index_list_t, test_size=noise) if float(noise) != 0.0 else (edge_index_list_t, [])
        edge_index_list_t = list(map(list, zip(*edge_index_list_t)))
        edge_index_test_list_t = list(map(list, zip(*edge_index_test_list_t)))
        edge_index_test_list.append([edge_index_test_list_t[0], edge_index_test_list_t[1]])
        
        edge_index_list.append(edge_index_list_t)
        node_list_t = list(set(node_list_t))
        node_list.append(node_list_t)

    edge_index_list = [[i[0], i[1]] for i in edge_index_list]

    with open(f'data/enron_discrete/enron_discrete_final.pkl', 'wb')as f:
        pickle.dump({'mode': mode, 'node_all_list': node_all_list, 'node_list': node_list, 'edge_index_list': edge_index_list, 'edge_index_test_list': edge_index_test_list}, f)
    return node_all_list, node_list, edge_index_list

def read_data_uci_continuous(m, mode, edge_feat=0):
    with open('data/uci_continuous/uci.json') as f:
        data = json.load(f)
    node_all_list = [i for i in range(data['num_nodes'])]
    edge_index_list = []
    edge_index_test_list = []
    node_list = []
    node_interaction_list = []

    for t, i in enumerate(data['graph']):
        edge_index_list_t = [[] for _ in range(4)]
        node_list_t = []
        for edge in i:
            edge_index_list_t[0].append(edge['src'])
            edge_index_list_t[1].append(edge['dst'])
            edge_index_list_t[2].append(edge['timestamp'])
            if edge_feat != 0:
                edge_index_list_t[3].append(edge['feature'])
            else:
                edge_index_list_t[3].append(0)

            node_list_t.append(edge['src'])
            node_list_t.append(edge['dst'])

        edge_index_list_t, edge_index_test_list_t = train_test_split(list(map(list, zip(*edge_index_list_t))), test_size=0.3)
        edge_index_list_t = list(map(list, zip(*edge_index_list_t)))
        edge_index_test_list_t = list(map(list, zip(*edge_index_test_list_t)))
        edge_index_test_list.append([edge_index_test_list_t[0], edge_index_test_list_t[1]])
        
        edge_index_list.append(edge_index_list_t)

        interaction_history = defaultdict(list)
        src, dst, timestamp, feature = edge_index_list_t
        for u, v, t_n, f_n in zip(src, dst, timestamp, feature):
            interaction_history[u].append((t_n, v, f_n))
            interaction_history[v].append((t_n, u, f_n))


        node_list_t = list(set(node_list_t))
        node_list.append(node_list_t)

        if t != 0:
            for item in edge_index_list[:t]:
                src, dst, timestamp, feature = item
                for u, v, t_n, f_n in zip(src, dst, timestamp, feature):
                    interaction_history[u].append((t_n, v, f_n))
                    interaction_history[v].append((t_n, u, f_n))
        if t == len(data['graph']) - 1:
            edge_index_list[t] = [edge_index_list_t[0], edge_index_list_t[1]]

        node_id_t, time_t, edge_feature_t = get_history_interaction(interaction_history, data['num_nodes'], m)
        if edge_feat != 0:
            node_interaction_list.append({'node_id': node_id_t, 'time': time_t, 'edge_feature': edge_feature_t})
        else:
            node_interaction_list.append({'node_id': node_id_t, 'time': time_t})

    edge_index_list = [[i[0], i[1]] for i in edge_index_list]

    with open(f'data/uci_continuous/uci_continuous_final.pkl', 'wb') as f:
        pickle.dump({'mode': mode, 'node_all_list': node_all_list, 'node_list': node_list, 'edge_index_list': edge_index_list, 'edge_index_test_list': edge_index_test_list, 'node_interaction_list': node_interaction_list}, f)
    return node_all_list, node_list, edge_index_list, node_interaction_list

def read_data_uci_discrete(mode):
    with open('data/uci_discrete/uci.json') as f:
        data = json.load(f)
    node_all_list = [i for i in range(data['num_nodes'])]
    edge_index_list = []
    edge_index_test_list = []
    node_list = []
    for i in data['graph']:
        
        edge_index_list_t = [[] for _ in range(2)]
        node_list_t = []
        for edge in i:
            edge_index_list_t[0].append(edge['src'])
            edge_index_list_t[1].append(edge['dst'])

            node_list_t.append(edge['src'])
            node_list_t.append(edge['dst'])

        edge_index_list_t, edge_index_test_list_t = train_test_split(list(map(list, zip(*edge_index_list_t))), test_size=0.3, shuffle=True)
        edge_index_list_t = list(map(list, zip(*edge_index_list_t)))
        edge_index_test_list_t = list(map(list, zip(*edge_index_test_list_t)))
        edge_index_test_list.append(edge_index_test_list_t)
        
        edge_index_list.append(edge_index_list_t)
        node_list_t = list(set(node_list_t))
        node_list.append(node_list_t)

    edge_index_list = [[i[0], i[1]] for i in edge_index_list]

    with open(f'data/uci_discrete/uci_discrete_final.pkl', 'wb')as f:
        pickle.dump({'mode': mode, 'node_all_list': node_all_list, 'node_list': node_list, 'edge_index_list': edge_index_list, 'edge_index_test_list': edge_index_test_list}, f)
    return node_all_list, node_list, edge_index_list
