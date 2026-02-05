import warnings
import argparse
import torch
from torch.utils.data import DataLoader
import tracemalloc
import time
import json

from process.data import read_data
import process.node_label as node_label
import process.edge_label as edge_label
from util import DatasetSample, generate_samples, collate_fn
from model import InvarTG



if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='InvarTG')

    parser.add_argument('--graph_type', type=str, default='continuous', choices=['discrete', 'continuous'])
    parser.add_argument('--data', type=str, default='wikipedia_continuous', choices=['wikipedia_continuous', 'wikipedia_discrete', 'enron_continuous', 'enron_discrete', 'uci_continuous', 'uci_discrete', 'reddit_continuous', 'reddit_discrete'])
    parser.add_argument('--mode', type=str, default='link_prediction', choices=['node_classification', 'link_prediction'])

    parser.add_argument('--ablation', type=str, default='none', choices=['none', 'no_g', 'no_F', 'no_G'])
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--edge_feat', type=int, default=20)
    
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--sample_percent', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save_result', type=int, default=1, choices=[0, 1])

    parser.add_argument('--walk_length', type=int, default=10)
    parser.add_argument('--num_walks', type=int, default=5)
    parser.add_argument('--num_neg', type=int, default=5)

    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--lambda_', type=float, default=1)
    parser.add_argument('--window_size', type=int, default=3)

    parser.add_argument('--in_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=2)


    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'

    print('开始加载数据...初次加载将较慢')
    if args.graph_type == 'continuous':
        node_all_list, node_list, edge_index_list, node_interaction_list = read_data(args.data, args.graph_type, args.m, args.mode, args.edge_feat, args.noise)
        edge_index_list = [torch.tensor(edge_index_list_t, dtype=torch.long, device=device) for edge_index_list_t in edge_index_list]
        if args.edge_feat != 0:
            node_interaction_list = [{
                'node_id': torch.tensor(node_interaction_list_t['node_id'], dtype=torch.long, device=device),
                'time': torch.tensor(node_interaction_list_t['time'], dtype=torch.long, device=device),
                'edge_feature': torch.tensor(node_interaction_list_t['edge_feature'], dtype=torch.float, device=device)
                } for node_interaction_list_t in node_interaction_list]
        else:
            node_interaction_list = [{
                'node_id': torch.tensor(node_interaction_list_t['node_id'], dtype=torch.long, device=device),
                'time': torch.tensor(node_interaction_list_t['time'], dtype=torch.long, device=device)
                } for node_interaction_list_t in node_interaction_list]
    elif args.graph_type == 'discrete':
        node_all_list, node_list, edge_index_list = read_data(args.data, args.graph_type, mode=args.mode)
        edge_index_list = [torch.tensor(edge_index_list_t, dtype=torch.long, device=device) for edge_index_list_t in edge_index_list]
        node_interaction_list = None



    print('加载数据完成！')

    print('开始加载随机游走样本...初次加载将较慢')
    samples = generate_samples(args.data, 
                               node_all_list, 
                               node_list,
                               edge_index_list,
                               args.device,
                               args.walk_length, 
                               args.window_size,
                               args.num_walks, 
                               args.num_neg)
    samples_load = DatasetSample(samples, args.sample_percent)
    samples_batch = DataLoader(samples_load, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    print('随机游走样本加载完成！')
    if args.ablation == 'none':
        print('准备运行全量模型...')
        if args.mode == 'node_classification':
            tracemalloc.start()

            print('开始加载不变任务（节点分类）样本...')
            train_label, test_label = node_label.read_label(args.data, args.device)
            print('不变任务样本（节点分类）加载完成！')

            print('开始训练模型...')
            print('初始化模型中...')
            model = InvarTG(len(node_all_list), args.in_dim, args.hidden_dim, args.m, args.graph_type, args.out_dim, args.lr, args.device, args.mode, args.data, args.edge_feat)
            model.to(args.device)
            loss_total = 0

            list_precision_micro = []
            list_recall_micro = []
            list_f1_micro = []
            list_precision_macro = []
            list_recall_macro = []
            list_f1_macro = []

            list_precision_micro_all = []
            list_recall_micro_all = []
            list_f1_micro_all = []
            list_precision_macro_all = []
            list_recall_macro_all = []
            list_f1_macro_all = []
            
            list_time = []

            print('逐Epoch训练中...')
            for epoch in range(args.epochs):
                time_start = time.time()

                model.train()
                model.train_g(node_list, edge_index_list, samples_batch, node_interaction_list)
                model.train_F(node_list, edge_index_list, train_label, node_interaction_list, lambda_=args.lambda_)
                loss = model.train_G(node_list, edge_index_list, samples_batch, node_interaction_list)
                loss_total += loss
                epoch += 1

                time_end = time.time()
                list_time.append(time_end - time_start)

                if epoch % 10 == 0:
                    print('Epoch: %d, Loss: %.4f' % (epoch, loss_total / 10))
                    loss_total = 0 
                    
                model.eval()
                precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro , precision_micro_all, recall_micro_all, f1_micro_all, precision_macro_all, recall_macro_all, f1_macro_all = model.eval_F(node_list, edge_index_list, test_label, node_interaction_list)

                list_precision_micro.append(precision_micro)
                list_recall_micro.append(recall_micro)
                list_f1_micro.append(f1_micro)
                list_precision_macro.append(precision_macro)
                list_recall_macro.append(recall_macro)
                list_f1_macro.append(f1_macro)

                list_precision_micro_all.append(precision_micro_all)
                list_recall_micro_all.append(recall_micro_all)
                list_f1_micro_all.append(f1_micro_all)
                list_precision_macro_all.append(precision_macro_all)
                list_recall_macro_all.append(recall_macro_all)
                list_f1_macro_all.append(f1_macro_all)

                print(f1_micro)

            print('模型训练与测试完成！')

            max_f1_micro = max(list_f1_micro)
            index_max = list_precision_micro.index(max_f1_micro)
            _, peak_memory = tracemalloc.get_traced_memory()
            
            print(f'内存占用峰值为: {peak_memory / 1024 / 1024:.1f} MB')
            # print(f'GPU开销峰值为: {monitor.get_peak_memory():.1f} MB')
            print(f'平均每Epoch耗时: {sum(list_time) / len(list_time):.4f} 秒')
            print(f'Precision-Micro: {list_precision_micro[index_max]:.4f}')
            print(f'Recall-Micro: {list_recall_micro[index_max]:.4f}')
            print(f'F1-Micro: {list_f1_micro[index_max]:.4f}')
            print(f'Precision-Macro: {list_precision_macro[index_max]:.4f}')
            print(f'Recall-Macro: {list_recall_macro[index_max]:.4f}')
            print(f'F1-Macro: {list_f1_macro[index_max]:.4f}')

            if args.save_result:
                print('正在保存测试结果...')
                with open(f'result/{args.data}_{args.mode}_result.json', 'w') as f:
                    json.dump({
                            'result': {
                                'Precision-Micro': list_precision_micro[index_max],
                                'Recall-Micro': list_recall_micro[index_max],
                                'F1-Micro': list_f1_micro[index_max],
                                'Precision-Macro': list_precision_macro[index_max],
                                'Recall-Macro': list_recall_macro[index_max],
                                'F1-Macro': list_f1_macro[index_max],

                                'Precision-Micro-All': list_precision_micro_all[index_max],
                                'Recall-Micro-All': list_recall_micro_all[index_max],
                                'F1-Micro-All': list_f1_micro_all[index_max],
                                'Precision-Macro-All': list_precision_macro_all[index_max],
                                'Recall-Macro-All': list_recall_macro_all[index_max],
                                'F1-Macro-All': list_f1_macro_all[index_max],
                            
                                'Average-Time Per Epoch': f'{sum(list_time) / len(list_time):.4f} 秒',
                                'Peak Memory': f'{peak_memory / 1024 / 1024:.1f} MB'
                            },
                            'args': args.__dict__
                        }, f, indent=4, ensure_ascii=False)
                print('测试结果已保存！')

        elif args.mode == 'link_prediction':
            tracemalloc.start()
            print('开始加载不变任务（链接预测）样本...')
            train_label, test_label = edge_label.read_label(args.data)
            print('不变任务样本（链接预测）加载完成！')

            print('开始训练模型...')
            print('初始化模型中...')
            model = InvarTG(len(node_all_list), args.in_dim, args.hidden_dim, args.m, args.graph_type, args.out_dim, args.lr, args.device, args.mode, args.data)
            model.to(args.device)
            loss_total = 0

            list_auc_roc = []
            list_ap = []

            list_roc_all = []
            list_auc_roc_all = []
            list_ap_all = []

            list_time = []

            for epoch in range(args.epochs):
                time_start = time.time()
                model.train()
                model.train_g(node_list, edge_index_list, samples_batch, node_interaction_list)
                model.train_F(node_list, edge_index_list, train_label, node_interaction_list, lambda_=args.lambda_)
                loss = model.train_G(node_list, edge_index_list, samples_batch, node_interaction_list)
                loss_total += loss
                epoch += 1

                time_end = time.time()
                list_time.append(time_end - time_start)

                if epoch % 10 == 0:
                    print('Epoch: %d, Loss: %.4f' % (epoch, loss_total / 10))
                    loss_total = 0
                
                model.eval()
                auc_roc, ap, roc_all, auc_roc_all, ap_all = model.eval_F(node_list, edge_index_list, test_label, node_interaction_list)
                list_auc_roc.append(auc_roc)
                list_ap.append(ap)

                print(auc_roc)


                list_roc_all.append(roc_all)
                list_auc_roc_all.append(auc_roc_all)
                list_ap_all.append(ap_all)
                # print(f'GPU开销峰值为: {peak_mem:.1f} MB')

            print('模型训练与测试完成！')

            max_auc_roc = max(list_auc_roc)
            index_max = list_auc_roc.index(max_auc_roc)
            _, peak_memory = tracemalloc.get_traced_memory()

            print(f'内存占用峰值为: {peak_memory / 1024 / 1024:.1f} MB')
            # print(f'GPU开销峰值为: {monitor.get_peak_memory():.1f} MB')
            print(f'平均每Epoch耗时: {sum(list_time) / len(list_time):.4f} 秒')
            print(f'AUC-ROC: {list_auc_roc[index_max]:.4f}')
            print(f'AP: {list_ap[index_max]:.4f}')

            if args.save_result:
                print('正在保存测试结果...')
                with open(f'result/{args.data}_{args.mode}_result.json', 'w') as f:
                    json.dump({
                            'result': {
                                'AUC-ROC': list_auc_roc[index_max],
                                'AP': list_ap[index_max],

                                'ROC-All': list_roc_all[index_max],
                                'AUC-ROC-All': list_auc_roc_all[index_max],
                                'AP-All': list_ap_all[index_max],

                                'Average-Time Per Epoch': f'{sum(list_time) / len(list_time):.4f} 秒',
                                'Peak Memory': f'{peak_memory / 1024 / 1024:.1f} MB'
                            },
                            'args': args.__dict__
                        }, f, indent=4, ensure_ascii=False)
                print('测试结果已保存！')
    elif args.ablation == 'no_g':
        print('准备运行消融模型：替换g为TGAT...')
        if args.mode == 'node_classification':

            print('开始加载不变任务（节点分类）样本...')
            train_label, test_label = node_label.read_label(args.data, args.device)
            print('不变任务样本（节点分类）加载完成！')

            print('开始训练模型...')
            print('初始化模型中...')
            model = InvarTG(len(node_all_list), args.in_dim, args.hidden_dim, args.m, args.graph_type, args.out_dim, args.lr, args.device, args.mode, args.data, args.edge_feat, args.ablation)            
            model.to(args.device)
            loss_total = 0

            list_precision_micro = []
            list_recall_micro = []
            list_f1_micro = []
            list_precision_macro = []
            list_recall_macro = []
            list_f1_macro = []

            list_precision_micro_all = []
            list_recall_micro_all = []
            list_f1_micro_all = []
            list_precision_macro_all = []
            list_recall_macro_all = []
            list_f1_macro_all = []

            print('逐Epoch训练中...')
            for epoch in range(args.epochs):
                model.train()
                model.ablation_g_g(node_list, edge_index_list, samples_batch)
                model.train_F(node_list, edge_index_list, train_label, lambda_=args.lambda_)
                loss = model.ablation_g_G(node_list, edge_index_list, samples_batch)

                loss_total += loss
                epoch += 1

                if epoch % 10 == 0:
                    print('Epoch: %d, Loss: %.4f' % (epoch, loss_total / 10))
                    loss_total = 0 

                model.eval()
                precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro , precision_micro_all, recall_micro_all, f1_micro_all, precision_macro_all, recall_macro_all, f1_macro_all = model.eval_ablation_g(node_list, edge_index_list, test_label)

                list_precision_micro.append(precision_micro)
                list_recall_micro.append(recall_micro)
                list_f1_micro.append(f1_micro)
                list_precision_macro.append(precision_macro)
                list_recall_macro.append(recall_macro)
                list_f1_macro.append(f1_macro)

                list_precision_micro_all.append(precision_micro_all)
                list_recall_micro_all.append(recall_micro_all)
                list_f1_micro_all.append(f1_micro_all)
                list_precision_macro_all.append(precision_macro_all)
                list_recall_macro_all.append(recall_macro_all)
                list_f1_macro_all.append(f1_macro_all)

                print(precision_micro)

            print('模型训练与测试完成！')

            max_f1_micro = max(list_f1_micro)
            index_max = list_f1_micro.index(max_f1_micro)

            print(f'Precision-Micro: {list_precision_micro[index_max]:.4f}')
            print(f'Recall-Micro: {list_recall_micro[index_max]:.4f}')
            print(f'F1-Micro: {list_f1_micro[index_max]:.4f}')
            print(f'Precision-Macro: {list_precision_macro[index_max]:.4f}')
            print(f'Recall-Macro: {list_recall_macro[index_max]:.4f}')
            print(f'F1-Macro: {list_f1_macro[index_max]:.4f}')

            if args.save_result:
                print('正在保存测试结果...')
                with open(f'result/{args.data}_{args.mode}_ablation_g_result.json', 'w') as f:
                    json.dump({
                            'result': {
                                'Precision-Micro': list_precision_micro[index_max],
                                'Recall-Micro': list_recall_micro[index_max],
                                'F1-Micro': list_f1_micro[index_max],
                                'Precision-Macro': list_precision_macro[index_max],
                                'Recall-Macro': list_recall_macro[index_max],
                                'F1-Macro': list_f1_macro[index_max],

                                'Precision-Micro-All': list_precision_micro_all[index_max],
                                'Recall-Micro-All': list_recall_micro_all[index_max],
                                'F1-Micro-All': list_f1_micro_all[index_max],
                                'Precision-Macro-All': list_precision_macro_all[index_max],
                                'Recall-Macro-All': list_recall_macro_all[index_max],
                                'F1-Macro-All': list_f1_macro_all[index_max],
                            }
                        }, f, indent=4)
                print('测试结果已保存！')
        elif args.mode == 'link_prediction':

            print('开始加载不变任务（链接预测）样本...')
            train_label, test_label = edge_label.read_label(args.data)
            print('不变任务样本（链接预测）加载完成！')

            print('开始训练模型...')
            print('初始化模型中...')
            model = InvarTG(len(node_all_list), args.in_dim, args.hidden_dim, args.m, args.graph_type, args.out_dim, args.lr, args.device, args.mode, args.data, args.edge_feat, args.ablation)            
            model.to(args.device)
            loss_total = 0

            list_auc_roc = []
            list_ap = []

            list_auc_roc_all = []
            list_ap_all = []

            for epoch in range(args.epochs):

                model.train()
                model.ablation_g_g(node_list, edge_index_list, samples_batch)
                model.train_F(node_list, edge_index_list, train_label, lambda_=args.lambda_)
                loss = model.ablation_g_G(node_list, edge_index_list, samples_batch)

                loss_total += loss
                epoch += 1

                if epoch % 10 == 0:
                    print('Epoch: %d, Loss: %.4f' % (epoch, loss_total / 10))
                    loss_total = 0

                model.eval()
                auc_roc, ap, _, auc_roc_all, ap_all = model.eval_ablation_g(node_list, edge_index_list, test_label)

                print(auc_roc)
                list_auc_roc.append(auc_roc)
                list_ap.append(ap)

                list_auc_roc_all.append(auc_roc_all)
                list_ap_all.append(ap_all)
        
            print('模型训练与测试完成！')

            max_auc_roc = max(list_auc_roc)
            index_max = list_auc_roc.index(max_auc_roc)

            print(f'AUC-ROC: {list_auc_roc[index_max]:.4f}')
            print(f'AP: {list_ap[index_max]:.4f}')

            if args.save_result:
                print('正在保存测试结果...')
                with open(f'result/{args.data}_{args.mode}_ablation_g_result.json', 'w') as f:
                    json.dump({
                            'result': {
                                'AUC-ROC': list_auc_roc[index_max],
                                'AP': list_ap[index_max],

                                'AUC-ROC-All': list_auc_roc_all[index_max],
                                'AP-All': list_ap_all[index_max],
                            },
                        }, f, indent=4, ensure_ascii=False)
                print('测试结果已保存！')
    elif args.ablation == 'no_F':
        print('准备运行消融模型：替换不变任务损失为交叉熵损失...')
        if args.mode == 'node_classification':
            
            print('开始加载不变任务（节点分类）样本...')
            train_label, test_label = node_label.read_label(args.data, args.device)
            print('不变任务样本（节点分类）加载完成！')

            print('开始训练模型...')
            print('初始化模型中...')
            model = InvarTG(len(node_all_list), args.in_dim, args.hidden_dim, args.m, args.graph_type, args.out_dim, args.lr, args.device, args.mode, args.data, args.edge_feat, args.ablation)
            model.to(args.device)
            loss_total = 0

            list_precision_micro = []
            list_recall_micro = []
            list_f1_micro = []
            list_precision_macro = []
            list_recall_macro = []
            list_f1_macro = []

            list_precision_micro_all = []
            list_recall_micro_all = []
            list_f1_micro_all = []
            list_precision_macro_all = []
            list_recall_macro_all = []
            list_f1_macro_all = []

            print('逐Epoch训练中...')
            for epoch in range(args.epochs):

                model.train()
                model.train_g(node_list, edge_index_list, samples_batch, node_interaction_list)
                model.ablation_F_F(node_list, edge_index_list, train_label, node_interaction_list)
                loss = model.train_G(node_list, edge_index_list, samples_batch, node_interaction_list)
                
                loss_total += loss
                epoch += 1
                if epoch % 10 == 0:
                    print('Epoch: %d, Loss: %.4f' % (epoch, loss_total / 10))
                    loss_total = 0 

                model.eval()
                precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro , precision_micro_all, recall_micro_all, f1_micro_all, precision_macro_all, recall_macro_all, f1_macro_all = model.eval_F(node_list, edge_index_list, test_label, node_interaction_list)

                list_precision_micro.append(precision_micro)
                list_recall_micro.append(recall_micro)
                list_f1_micro.append(f1_micro)
                list_precision_macro.append(precision_macro)
                list_recall_macro.append(recall_macro)
                list_f1_macro.append(f1_macro)

                list_precision_micro_all.append(precision_micro_all)
                list_recall_micro_all.append(recall_micro_all)
                list_f1_micro_all.append(f1_micro_all)
                list_precision_macro_all.append(precision_macro_all)
                list_recall_macro_all.append(recall_macro_all)
                list_f1_macro_all.append(f1_macro_all)

                print(f1_micro)

            print('模型训练与测试完成！')

            max_f1_micro = max(list_f1_micro)
            index_max = list_f1_micro.index(max_f1_micro)

            print(f'Precision-Micro: {list_precision_micro[index_max]:.4f}')
            print(f'Recall-Micro: {list_recall_micro[index_max]:.4f}')
            print(f'F1-Micro: {list_f1_micro[index_max]:.4f}')
            print(f'Precision-Macro: {list_precision_macro[index_max]:.4f}')
            print(f'Recall-Macro: {list_recall_macro[index_max]:.4f}')
            print(f'F1-Macro: {list_f1_macro[index_max]:.4f}')

            if args.save_result:
                print('正在保存测试结果...')
                with open(f'result/{args.data}_{args.mode}_ablation_F_result.json', 'w') as f:
                    json.dump({
                            'result': {
                                'Precision-Micro': list_precision_micro[index_max],
                                'Recall-Micro': list_recall_micro[index_max],
                                'F1-Micro': list_f1_micro[index_max],
                                'Precision-Macro': list_precision_macro[index_max],
                                'Recall-Macro': list_recall_macro[index_max],
                                'F1-Macro': list_f1_macro[index_max],

                                'Precision-Micro-All': list_precision_micro_all[index_max],
                                'Recall-Micro-All': list_recall_micro_all[index_max],
                                'F1-Micro-All': list_f1_micro_all[index_max],
                                'Precision-Macro-All': list_precision_macro_all[index_max],
                                'Recall-Macro-All': list_recall_macro_all[index_max],
                                'F1-Macro-All': list_f1_macro_all[index_max],
                            }
                        }, f, indent=4)
                print('测试结果已保存！')

        elif args.mode == 'link_prediction':

            print('开始加载不变任务（链接预测）样本...')
            train_label, test_label = edge_label.read_label(args.data)
            print('不变任务样本（链接预测）加载完成！')

            print('开始训练模型...')
            print('初始化模型中...')
            model = InvarTG(len(node_all_list), args.in_dim, args.hidden_dim, args.m, args.graph_type, args.out_dim, args.lr, args.device, args.mode, args.data, args.edge_feat, args.ablation)
            model.to(args.device)
            loss_total = 0

            list_auc_roc = []
            list_ap = []

            list_auc_roc_all = []
            list_ap_all = []

            for epoch in range(args.epochs):
                
                model.train()
                model.train_g(node_list, edge_index_list, samples_batch, node_interaction_list)
                model.ablation_F_F(node_list, edge_index_list, train_label, node_interaction_list)
                loss = model.train_G(node_list, edge_index_list, samples_batch, node_interaction_list)

                loss_total += loss
                epoch += 1

                if epoch % 10 == 0:
                    print('Epoch: %d, Loss: %.4f' % (epoch, loss_total / 10))
                    loss_total = 0 

                model.eval()
                auc_roc, ap, _, auc_roc_all, ap_all = model.eval_F(node_list, edge_index_list, test_label, node_interaction_list)

                list_auc_roc.append(auc_roc)
                list_ap.append(ap)

                list_auc_roc_all.append(auc_roc_all)
                list_ap_all.append(ap_all)

                print(auc_roc)

            print('模型训练与测试完成！')

            max_auc_roc = max(list_auc_roc)
            index_max = list_auc_roc.index(max_auc_roc)

            print(f'AUC-ROC: {list_auc_roc[index_max]:.4f}')
            print(f'AP: {list_ap[index_max]:.4f}')

            if args.save_result:
                print('正在保存测试结果...')
                with open(f'result/{args.data}_{args.mode}_ablation_F_result.json', 'w') as f:
                    json.dump({
                            'result': {
                                'AUC-ROC': list_auc_roc[index_max],
                                'AP': list_ap[index_max],

                                'AUC-ROC-All': list_auc_roc_all[index_max],
                                'AP-All': list_ap_all[index_max],
                            },
                        }, f, indent=4, ensure_ascii=False)
                print('测试结果已保存！')
    elif args.ablation == 'no_G':
        print('准备运行消融模型：删除L_G...')
        if args.mode == 'node_classification':

            print('开始加载不变任务（节点分类）样本...')
            train_label, test_label = node_label.read_label(args.data, args.device)
            print('不变任务样本（节点分类）加载完成！')

            print('开始训练模型...')
            print('初始化模型中...')
            model = InvarTG(len(node_all_list), args.in_dim, args.hidden_dim, args.m, args.graph_type, args.out_dim, args.lr, args.device, args.mode, args.data, args.edge_feat, args.ablation)
            model.to(args.device)
            loss_total = 0

            list_precision_micro = []
            list_recall_micro = []
            list_f1_micro = []
            list_precision_macro = []
            list_recall_macro = []
            list_f1_macro = []

            list_precision_micro_all = []
            list_recall_micro_all = []
            list_f1_micro_all = []
            list_precision_macro_all = []
            list_recall_macro_all = []
            list_f1_macro_all = []

            print('逐Epoch训练中...')
            for epoch in range(args.epochs):

                model.train()
                loss = model.train_g(node_list, edge_index_list, samples_batch, node_interaction_list)
                model.train_F(node_list, edge_index_list, train_label, node_interaction_list, lambda_=args.lambda_)
                loss_total += loss
                epoch += 1

                if epoch % 10 == 0:
                    print('Epoch: %d, Loss: %.4f' % (epoch, loss_total / 10))
                    loss_total = 0 
                    
                model.eval()
                precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro , precision_micro_all, recall_micro_all, f1_micro_all, precision_macro_all, recall_macro_all, f1_macro_all = model.eval_F(node_list, edge_index_list, test_label, node_interaction_list)

                list_precision_micro.append(precision_micro)
                list_recall_micro.append(recall_micro)
                list_f1_micro.append(f1_micro)
                list_precision_macro.append(precision_macro)
                list_recall_macro.append(recall_macro)
                list_f1_macro.append(f1_macro)

                list_precision_micro_all.append(precision_micro_all)
                list_recall_micro_all.append(recall_micro_all)
                list_f1_micro_all.append(f1_micro_all)
                list_precision_macro_all.append(precision_macro_all)
                list_recall_macro_all.append(recall_macro_all)
                list_f1_macro_all.append(f1_macro_all)

                print(f1_micro)

            print('模型训练与测试完成！')

            max_f1_micro = max(list_f1_micro)
            index_max = list_precision_micro.index(max_f1_micro)
            
            print(f'Precision-Micro: {list_precision_micro[index_max]:.4f}')
            print(f'Recall-Micro: {list_recall_micro[index_max]:.4f}')
            print(f'F1-Micro: {list_f1_micro[index_max]:.4f}')
            print(f'Precision-Macro: {list_precision_macro[index_max]:.4f}')
            print(f'Recall-Macro: {list_recall_macro[index_max]:.4f}')
            print(f'F1-Macro: {list_f1_macro[index_max]:.4f}')

            if args.save_result:
                print('正在保存测试结果...')
                with open(f'result/{args.data}_{args.mode}_ablation_G_result.json', 'w') as f:
                    json.dump({
                            'result': {
                                'Precision-Micro': list_precision_micro[index_max],
                                'Recall-Micro': list_recall_micro[index_max],
                                'F1-Micro': list_f1_micro[index_max],
                                'Precision-Macro': list_precision_macro[index_max],
                                'Recall-Macro': list_recall_macro[index_max],
                                'F1-Macro': list_f1_macro[index_max],

                                'Precision-Micro-All': list_precision_micro_all[index_max],
                                'Recall-Micro-All': list_recall_micro_all[index_max],
                                'F1-Micro-All': list_f1_micro_all[index_max],
                                'Precision-Macro-All': list_precision_macro_all[index_max],
                                'Recall-Macro-All': list_recall_macro_all[index_max],
                                'F1-Macro-All': list_f1_macro_all[index_max],
                            },
                        }, f, indent=4)
                print('测试结果已保存！')

        elif args.mode == 'link_prediction':

            print('开始加载不变任务（链接预测）样本...')
            train_label, test_label = edge_label.read_label(args.data)
            print('不变任务样本（链接预测）加载完成！')

            print('开始训练模型...')
            print('初始化模型中...')
            model = InvarTG(len(node_all_list), args.in_dim, args.hidden_dim, args.m, args.graph_type, args.out_dim, args.lr, args.device, args.mode, args.data)
            model.to(args.device)
            loss_total = 0

            list_auc_roc = []
            list_ap = []

            list_auc_roc_all = []
            list_ap_all = []

            list_time = []

            for epoch in range(args.epochs):

                model.train()
                loss = model.train_g(node_list, edge_index_list, samples_batch, node_interaction_list)
                model.train_F(node_list, edge_index_list, train_label, node_interaction_list, lambda_=args.lambda_)
                loss_total += loss
                epoch += 1

                if epoch % 10 == 0:
                    print('Epoch: %d, Loss: %.4f' % (epoch, loss_total / 10))
                    loss_total = 0
                
                model.eval()
                auc_roc, ap, roc_all, auc_roc_all, ap_all = model.eval_F(node_list, edge_index_list, test_label, node_interaction_list)
                
                list_auc_roc.append(auc_roc)
                list_ap.append(ap)

                list_auc_roc_all.append(auc_roc_all)
                list_ap_all.append(ap_all)

                print(auc_roc)

            print('模型训练与测试完成！')

            max_auc_roc = max(list_auc_roc)
            index_max = list_auc_roc.index(max_auc_roc)

            print(f'AUC-ROC: {list_auc_roc[index_max]:.4f}')
            print(f'AP: {list_ap[index_max]:.4f}')

            if args.save_result:
                print('正在保存测试结果...')
                with open(f'result/{args.data}_{args.mode}_ablation_G_result.json', 'w') as f:
                    json.dump({
                            'result': {
                                'AUC-ROC': list_auc_roc[index_max],
                                'AP': list_ap[index_max],

                                'AUC-ROC-All': list_auc_roc_all[index_max],
                                'AP-All': list_ap_all[index_max],
                            }
                        }, f, indent=4)
                print('测试结果已保存！')
