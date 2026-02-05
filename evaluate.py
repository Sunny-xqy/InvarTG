import torch
from sklearn.metrics import precision_score, recall_score, f1_score, auc, roc_curve, average_precision_score

def evaluate_node_classification(y_hat, y_true):
    '''
    节点分类任务的评估函数（F1，PR，RC）
    
    :param y_hat: Description
    :param y_true: Description
    '''
    for i in range(len(y_hat)):
        y_hat[i] = y_hat[i][y_true[i][0]]
        y_true[i] = y_true[i][1]

    list_precision_micro = []
    list_recall_micro = []
    list_f1_micro = []

    list_precision_macro = []
    list_recall_macro = []
    list_f1_macro = []

    for y_hat_t, y_true_t in zip(y_hat, y_true):
        y_hat_t = torch.argmax(y_hat_t, dim=1).detach().cpu().numpy()
        y_true_t = y_true_t.detach().cpu().numpy()
        precision_micro = precision_score(y_true_t, y_hat_t, average='micro')
        recall_micro = recall_score(y_true_t, y_hat_t, average='micro')
        f1_micro = f1_score(y_true_t, y_hat_t, average='micro')

        list_precision_micro.append(precision_micro)
        list_recall_micro.append(recall_micro)
        list_f1_micro.append(f1_micro)

        precision_macro = precision_score(y_true_t, y_hat_t, average='macro')
        recall_macro = recall_score(y_true_t, y_hat_t, average='macro')
        f1_macro = f1_score(y_true_t, y_hat_t, average='macro')
        
        list_f1_macro.append(f1_macro)
        list_precision_macro.append(precision_macro)
        list_recall_macro.append(recall_macro)
    
    return [x / len(y_hat) for x in (sum(list_precision_micro), sum(list_recall_micro), sum(list_f1_micro), sum(list_precision_macro), sum(list_recall_macro), sum(list_f1_macro))] + [list_precision_micro, list_recall_micro, list_f1_micro, list_precision_macro, list_recall_macro, list_f1_macro]


def evaluate_link_prediction(y_hat, y_true):
    '''
    链接预测任务的评估函数（AUC-ROC，AP）
    
    :param y_hat: Description
    :param y_true: Description
    '''

    list_roc = []
    list_auc_roc = []
    list_ap = []

    for y_hat_t, y_true_t in zip(y_hat, y_true):
        # y_hat_t = torch.argmax(y_hat_t, dim=1).detach().cpu().numpy()
        y_hat_t = torch.max(y_hat_t, dim=1).values.detach().cpu().numpy()
        # y_hat_t = torch.softmax(y_hat_t, dim=1)[:, 1].detach().cpu().numpy()
        y_true_t = y_true_t.detach().cpu().numpy()

        fpr, tpr, _ = roc_curve(y_true_t, y_hat_t)
        auc_roc = float(auc(fpr, tpr))
        ap = average_precision_score(y_true_t, y_hat_t)

        list_roc.append([fpr.tolist(), tpr.tolist()])
        list_auc_roc.append(auc_roc)
        list_ap.append(ap)

    return [x / len(y_hat) for x in (sum(list_auc_roc), sum(list_ap))] + [list_roc, list_auc_roc, list_ap]