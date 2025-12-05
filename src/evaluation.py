import math
import numpy as np


def IDCG(ground_truth, topn):
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    idcg = 0
    for i in range(topn):
        idcg += ((2 ** t[i]) - 1) / math.log(i + 2, 2)
    return idcg


def nDCG(ranked_list, ground_truth, topn):
    dcg = 0
    idcg = IDCG(ground_truth, topn)
    for i in range(topn):
        idx = ranked_list[i]
        dcg += ((2 ** ground_truth[idx]) - 1)/ math.log(i + 2, 2)
    return dcg / idcg


def evaluate_model(pred, label, k1, k2, k3, k4, k5, k6):
    # Below is for nDCG
    ks = [k1, k2, k3, k4, k5, k6]
    y_pred = np.array(pred.cpu().detach().tolist())
    y_true_hot = np.array(label.cpu().detach().tolist())
    ndcg = np.zeros((len(ks), ))
    for i, topn in enumerate(ks):
        for pred2, true_hot in zip(y_pred, y_true_hot):
            ranked_list = np.flip(np.argsort(pred2))
            ndcg[i] += nDCG(ranked_list, true_hot, topn)
    n_list = ndcg / len(y_true_hot)
    metric_n_1 = n_list[0]; metric_n_2 = n_list[1]; metric_n_3 = n_list[2]; metric_n_4 = n_list[3]; metric_n_5 = n_list[4]; metric_n_6 = n_list[5]
    # Below is for precision and recall
    a = np.zeros((len(ks), )); r = np.zeros((len(ks), ))
    for pred2, true_hot in zip(y_pred, y_true_hot):
        pred2 = np.flip(np.argsort(pred2))
        true = np.where(true_hot == 1)[0].tolist()
        t = set(true)
        for i, k in enumerate(ks):
            p = set(pred2[:k])
            it = p.intersection(t)
            a[i] += len(it) / k
            r[i] += len(it) / len(t)
    p_list = a / len(y_true_hot); r_list = r / len(y_true_hot)
    metric_p_1 = p_list[0]; metric_p_2 = p_list[1]; metric_p_3 = p_list[2]; metric_p_4 = p_list[3]; metric_p_5 = p_list[4]; metric_p_6 = p_list[5]
    metric_r_1 = r_list[0]; metric_r_2 = r_list[1]; metric_r_3 = r_list[2]; metric_r_4 = r_list[3]; metric_r_5 = r_list[4]; metric_r_6 = r_list[5]
    return metric_p_1, metric_r_1, metric_n_1, metric_p_2, metric_r_2, metric_n_2, metric_p_3, metric_r_3, metric_n_3, metric_p_4, metric_r_4, metric_n_4, metric_p_5, metric_r_5, metric_n_5, metric_p_6, metric_r_6, metric_n_6

