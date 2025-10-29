import numpy as np
from sklearn import metrics


def compute_all_metrics(conf, label, pred):
    # 设置精度
    np.set_printoptions(precision=3)
    # 设置recall阈值为0.95
    recall = 0.95
    auroc, aupr_in, aupr_out, fpr95, fpr99 = auc_and_fpr_recall(conf, label, recall)
    # 计算准确率
    accuracy = acc(pred, label)

    results = [fpr99, fpr95, auroc, aupr_in, aupr_out, accuracy]

    return results


# accuracy
def acc(pred, label):
    # 计算id样本的分类准确率
    # 过滤掉未标记的样本
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]
    
    # 比较预测值与真实值相等的比例
    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)

    return acc


# fpr_recall，这个函数实际上没有用到，因为我们在计算auroc时已经计算了fpr。
def fpr_recall(conf, label, tpr):
    # 构造二分类标签：ID = 1，OOD = 0
    gt = np.ones_like(label)
    gt[label == -1] = 0
    
    # 使用 ROC 曲线找到 TPR ≥ 0.95 时对应的 FPR 和阈值
    fpr_list, tpr_list, threshold_list = metrics.roc_curve(gt, conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr)]
    thresh = threshold_list[np.argmax(tpr_list >= tpr)]
    
    return fpr, thresh


# auc
def auc_and_fpr_recall(conf, label, tpr_th):
    # following convention in ML we treat OOD as positive
    # 构造二分类标签：ID = 0，OOD = 1
    # 使用 ROC 曲线找到 TPR ≥ 0.95 时对应的 FPR 和阈值
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1
    
    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    # 用 -conf 是因为 postprocessor 默认 ID 样本 conf 较大，而 roc_curve 默认“正类”置信度要大。
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)
    # np.argmax() 函数返回布尔数组中 第一个满足条件的索引
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]
    fpr99 = fpr_list[np.argmax(tpr_list >= 0.99)]

    # 计算 AUPR 时，我们需要将 OOD 样本的置信度取负，因为 AUPR 是针对正类的。
    # AUPR_IN：ID 样本为正类，conf 高是好事。
    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(1 - ood_indicator, conf)

    # AUPR_OUT：OOD 样本为正类，conf 越低越好 → 所以用了 -conf。
    precision_out, recall_out, thresholds_out \
        = metrics.precision_recall_curve(ood_indicator, -conf)

    # 三个指标用 metrics.auc 分别计算出对应面积。
    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, fpr, fpr99