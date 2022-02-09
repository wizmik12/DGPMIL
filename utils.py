import numpy as np
import collections
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, \
cohen_kappa_score, f1_score, accuracy_score, precision_score, recall_score, average_precision_score
import torch
torch.manual_seed(10)

def bag_labels(bag_index, inst_labels):
    bags = np.unique(bag_index)
    labels = [inst_labels[np.where(bag_index==b)][0] for b in bags]
    return np.array(labels)

def index_bags(bags):
    index = np.array([])
    j=0
    for x in bags:
        index = np.hstack((index, np.array(len(x)*[j])))
        j+=1
    return index

def inst_bag_labels(bags, bag_labels):
    labels = np.array([])
    for x,y in zip(bags,bag_labels):
        labels = np.hstack((labels,len(x)*[y]))
    return labels

def predict_bags(predictions, bag_index_test):
    pred_bags = np.zeros(len(np.unique(bag_index_test)))
    for ib, b in enumerate(np.unique(bag_index_test)):
        preds = predictions[bag_index_test==b]
        pred_bags[int(ib)] = np.max(preds)
    return np.array(pred_bags)

def bag_metrics(model, X, y_true, index, subset):
    predictive_means, _, _ = model.predict(X)
    prob = predictive_means.mean(0).cpu().numpy()

    pred_bags = predict_bags(prob, index)
    auc = roc_auc_score(y_true, pred_bags)
    pr = average_precision_score(y_true, pred_bags)
    ll = log_loss(y_true, pred_bags)
    pred = np.zeros(len(y_true))
    pred[pred_bags>0.5] = 1
    acc = np.mean(pred==y_true)

    # keys
    auc_key = 'AUC_' + subset
    acc_key = 'acc_' + subset
    pr_key = 'PR_' + subset
    ll_key = 'LL_' + subset

    return {acc_key: acc, auc_key: auc, pr_key: pr, ll_key: ll}

def bag_metrics_CT(model, X, y_true, index, subset):
    predictive_means, _, _ = model.predict(X)
    prob = predictive_means.mean(0).cpu().numpy()

    pred_bags = predict_bags(prob, index)
    pred = np.zeros(len(y_true))
    pred[pred_bags>=0.5] = 1

    kappa = cohen_kappa_score(y_true, pred)
    pr = precision_score(y_true, pred)
    recall = recall_score(y_true, pred)
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred)
    auc = roc_auc_score(y_true, pred_bags)
    pr = average_precision_score(y_true, pred_bags)

    # keys
    bag = 'bag_'
    kappa_key = bag + 'kappa_' + subset
    auc_key = bag + 'AUC_' + subset
    acc_key = bag + 'acc_' + subset
    pr_key = bag + 'pr_' + subset
    rec_key = bag + 'rec_' + subset
    f1_key = bag + 'f1_' + subset
    pr_auc_key = bag + 'pr_auc_' + subset

    dict_results = {acc_key: acc, kappa_key: kappa, pr_key: pr, rec_key: recall,
    f1_key: f1, auc_key: auc, pr_auc_key: pr}

    return dict_results, pred_bags

def instance_metrics_CT(model, X, y_true, subset):
    # predictions
    predictive_means, _, _ = model.predict(X)
    prob = predictive_means.mean(0).cpu().numpy()
    pred = np.zeros(len(prob))
    pred[prob>=0.5] = 1

    # metrics
    y_true = y_true.cpu().numpy()

    kappa = cohen_kappa_score(y_true, pred)
    pr = precision_score(y_true, pred)
    recall = recall_score(y_true, pred)
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred)
    auc = roc_auc_score(y_true, prob)

    # keys
    kappa_key = 'kappa_' + subset
    auc_key = 'AUC_' + subset
    acc_key = 'acc_' + subset
    pr_key = 'pr_' + subset
    rec_key = 'rec_' + subset
    f1_key = 'f1_' + subset

    dict_results = {acc_key: acc, kappa_key: kappa, pr_key: pr, rec_key: recall,
    f1_key: f1, auc_key: auc}

    return dict_results, prob
