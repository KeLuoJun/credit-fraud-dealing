from sklearn.metrics import (
    accuracy_score, recall_score, 
    precision_score, f1_score, roc_auc_score
)
import pandas as pd
from config import CONFIG
from typing import Dict

def weighted_score_auto(metrics: Dict, weights: Dict = None):
    """
    根据一个或多个评估指标计算加权得分（支持单指标直接返回）

    Parameters
    ----------
    metrics: Dict
        评估指标得分， 例如 {'accuracy': 0.9, 'f1': 0.85}

    weights: Dict, optional
        每个指标的权重字典，例如 {'accuracy': 0.5, 'f1': 0.5}
        如果未提供，则默认等权重平均。

    Returns
    -------
    float
        加权得分或单指标得分。
    """
    if not metrics:
        raise ValueError("metrics 字典不能为空")
    
    if len(metrics) == 1:
        return next(iter(metrics.values()))
    
    if weights:
        return sum(metrics.get(k, 0) * weights.get(k, 0) for k in weights)
    else:
        return sum(metrics.values()) / len(metrics)


def eval_metric_data(model, X_train, y_train, X_test, y_test,
                     metrics=[
                         accuracy_score,
                         recall_score,
                         precision_score,
                         f1_score,
                         roc_auc_score
                     ],
                     weights=CONFIG.METRIC_WEIGHTS):
    """
    评估模型在训练集和测试集上的各项指标，支持自动加权求和。

    Parameters
    ----------
    model: 已训练好的模型
    X_train, y_train,  X_test, y_test: 训练集和测试集
    metrcis: dict, optional
        各项指标对应的权重（用于计算综合得分）

    Returns
    -------
    DataFrame
        包含各项指标得分和 weighted_score 的评估表
    """

    # 预测值
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    res_train = {}
    res_test = {}

    for func in metrics:
        name = func.__name__
        res_train[name] = func(y_train, y_train_pred)
        res_test[name] = func(y_test, y_test_pred)

    # 计算加权得分
    res_train['weighted_score'] = weighted_score_auto(res_train, weights)
    res_test['weighted_score'] = weighted_score_auto(res_test, weights)

    df = pd.DataFrame([res_train, res_test], index=['train_eval', 'test_eval'])
    return df