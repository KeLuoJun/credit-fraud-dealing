import optuna
import numpy as np
import pandas as pd
import polars as pl
from typing import Literal, Optional, Union

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.pipeline import Pipeline

# from sklearn.pipeline import Pipeline
import sklearn
from sklearn.base import clone
import sklearn.pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import CONFIG
from metric import weighted_score_auto
from ml_module import LR_Threshold, DT_Threshold, XGB_Threshold

import torch
import torch.nn as nn
from skorch import NeuralNetClassifier

import gc


def cross_val_eval(pipelinre, X, y,
                   metric_funcs=CONFIG.METRICS_FUNC,
                   n_splits=5,
                   random_state=CONFIG.RANDOM_STATE):
    """
    对任意 pipeline 执行 StratifiedKFold 交叉验证，返回平均指标得分字典

    Parameters
    ----------
    pipeline: sklearn.pipeline.Pipeline
        已构建好的模型管道（包含预处理和模型）
    X: pd.DataFrame
        特征矩阵
    y: pd.DataFrame
        目标变量
    n_splits: int, optional
        K 折交叉验证的 K 值，默认为 5
    metric_funcs: list of callable
        指标函数列表
    random_state: int
        随机种子

    Returns
    -------
    dict
        每个指标的得分，如 {'accuarcy_score': 0.8, 'f1_score': 0.9}
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metric_name = [f.__name__ for f in metric_funcs]
    score_map = {name: [] for name in metric_name}

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        new_pipeline = clone(pipelinre)  # 每次都克隆一个干净的 pipeline
        # 否则：
        # 1、原始的 pipeline 会被最后一批训练数据“污染”。
        # 2、如果你在交叉验证之后还想用这个 pipeline 做其他事情（比如重新训练或预测），它的状态已经变了。
        new_pipeline.fit(X_tr, y_tr.to_numpy().astype(np.float32).reshape(-1, 1))
        y_pred = new_pipeline.predict(X_val)

        for func in metric_funcs:
            score_map[func.__name__].append(func(y_val, y_pred))

        del new_pipeline, X_tr, y_tr, X_val, y_val, y_pred
        gc.collect()

    res = {k: np.mean(v) for k, v in score_map.items()}
    return res


def lr_model_objective(trial: optuna.Trial, 
                       X_train: pd.DataFrame, 
                       y_train: pd.DataFrame, 
                       preprocessor: sklearn.pipeline.Pipeline, 
                       sampling: Optional[Literal['random_usample', 'nearmiss', 'smote', 'adasyn']] = None):
    """
    逻辑回归模型超参数目标函数
    """
    penalty_solver_combo = trial.suggest_categorical('penalty_solver', [
        ('l1', 'liblinear'),
        ('l1', 'saga'),

        ('l2', 'lbfgs'),
        ('l2', 'liblinear'),
        ('l2', 'newton-cg'),
        ('l2', 'sag'),
        ('l2', 'saga'),

        ('elasticnet', 'saga')
    ])
    penalty, solver = penalty_solver_combo
    C = trial.suggest_float('C', 1e-5, 1e5, log=True)
    max_iter = trial.suggest_int('max_iter', 1, 1000, step=10)
    l1_ratio = trial.suggest_float('l1_ratio', 0.2, 0.8) if penalty == 'elasticnet' else None

    model = LogisticRegression(
        penalty=penalty,
        solver=solver,
        C=C,
        max_iter=max_iter,
        l1_ratio=l1_ratio,
        random_state=CONFIG.RANDOM_STATE
    )

    if sampling:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            (sampling, CONFIG.SAMPLING[sampling]),
            ('classifier', model)
        ])
    else:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

    # 交叉验证
    score_map = cross_val_eval(pipeline, X_train, y_train)
    return weighted_score_auto(score_map, weights=CONFIG.METRIC_WEIGHTS)

def lr_thr_model_objective(trial: optuna.Trial,
                           X_train: pd.DataFrame, 
                           y_train: pd.DataFrame,  
                           preprocessor: sklearn.pipeline.Pipeline,
                           sampling: Optional[Literal['random_usample', 'nearmiss', 'smote', 'adasyn']] = None):
    penalty_solver_combo = trial.suggest_categorical('penalty_solver', [
        ('l1', 'liblinear'),
        ('l1', 'saga'),

        ('l2', 'lbfgs'),
        ('l2', 'liblinear'),
        ('l2', 'newton-cg'),
        ('l2', 'sag'),
        ('l2', 'saga'),

        ('elasticnet', 'saga')
    ])
    penalty, solver = penalty_solver_combo
    C = trial.suggest_float('C', 1e-5, 1e5, log=True)
    max_iter = trial.suggest_int('max_iter', 1, 1000, step=10)
    l1_ratio = trial.suggest_float('l1_ratio', 0.2, 0.8) if penalty == 'elasticnet' else None
    thr = trial.suggest_float('threshold', 0, 1)

    model = LR_Threshold(penalty=penalty,
                         C=C,
                         max_iter=max_iter,
                         solver=solver,
                         l1_ratio=l1_ratio,
                         thr=thr)

    if sampling:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            (sampling, CONFIG.SAMPLING[sampling]),
            ('classifier', model)
        ])
    else:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
    
    # 交叉验证
    score_map = cross_val_eval(pipeline, X_train, y_train)
    return weighted_score_auto(score_map, weights=CONFIG.METRIC_WEIGHTS)

def dt_model_objective(trial: optuna.Trial,
                        X_train: pd.DataFrame, 
                        y_train: pd.DataFrame, 
                        preprocessor: sklearn.pipeline.Pipeline,
                        sampling: Optional[Literal['random_usample', 'nearmiss', 'smote', 'adasyn']] = None):
    # 决策树参数
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
    max_depth = trial.suggest_int('max_depth', 2, 10)
    min_samples_split = trial.suggest_int('min_smaples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 10, 100)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    cpp_alpha = trial.suggest_float('cpp_alpha', 0.0, 0.5)

    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
        max_features=max_features,
        ccp_alpha=cpp_alpha,
        random_state=CONFIG.RANDOM_STATE
    )

    if sampling:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            (sampling, CONFIG.SAMPLING[sampling]),
            ('classifier', model)
        ])
    else:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
    
    score_map = cross_val_eval(pipeline, X_train, y_train)
    return weighted_score_auto(score_map, weights=CONFIG.METRIC_WEIGHTS)

def dt_thr_model_objective(trial: optuna.Trial,
                        X_train: pd.DataFrame, 
                        y_train: pd.DataFrame, 
                        preprocessor: sklearn.pipeline.Pipeline,
                        sampling: Optional[Literal['random_usample', 'nearmiss', 'smote', 'adasyn']] = None):
    """ 决策树 + 阈值调优目标函数 """
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
    max_depth = trial.suggest_int('max_depth', 2, 10)
    min_samples_split = trial.suggest_int('min_smaples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 10, 100)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    cpp_alpha = trial.suggest_float('cpp_alpha', 0.0, 0.5)
    thr = trial.suggest_float('threshold', 0.1, 0.9)

    model = DT_Threshold(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        ccp_alpha=cpp_alpha,
        thr=thr
    )

    if sampling:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            (sampling, CONFIG.SAMPLING[sampling]),
            ('classifier', model)
        ])
    else:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
    
    score_map = cross_val_eval(pipeline, X_train, y_train)
    return weighted_score_auto(score_map, weights=CONFIG.METRIC_WEIGHTS)


def xgb_model_objective(trial: optuna.Trial,
                        X_train: pd.DataFrame, 
                        y_train: pd.DataFrame, 
                        preprocessor: sklearn.pipeline.Pipeline,
                        sampling: Optional[Literal['random_usample', 'nearmiss', 'smote', 'adasyn']] = None):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 20, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_aplha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': CONFIG.RANDOM_STATE
    }

    model = XGBClassifier(**params)

    if sampling:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            (sampling, CONFIG.SAMPLING[sampling]),
            ('classifier', model)
        ])
    else:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
    
    score_map = cross_val_eval(pipeline, X_train, y_train)
    return weighted_score_auto(score_map, weights=CONFIG.METRIC_WEIGHTS)

def xgb_thr_model_objective(trial: optuna.Trial,
                        X_train: pd.DataFrame, 
                        y_train: pd.DataFrame, 
                        preprocessor: sklearn.pipeline.Pipeline,
                        sampling: Optional[Literal['random_usample', 'nearmiss', 'smote', 'adasyn']] = None):
    """ XGBoost + 阈值调优目标函数 """
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
        "thr": trial.suggest_float("thr", 0.1, 0.9),
    }

    model = XGB_Threshold(**params)

    if sampling:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            (sampling, CONFIG.SAMPLING[sampling]),
            ('classifier', model)
        ])
    else:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
    
    score_map = cross_val_eval(pipeline, X_train, y_train)
    return weighted_score_auto(score_map, weights=CONFIG.METRIC_WEIGHTS)



