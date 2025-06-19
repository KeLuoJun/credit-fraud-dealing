from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from imblearn.under_sampling import (
    RandomUnderSampler, NearMiss,
    ClusterCentroids,
    EditedNearestNeighbours)
from imblearn.over_sampling import (
    KMeansSMOTE, ADASYN,
    SMOTE, BorderlineSMOTE)

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTEENN

class CONFIG:
    DATA_PATH = './data/creditcard.csv'

    NUMERIC_COLS = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

    TARGET = 'Class'

    RANDOM_STATE = 2025

    # 任务是信用卡欺诈检测
    # 目标是确保标记的欺诈行为大概率真实
    # 因此为了减少误伤正常交易，同时保持一定检出率，设置以下权重

    METRIC_WEIGHTS = {
        'precision_score': .2,
        'recall_score': .4,
        'f1_score': .2,
        'roc_auc_score': .15,
        'accuracy_score': .05
    }

    METRICS_FUNC = [
        accuracy_score,
        recall_score,
        precision_score,
        f1_score,
        roc_auc_score
    ]


    SAMPLING = {
        'random_usample': RandomUnderSampler(random_state=RANDOM_STATE),
        'kmeans_smote': KMeansSMOTE(
            kmeans_estimator=KMeans(n_init=10, n_clusters=10),
            cluster_balance_threshold=0.1,
            random_state=RANDOM_STATE
        ),
        'borderline_smote': BorderlineSMOTE(
            kind='borderline-1', 
            sampling_strategy=0.05,  # 保守过采样
            k_neighbors=10,          # 增加邻居数
            random_state=RANDOM_STATE
        ),
        'smote': SMOTE(
            sampling_strategy='auto',
            random_state=RANDOM_STATE,  # 注意：保持与之前相同的随机种子
            k_neighbors=5
        ),
        'adasyn': ADASYN(
            sampling_strategy='minority',
            n_neighbors=5,
            random_state=RANDOM_STATE
        )
    }


    SAMPLING_NAME = "kmeans_smote"
