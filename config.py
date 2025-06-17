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
    """
    一、指标权重分配原则
    ​​召回率（Recall）权重最高​​
    ​​推荐权重：0.3-0.4​​
    ​​原因​​：漏检欺诈交易（FN）的损失远高于误报（FP）。银行场景中，1笔漏检欺诈可能损失数千美元，而误报仅增加人工审核成本。
    ​​调整依据​​：根据欺诈交易平均损失 vs. 人工审核单笔成本的比例动态调整。
    ​​精确率（Precision）与F1分数（F1-Score）中等权重​​
    ​​推荐权重：各0.2​​
    ​​原因​​：
    精确率控制误报量，避免正常交易被频繁拦截影响用户体验；
    F1平衡召回与精确率，综合反映少数类识别能力。
    ​​AUC（ROC-AUC）权重次之​​
    ​​推荐权重：0.1-0.15​​
    ​​原因​​：AUC衡量模型在不同阈值下的整体区分能力，但对高不平衡数据可能过于乐观。需结合PR-AUC（未列出）补充评估。
    ​​准确率（Accuracy）权重最低​​
    ​​推荐权重：≤0.05​​
    ​​原因​​：欺诈占比通常＜0.1%，即使模型全预测正常，准确率仍＞99.9%，毫无参考价值。
    """
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
        'cluster_clean': ClusterCentroids(
            estimator=KMeans(n_init='auto', n_clusters=20),
            random_state=RANDOM_STATE
        ),
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
        'smote_enn': SMOTEENN(
            sampling_strategy='auto',
            random_state=RANDOM_STATE,
            smote=SMOTE(k_neighbors=5),
            enn=EditedNearestNeighbours(n_neighbors=5)
        ),
        'adasyn': ADASYN(
            sampling_strategy='minority',
            n_neighbors=5,
            random_state=RANDOM_STATE
        )
    }


