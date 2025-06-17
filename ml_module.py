from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree  import DecisionTreeClassifier
from xgboost import XGBClassifier
from config import CONFIG

class LR_Threshold(BaseEstimator, TransformerMixin):
    def __init__(self,
                 penalty='l2',
                 C=1.0,
                 max_iter=100,
                 solver='lbfgs',
                 l1_ratio=None,
                 class_weight=None,
                 thr=0.5):
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.l1_ratio = l1_ratio
        self.class_weight = class_weight
        self.thr = thr

    def fit(self, X, y):
        self.clf = LogisticRegression(penalty=self.penalty,
                                 C=self.C,
                                 max_iter=self.max_iter,
                                 solver=self.solver,
                                 l1_ratio=self.l1_ratio,
                                 class_weight=self.class_weight,
                                 random_state=CONFIG.RANDOM_STATE)
        self.clf.fit(X, y)
        self.coef_ = self.clf.coef_

        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def predict(self, X):
        res = (self.predict_proba(X)[:, 1] >= self.thr) * 1
        return res.astype(int)

    @property
    def classes_(self):
        return self.clf.classes_


class DT_Threshold(BaseEstimator, TransformerMixin):
    def __init__(self,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=None,
                 max_leaf_nodes=None,
                 ccp_alpha=0.0,
                 thr=0.5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.ccp_alpha = ccp_alpha
        self.thr = thr

    def fit(self, X, y):
        self.clf = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            ccp_alpha=self.ccp_alpha,
            random_state=CONFIG.RANDOM_STATE
        )
        self.clf.fit(X, y)

        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def predict(self, X):
        res = (self.predict_proba(X)[:, 1] >= self.thr) * 1
        return res.astype(int)
    
    @property
    def classes_(self):
        return self.clf.classes_
    
class XGB_Threshold(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 max_depth=3,
                 n_estimators=100,
                 learning_rate=0.1,
                 subsample=1.0,
                 colsample_bytree=1.0,
                 reg_lambda=1,
                 reg_alpha=0,
                 thr=0.5):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.thr = thr

    def fit(self, X, y):
        self.clf = XGBClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            random_state=CONFIG.RANDOM_STATE,
            use_lable_encoder=False,
            eval_metric='logloss'
        )
        self.clf.fit(X, y)
        return self
    
    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def predict(self, X):
        res = (self.predict_proba(X)[:, 1] >= self.thr) * 1
        return res.astype(int)

    @property
    def classes_(self):
        return self.clf.classes_
    




