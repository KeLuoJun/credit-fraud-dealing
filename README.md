# 信用卡欺诈检测系统

该项目实现了一个完整的信用卡欺诈检测系统，使用机器学习方法来识别可疑的欺诈交易。
![数据集下载地址](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## 项目结构

```
credit-fraud-dealing/
  ├── config.py           # 配置文件，包含数据路径、模型参数等配置
  ├── data/               # 数据目录
  │   ├── creditcard.csv  # 原始信用卡交易数据
  │   └── new_df.csv      # 处理后的数据
  ├── main.ipynb          # 主要的分析和模型训练notebook
  ├── metric.py           # 模型评估指标实现
  ├── ml_module.py        # 机器学习模型实现
  ├── tuning.py           # 模型调优相关代码
  └── utils.py            # 工具函数
```

## 主要功能

1. **数据处理**：处理信用卡交易数据，包括特征工程和数据清洗
2. **不平衡处理**：实现了多种采样方法来处理类别不平衡问题：
   - 随机欠采样 (RandomUnderSampler)
   - SMOTE过采样
   - Borderline-SMOTE
   - ADASYN
   - KMeans-SMOTE

3. **模型实现**：提供了多个分类器实现：
   - 逻辑回归 (LR_Threshold)
   - 决策树 (DT_Threshold)
   - XGBoost (XGB_Threshold)
   
   所有模型都支持自定义决策阈值，以适应不同的业务场景。

4. **评估系统**：
   实现了完整的模型评估体系，包括：
   - 准确率 (Accuracy)
   - 召回率 (Recall)
   - 精确率 (Precision)
   - F1分数
   - ROC-AUC

   评估指标采用加权方式计算最终得分，权重分配基于业务重要性：
   - 召回率：40% (最重要，避免漏检)
   - 精确率：20%
   - F1分数：20%
   - ROC-AUC：15%
   - 准确率：5%

## 使用方法

1. **环境配置**
```bash
pip install -r requirements.txt
```

2. **数据准备**
- 将信用卡交易数据放置在 `data/creditcard.csv`

3. **模型训练**
- 打开 `main.ipynb` 运行相关代码
- 可以在 `config.py` 中调整模型参数和采样方法

## 注意事项

1. 数据不平衡处理
   - 项目默认使用kmeans_smote方法处理类别不平衡
   - 可以在 `config.py` 中修改 `SAMPLING_NAME` 来切换不同的采样方法

2. 模型阈值调整
   - 所有模型都支持自定义决策阈值
   - 默认阈值为0.5，可以根据业务需求调整

3. 评估指标
   - 评估指标的权重设置基于欺诈检测的业务特点
   - 可以在 `config.py` 中的 `METRIC_WEIGHTS` 调整权重

