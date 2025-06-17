import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch
import random
from config import CONFIG


def seed_everything(seed=CONFIG.RANDOM_STATE):
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    # 设置 CUDNN 为确定性操作，确保每次运行结果相同（有助于复现）
    torch.backends.cudnn.deterministic = True
    
    # 禁用 CUDNN 的自动优化功能，以避免影响确定性行为
    torch.backends.cudnn.benchmark = False
    
    # 设置 numpy 的随机种子，确保 numpy 的随机数生成器是可复现的
    np.random.seed(seed)
    
    # 设置 Pytorch 的 CPU 随机种子，确保 CPU 上的操作是可复现的
    torch.manual_seed(seed)
    
    # 如果有 GPU 可用，设置所有 GPU 设备的随机数种子，确保 GPU 上的操作也是可复现的
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.empty_cache()
    pl.set_random_seed(seed)
    print(f"Set Seed = {seed}")


def kde_plot(
        df: pl.DataFrame,
        feature_name: str,
        target_column: str,
        target_classes: tuple = (0, 1)):
    sns.set(style='whitegrid')
    plt.rcParams['axes.unicode_minus'] = False

    # 提取对应类别的数据
    def get_values(label):
        return df.filter(pl.col(target_column) == label).select(pl.col(feature_name)).to_series().to_numpy()

    data_0 = get_values(target_classes[0])
    data_1 = get_values(target_classes[1])

    plt.figure(figsize=(10, 6), dpi=200)
    sns.kdeplot(data_0, 
                label=f"{target_classes[0]} (mean={data_0.mean():.2f})",
                linewidth=2)
    sns.kdeplot(data_1,
                label=f"{target_classes[1]} (mean={data_1.mean():.2f})",
                linewidth=2, linestyle='--')

    plt.axvline(data_0.mean(), color='blue', linestyle=':', alpha=.7)
    plt.axvline(data_1.mean(), color='orange', linestyle=':', alpha=.7)

    plt.title(f"{feature_name} Distribution by {target_classes}", fontsize=14)
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=.5)
    plt.show()


def joint_kde_2d_plot(
        df: pl.DataFrame,
        x: str,
        y: str,
        target_column: str = 'Class',
        class_lables: tuple = (0, 1),
        cmap_colors: tuple = ('Blues', "Oranges"),
        figsize: tuple = (12, 5),
        dpi: int = 300
):
    """
    绘制二维联合密度热力图，展示x与y的联合分布，并按target_column分类
    """
    df_pd = df.select([x, y, target_column]).to_pandas()
    plt.figure(figsize=figsize, dpi=dpi)
    for i, label in enumerate(class_lables):
        plt.subplot(1, 2, i + 1)
        subset = df_pd[df_pd[target_column] == label]
        sns.kdeplot(
            data=subset,
            x=x,
            y=y,
            fill=True,
            cmap=cmap_colors[i],
            thresh=0.05,
            levels=100,
            alpha=.9)
        plt.title(f"{label} - {x} vs {y}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
