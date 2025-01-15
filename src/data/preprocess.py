import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging
from typing import Dict, Tuple, List, Any
from pathlib import Path
import yaml # 用于加载配置文件
logger = logging.getLogger(__name__)

# 数据预处理器
class DataPreprocessor:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """缩放数值特征"""
        normalization_map = self.config['data_processing_config']['scaling']['normalization_map']  # 获取归一化方法映射

        df_normalized = df.copy()

        for column, method in normalization_map.items():
            if method == 'min-max变换':
                scaler = MinMaxScaler()
                df_normalized[column] = scaler.fit_transform(df_normalized[[column]])
            
            elif method == 'log 变换+标准化':
                df_normalized[column] = np.log1p(df_normalized[column])
                scaler = StandardScaler()
                df_normalized[column] = scaler.fit_transform(df_normalized[[column]])
            
            elif method == '标准化':
                scaler = StandardScaler()
                df_normalized[column] = scaler.fit_transform(df_normalized[[column]])
            
            elif method == 'robustscaler变换':
                scaler = RobustScaler()
                df_normalized[column] = scaler.fit_transform(df_normalized[[column]])
            
            elif method == '数据平移+log 变换+标准化':
                df_normalized[column] = np.log1p(df_normalized[column] + 1)
                scaler = StandardScaler()
                df_normalized[column] = scaler.fit_transform(df_normalized[[column]])
            
            else:
                raise ValueError(f"Unsupported normalization method: {method}")  # 不支持的归一化方法

        return df_normalized
    
    def save(self, path: str):
        """保存预处理器"""
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)  # 创建保存路径
        joblib.dump(self, path)  # 保存预处理器
        
    @classmethod
    def load(cls, path: str) -> 'DataPreprocessor':
        """加载预处理器"""
        import joblib
        return joblib.load(path)  # 加载预处理器