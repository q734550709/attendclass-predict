import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
import logging
from typing import Dict, List, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config: Dict):
        """初始化数据预处理器"""
        self.config = config
        self.scalers = {}
        self.encoders = {}
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合并转换数据"""
        df = self._handle_missing_values(df)
        df = self._scale_numerical_features(df)
        df = self._encode_categorical_features(df)
        df = self._encode_temporal_features(df)
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换数据"""
        df = self._handle_missing_values(df)
        df = self._apply_scaling(df)
        df = self._apply_encoding(df)
        df = self._encode_temporal_features(df)
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        strategy = self.config['preprocessing'].get('missing_values', {})
        
        # 数值型特征的缺失值处理
        numerical_strategy = strategy.get('numerical', 'mean')
        if numerical_strategy == 'mean':
            df = df.fillna(df.mean())
        elif numerical_strategy == 'median':
            df = df.fillna(df.median())
        elif numerical_strategy == 'zero':
            df = df.fillna(0)
            
        # 分类特征的缺失值处理
        categorical_strategy = strategy.get('categorical', 'mode')
        if categorical_strategy == 'mode':
            df = df.fillna(df.mode().iloc[0])
        elif categorical_strategy == 'unknown':
            df = df.fillna('unknown')
            
        return df
    
    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """缩放数值特征"""
        numerical_features = self._get_numerical_features(df)
        scaling_method = self.config['preprocessing']['scaling']['method']
        
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {scaling_method}")
            
        if numerical_features:
            self.scalers['numerical'] = scaler
            scaled_values = scaler.fit_transform(df[numerical_features])
            df[numerical_features] = scaled_values
            
        return df
    
    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用已拟合的缩放器"""
        numerical_features = self._get_numerical_features(df)
        if numerical_features and 'numerical' in self.scalers:
            scaled_values = self.scalers['numerical'].transform(df[numerical_features])
            df[numerical_features] = scaled_values
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码分类特征"""
        categorical_features = self._get_categorical_features(df)
        encoding_method = self.config['preprocessing']['encoding']['categorical']
        
        if encoding_method == 'label':
            for feature in categorical_features:
                encoder = LabelEncoder()
                df[feature] = encoder.fit_transform(df[feature].astype(str))
                self.encoders[feature] = encoder
        elif encoding_method == 'one_hot':
            df = pd.get_dummies(df, columns=categorical_features)
            
        return df
    
    def _apply_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用已拟合的编码器"""
        categorical_features = self._get_categorical_features(df)
        encoding_method = self.config['preprocessing']['encoding']['categorical']
        
        if encoding_method == 'label':
            for feature in categorical_features:
                if feature in self.encoders:
                    df[feature] = self.encoders[feature].transform(df[feature].astype(str))
        elif encoding_method == 'one_hot':
            df = pd.get_dummies(df, columns=categorical_features)
            
        return df
    
    def _encode_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码时间特征"""
        temporal_features = self._get_temporal_features(df)
        encoding_method = self.config['preprocessing']['encoding']['temporal']
        
        if encoding_method == 'cyclical':
            for feature in temporal_features:
                if df[feature].dtype == 'datetime64[ns]':
                    df = self._add_cyclical_features(df, feature)
                    
        return df
    
    def _add_cyclical_features(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """添加周期性特征"""
        # 时间特征转换为周期性特征
        df[f'{feature}_hour_sin'] = np.sin(2 * np.pi * df[feature].dt.hour / 24)
        df[f'{feature}_hour_cos'] = np.cos(2 * np.pi * df[feature].dt.hour / 24)
        df[f'{feature}_day_sin'] = np.sin(2 * np.pi * df[feature].dt.day / 31)
        df[f'{feature}_day_cos'] = np.cos(2 * np.pi * df[feature].dt.day / 31)
        df[f'{feature}_month_sin'] = np.sin(2 * np.pi * df[feature].dt.month / 12)
        df[f'{feature}_month_cos'] = np.cos(2 * np.pi * df[feature].dt.month / 12)
        df[f'{feature}_dayofweek_sin'] = np.sin(2 * np.pi * df[feature].dt.dayofweek / 7)
        df[f'{feature}_dayofweek_cos'] = np.cos(2 * np.pi * df[feature].dt.dayofweek / 7)
        
        return df
    
    def _get_numerical_features(self, df: pd.DataFrame) -> List[str]:
        """获取数值特征"""
        return df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    def _get_categorical_features(self, df: pd.DataFrame) -> List[str]:
        """获取分类特征"""
        return df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def _get_temporal_features(self, df: pd.DataFrame) -> List[str]:
        """获取时间特征"""
        return df.select_dtypes(include=['datetime64']).columns.tolist()
    
    def save(self, path: str):
        """保存预处理器"""
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        
    @classmethod
    def load(cls, path: str) -> 'DataPreprocessor':
        """加载预处理器"""
        import joblib
        return joblib.load(path) 