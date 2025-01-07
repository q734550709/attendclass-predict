import xgboost as xgb
from .base import BaseModel
import numpy as np

class XGBoostModel(BaseModel):
    def __init__(self, config):
        self.config = config
        self.model = xgb.XGBClassifier(
            n_estimators=config.get('n_estimators', 100),
            learning_rate=config.get('learning_rate', 0.1),
            max_depth=config.get('max_depth', 6),
            random_state=config.get('random_state', 42),
            use_label_encoder=False,
            eval_metric='logloss'
        )
    
    def train(self, X, y):
        """训练XGBoost模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
        """
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """预测类别
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的类别标签
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """预测概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的概率值
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y):
        """评估模型性能
        
        Args:
            X: 特征矩阵
            y: 真实标签
            
        Returns:
            dict: 包含准确率等评估指标的字典
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return {
            'accuracy': accuracy
        } 