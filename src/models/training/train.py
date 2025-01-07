import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import yaml
import logging
import joblib
from pathlib import Path
from typing import Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path: str):
        """初始化模型训练器"""
        self.config = self._load_config(config_path)
        self.model = self._initialize_model()
        self.metrics = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_model(self) -> Any:
        """初始化模型"""
        model_config = self.config['model']
        if model_config['type'] == 'RandomForest':
            return RandomForestClassifier(**model_config['params'])
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")
    
    def load_data(self) -> tuple:
        """加载训练数据"""
        data_path = self.config['data']['paths']['processed']
        
        train_df = pd.read_csv(f"{data_path}/train.csv")
        val_df = pd.read_csv(f"{data_path}/val.csv")
        test_df = pd.read_csv(f"{data_path}/test.csv")
        
        # 分离特征和标签
        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_val = val_df.drop('target', axis=1)
        y_val = val_df['target']
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """训练模型"""
        logger.info("Starting model training")
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, dataset_type: str = 'val') -> Dict:
        """评估模型"""
        logger.info(f"Evaluating model on {dataset_type} set")
        
        # 获取预测结果
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        
        # 计算评估指标
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc_roc': roc_auc_score(y, y_prob)
        }
        
        # 记录评估结果
        self.metrics[dataset_type] = metrics
        
        # 输出评估结果
        for metric_name, value in metrics.items():
            logger.info(f"{dataset_type} {metric_name}: {value:.4f}")
        
        return metrics
    
    def save_model(self, exp_dir: str):
        """保存模型和评估结果"""
        # 创建实验目录
        exp_path = Path(exp_dir)
        exp_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        model_path = exp_path / 'model.joblib'
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # 保存评估指标
        metrics_path = exp_path / 'metrics.yaml'
        with open(metrics_path, 'w') as f:
            yaml.dump(self.metrics, f)
        logger.info(f"Metrics saved to {metrics_path}")

def main():
    """主函数"""
    # 初始化训练器
    trainer = ModelTrainer("configs/base/model.yaml")
    
    # 加载数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer.load_data()
    
    # 训练模型
    trainer.train(X_train, y_train)
    
    # 评估模型
    trainer.evaluate(X_val, y_val, 'validation')
    trainer.evaluate(X_test, y_test, 'test')
    
    # 保存模型和评估结果
    trainer.save_model("experiments/exp001")

if __name__ == "__main__":
    main() 