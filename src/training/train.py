import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")

import sys
import os
# 添加src目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import pandas as pd
import numpy as np
from datetime import datetime
from models.factory import ModelFactory
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
import yaml
import logging
import joblib
from typing import Dict, List, Tuple, Any


# 设置日志
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path: str):
        """初始化模型训练器"""
        # 获取当前脚本文件所在的目录
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        # 加载配置文件
        self.config = self._load_config(config_path)
        self.model = self._initialize_model()
        self.metrics = {}
        # 从配置文件中获取实验目录和实验名称
        self.exp_name = self.config['experiment']['name']
        self.exp_dir = self.config['output']['exp_dir']
        self.exp_model_dir = self.config['output']['exp_model_dir']

        # 创建实验目录
        self.exp_path = os.path.join(self.current_dir, self.exp_dir)
        os.makedirs(self.exp_path, exist_ok=True)

        # 创建模型目录
        self.exp_model_path = os.path.join(self.current_dir, self.exp_model_dir)
        os.makedirs(self.exp_model_path, exist_ok=True)
        
        # 设置日志
        self.setup_logging()

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self) -> None:
        """设置日志"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        
        # 配置根日志记录器
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 如果配置了文件日志
        if log_config.get('save_path'):

            relative_log_path = log_config['save_path']
            log_path = os.path.normpath(os.path.join(self.current_dir, relative_log_path))
            os.makedirs(log_path, exist_ok=True)
            
            file_handler = logging.FileHandler(
                os.path.join(log_path, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            )
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logging.getLogger().addHandler(file_handler)

    def _initialize_model(self) -> Any:
        """初始化模型"""
        model_config = self.config['model']
        return ModelFactory.create_model(model_config['type'], model_config['params'])
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """加载训练数据"""
        relative_data_path = self.config['data']['paths']
        
        # 读取数据
        X_train = pd.read_csv(os.path.normpath(os.path.join(self.current_dir, relative_data_path['X_train'])))
        # 保存用户id
        user_ids = X_train['用户id']
        # 删除用户id后的数据作为特征
        X_train = X_train.drop(columns=['用户id'])

        # 读取标签
        y_train = pd.read_csv(os.path.normpath(os.path.join(self.current_dir, relative_data_path['y_train']))).squeeze()
        
        return X_train, y_train, user_ids
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """训练模型"""
        logger.info("Starting model training")
        self.model.fit(X_train, y_train)
    
    def perform_cross_validation(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """执行交叉验证"""
        cv_config = self.config['cross_validation']
        n_splits = cv_config.get('n_splits', 5)
        random_state = self.config['model']['params'].get('random_state', 42)
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        metrics = {
            'metrics_type': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc_roc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 训练模型
            self.model.fit(X_train, y_train)
            
            # 预测
            y_pred = self.model.predict(X_val)
            y_prob = self.model.predict_proba(X_val)[:, 1]
            
            # 计算指标
            metrics['metrics_type'].append(f'cross_validation_fold_{fold}')
            metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics['precision'].append(precision_score(y_val, y_pred))
            metrics['recall'].append(recall_score(y_val, y_pred))
            metrics['f1'].append(f1_score(y_val, y_pred))
            metrics['auc_roc'].append(roc_auc_score(y_val, y_prob))
        
        # 计算平均值和标准差
        metrics['metrics_type'].append('cross_validation_mean')
        metrics['metrics_type'].append('cross_validation_std')
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
            mean_value = np.mean(metrics[metric])
            std_value = np.std(metrics[metric])
            metrics[metric].append(mean_value)
            metrics[metric].append(std_value)
        
        cv_results = pd.DataFrame(metrics)
        
        logger.info(f"Cross-validation results: {cv_results}")

        return cv_results

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """评估模型"""
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        
        metrics = {
            'metrics_type': ['train'],
            'accuracy': [accuracy_score(y, y_pred)],
            'precision': [precision_score(y, y_pred)],
            'recall': [recall_score(y, y_pred)],
            'f1': [f1_score(y, y_pred)],
            'auc_roc': [roc_auc_score(y, y_prob)]
        }
        
        metrics_result = pd.DataFrame(metrics)
        
        logger.info(f"Train evaluation results: {metrics_result}")

        return metrics_result
    
    def save_model(self) -> None:
        """保存模型结果"""
        relative_model_path = os.path.join(self.exp_model_path, f'{self.exp_name}_train_model.joblib')
        model_path = os.path.normpath(os.path.join(self.current_dir, relative_model_path))

        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

    def save_cvresults_to_csv(self, cv_results: pd.DataFrame) -> None:
        """保存交叉验证结果"""
        relative_cv_results_path = os.path.join(self.exp_path, f'{self.exp_name}_cv_results.csv')
        cv_results_path = os.path.normpath(os.path.join(self.current_dir, relative_cv_results_path))

        cv_results.to_csv(cv_results_path, index=False)
        logger.info(f"Cross-validation results saved to {cv_results_path}")
    
    def save_evaluation_to_csv(self, evaluation: pd.DataFrame) -> None:
        """保存评估结果"""
        relative_evaluation_path = os.path.join(self.exp_path, f'{self.exp_name}_train_evaluation.csv')
        evaluation_path = os.path.normpath(os.path.join(self.current_dir, relative_evaluation_path))

        evaluation.to_csv(evaluation_path, index=False)
        logger.info(f"Evaluation results saved to {evaluation_path}")

    def save_results_to_csv(self, cv_results: pd.DataFrame, evaluation: pd.DataFrame) -> None:
        """保存交叉验证和评估结果"""
        relative_results_path = os.path.join(self.exp_path, f'{self.exp_name}_train_evaluation_results.csv')
        results_path = os.path.normpath(os.path.join(self.current_dir, relative_results_path))

        # 合并结果
        combined_results = pd.concat([cv_results, evaluation], ignore_index=True)
        combined_results.to_csv(results_path, index=False)
        
        logger.info(f"Cross-validation and evaluation results saved to {results_path}")

def main(config_path: str) -> None:
    """主函数"""
    # 初始化训练器
    trainer = ModelTrainer(config_path)
    
    # 加载数据
    X_train, y_train, user_ids = trainer.load_data()
    # 训练模型
    trainer.train(X_train, y_train)
    # 交叉验证
    cv_results = trainer.perform_cross_validation(X_train, y_train)

    # 评估模型
    metrics = trainer.evaluate(X_train, y_train)
    
    # 保存模型
    trainer.save_model()
    # 保存交叉验证结果
    # trainer.save_cvresults_to_csv(cv_results)
    # 保存评估结果
    # trainer.save_evaluation_to_csv(metrics)

    # 保存交叉验证和评估结果
    trainer.save_results_to_csv(cv_results, metrics)

    return cv_results, metrics

if __name__ == "__main__":
    # 加载数据配置
    config_path = "../../configs/experiments/exp_preview.yaml"
    main(config_path)