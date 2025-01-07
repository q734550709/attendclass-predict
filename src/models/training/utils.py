import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
import yaml
from datetime import datetime

logger = logging.getLogger(__name__)

def setup_logging(config: Dict):
    """设置日志"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    # 配置根日志记录器
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 如果配置了文件日志
    if log_config.get('save_path'):
        log_path = Path(log_config['save_path'])
        log_path.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_path / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(file_handler)

def load_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """加载数据"""
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y

def save_model(model: Any, metrics: Dict, config: Dict, output_path: str):
    """保存模型和相关信息"""
    import joblib
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    model_path = output_dir / 'model.joblib'
    joblib.dump(model, model_path)
    
    # 保存评估指标
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 保存配置
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
        
    logger.info(f"Model and related files saved to {output_dir}")

def perform_cross_validation(model: Any, X: pd.DataFrame, y: pd.Series,
                           config: Dict) -> Dict[str, List[float]]:
    """执行交叉验证"""
    cv_config = config.get('training', {}).get('cross_validation', {})
    n_splits = cv_config.get('n_splits', 5)
    
    if cv_config.get('stratified', True):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                            random_state=config['model']['params']['random_state'])
    else:
        kf = KFold(n_splits=n_splits, shuffle=True,
                   random_state=config['model']['params']['random_state'])
    
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc_roc': [],
        'average_precision': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        # 计算指标
        metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['precision'].append(precision_score(y_val, y_pred))
        metrics['recall'].append(recall_score(y_val, y_pred))
        metrics['f1'].append(f1_score(y_val, y_pred))
        metrics['auc_roc'].append(roc_auc_score(y_val, y_prob))
        metrics['average_precision'].append(average_precision_score(y_val, y_prob))
        
        logger.info(f"Fold {fold} - Accuracy: {metrics['accuracy'][-1]:.4f}, "
                   f"F1: {metrics['f1'][-1]:.4f}")
    
    # 计算平均值和标准差
    cv_results = {}
    for metric, values in metrics.items():
        cv_results[f'{metric}_mean'] = np.mean(values)
        cv_results[f'{metric}_std'] = np.std(values)
    
    return cv_results

def plot_feature_importance(model: Any, feature_names: List[str],
                          output_path: Optional[str] = None):
    """绘制特征重要性"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning("Model doesn't support feature importance visualization")
        return
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # 绘制图形
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(20), x='importance', y='feature')
    plt.title('Top 20 Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    if output_path:
        output_file = Path(output_path) / 'feature_importance.png'
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        
        # 保存特征重要性数据
        importance_df.to_csv(
            Path(output_path) / 'feature_importance.csv',
            index=False
        )
    else:
        plt.show()

def plot_learning_curves(train_scores: List[float], val_scores: List[float],
                        metric_name: str, output_path: Optional[str] = None):
    """绘制学习曲线"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_scores) + 1)
    
    plt.plot(epochs, train_scores, 'b-', label='Training')
    plt.plot(epochs, val_scores, 'r-', label='Validation')
    
    plt.title(f'Learning Curves - {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    
    if output_path:
        output_file = Path(output_path) / f'learning_curve_{metric_name}.png'
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        
        # 保存学习曲线数据
        curve_df = pd.DataFrame({
            'epoch': epochs,
            'train_score': train_scores,
            'val_score': val_scores
        })
        curve_df.to_csv(
            Path(output_path) / f'learning_curve_{metric_name}.csv',
            index=False
        )
    else:
        plt.show()

def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                      metric: str = 'f1') -> Tuple[float, float]:
    """优化决策阈值"""
    thresholds = np.arange(0.1, 1.0, 0.01)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
            
        scores.append(score)
    
    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]

def save_experiment_info(exp_dir: str, config: Dict, metrics: Dict,
                        additional_info: Optional[Dict] = None):
    """保存实验信息"""
    exp_path = Path(exp_dir)
    exp_path.mkdir(parents=True, exist_ok=True)
    
    # 基本信息
    info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': config,
        'metrics': metrics
    }
    
    # 添加额外信息
    if additional_info:
        info.update(additional_info)
    
    # 保存为JSON文件
    with open(exp_path / 'experiment_info.json', 'w') as f:
        json.dump(info, f, indent=4)
    
    logger.info(f"Experiment information saved to {exp_path}")

def create_experiment_directory(base_dir: str, exp_name: Optional[str] = None) -> str:
    """创建实验目录"""
    if exp_name is None:
        exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    exp_dir = Path(base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (exp_dir / 'models').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'metrics').mkdir(exist_ok=True)
    (exp_dir / 'plots').mkdir(exist_ok=True)
    
    return str(exp_dir) 