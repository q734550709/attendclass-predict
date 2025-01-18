import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import yaml
import logging
import joblib
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
# 查找并设置中文字体
font_path = fm.findfont("SimHei")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


# 设置日志
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, config_path: str):
        """初始化评估器"""
        self.config = self._load_config(config_path)
        self.metrics = {}
                
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
            log_path = Path(log_config['save_path'])
            log_path.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_path / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logging.getLogger().addHandler(file_handler)

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """加载测试数据"""
        data_paths = self.config['data']['paths']
        
        # 读取测试数据
        X_test = pd.read_csv(data_paths['X_test'])
        # 保存用户id
        user_ids = X_test['用户id']
        # 删除用户id后的数据作为特征
        X_test = X_test.drop(columns=['用户id'])

        # 读取标签
        y_test = pd.read_csv(data_paths['y_test']).squeeze()
        
        return X_test, y_test, user_ids

    def evaluate(self, y_test: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
        """评估模型"""
        self.metrics = {
            'metrics_type': ['test'],
            'accuracy': [accuracy_score(y_test, y_pred)],
            'precision': [precision_score(y_test, y_pred)],
            'recall': [recall_score(y_test, y_pred)],
            'f1': [f1_score(y_test, y_pred)],
            'auc_roc': [roc_auc_score(y_test, y_prob)]
        }
        
        metrics_result = pd.DataFrame(self.metrics)
        
        logger.info(f"Test evaluation results: {metrics_result}")

        return metrics_result

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            output_path: str, exp_name: str) -> None:
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 保存图片
        output_file = Path(output_path) / f'{exp_name}_confusion_matrix.png'
        plt.savefig(output_file)
        plt.close()
        
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                      output_path: str, exp_name: str) -> None:
        """绘制ROC曲线"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {self.metrics["auc_roc"][0]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # 保存图片
        output_file = Path(output_path) / f'{exp_name}_roc_curve.png'
        plt.savefig(output_file)
        plt.close()
    
    def plot_feature_importance(self, model: Any, feature_names: List[str], 
                                output_path: str, exp_name: str) -> None:
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
        
        # 保存图片
        output_file = Path(output_path) / f'{exp_name}_feature_importance.png'
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        
        # 保存特征重要性数据
        importance_df.to_csv(
            Path(output_path) / f'{exp_name}_feature_importance.csv',
            index=False
        )

def main(config_path: str, model_type: str) -> None:
    """主函数"""
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 获取实验名称和模型目录
    exp_name = config['experiment']['name']
    exp_dir = config['output']['exp_dir']
    exp_model_dir = config['output']['exp_model_dir']
    
    # 创建实验目录
    exp_path = Path(exp_dir)
    exp_path.mkdir(parents=True, exist_ok=True)
    
    # 创建模型目录
    exp_model_path = Path(exp_model_dir)
    exp_model_path.mkdir(parents=True, exist_ok=True)

    # 根据模型类型加载模型
    model_filename = f'{exp_name}_{model_type}_model.joblib'
    model_path = exp_model_path / model_filename
    model = joblib.load(model_path)

    # 初始化评估器
    evaluator = ModelEvaluator(config_path)
    
    # 加载测试数据
    X_test, y_test, user_ids = evaluator.load_data()

    # 获取预测结果
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 评估模型
    metrics_result = evaluator.evaluate(y_test, y_pred, y_prob)
    
    # 保存评估结果
    evaluation_path = exp_path / f'{exp_name}_test_evaluation.csv'
    metrics_result.to_csv(evaluation_path, index=False)
    logger.info(f"Evaluation results saved to {evaluation_path}")

    # 生成可视化
    evaluator.plot_confusion_matrix(y_test, y_pred, exp_path, exp_name)
    evaluator.plot_roc_curve(y_test, y_prob, exp_path, exp_name)
    evaluator.plot_feature_importance(model, X_test.columns, exp_path, exp_name)    
    logger.info("Evaluation visualizations saved")

if __name__ == "__main__":
    # 加载数据配置
    config_path = "../../configs/experiments/exp_preview.yaml"
    main(config_path, 'train')