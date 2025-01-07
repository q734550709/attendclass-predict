import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Tuple
import json

from .predict import AttendancePredictor

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, config_path: str):
        """初始化评估器"""
        self.config = self._load_config(config_path)
        self.metrics = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
        """评估模型性能"""
        metrics = {}
        
        # 基础分类指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred)
        
        # 概率相关指标
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        metrics['average_precision'] = average_precision_score(y_true, y_prob)
        
        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        self.metrics = metrics
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            output_path: str):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 保存图片
        output_file = Path(output_path) / 'confusion_matrix.png'
        plt.savefig(output_file)
        plt.close()
        
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                      output_path: str):
        """绘制ROC曲线"""
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {self.metrics["auc_roc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # 保存图片
        output_file = Path(output_path) / 'roc_curve.png'
        plt.savefig(output_file)
        plt.close()
        
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                  output_path: str):
        """绘制PR曲线"""
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 
                label=f'PR curve (AP = {self.metrics["average_precision"]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        # 保存图片
        output_file = Path(output_path) / 'pr_curve.png'
        plt.savefig(output_file)
        plt.close()
        
    def analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      data: pd.DataFrame) -> pd.DataFrame:
        """分析预测错误"""
        # 创建错误分析DataFrame
        error_df = data.copy()
        error_df['true_label'] = y_true
        error_df['predicted_label'] = y_pred
        error_df['is_error'] = (y_true != y_pred).astype(int)
        
        # 计算各特征的错误率
        error_analysis = {}
        
        # 分析分类特征
        categorical_features = self._get_categorical_features(error_df)
        for feature in categorical_features:
            error_analysis[feature] = error_df.groupby(feature)['is_error'].mean()
            
        # 分析数值特征
        numerical_features = self._get_numerical_features(error_df)
        for feature in numerical_features:
            error_analysis[feature] = {
                'correlation': error_df[feature].corr(error_df['is_error']),
                'error_mean': error_df[error_df['is_error'] == 1][feature].mean(),
                'correct_mean': error_df[error_df['is_error'] == 0][feature].mean()
            }
            
        return pd.DataFrame(error_analysis)
    
    def save_results(self, output_path: str):
        """保存评估结果"""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存指标
        metrics_file = output_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
            
        logger.info(f"Evaluation results saved to {output_dir}")
    
    def _get_categorical_features(self, df: pd.DataFrame) -> List[str]:
        """获取分类特征"""
        return df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def _get_numerical_features(self, df: pd.DataFrame) -> List[str]:
        """获取数值特征"""
        return df.select_dtypes(include=['int64', 'float64']).columns.tolist()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='模型评估')
    parser.add_argument('--model-path', type=str, required=True,
                      help='模型路径')
    parser.add_argument('--config-path', type=str, required=True,
                      help='配置文件路径')
    parser.add_argument('--test-data', type=str, required=True,
                      help='测试数据路径')
    parser.add_argument('--output-path', type=str, required=True,
                      help='输出结果路径')
    
    args = parser.parse_args()
    
    # 加载测试数据
    test_df = pd.read_csv(args.test_data)
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    # 初始化预测器和评估器
    predictor = AttendancePredictor(args.model_path, args.config_path)
    evaluator = ModelEvaluator(args.config_path)
    
    # 获取预测结果
    y_pred = predictor.predict(X_test)
    y_prob = predictor.predict_proba(X_test)
    
    # 评估模型
    metrics = evaluator.evaluate(y_test, y_pred, y_prob)
    
    # 生成可视化
    evaluator.plot_confusion_matrix(y_test, y_pred, args.output_path)
    evaluator.plot_roc_curve(y_test, y_prob, args.output_path)
    evaluator.plot_precision_recall_curve(y_test, y_prob, args.output_path)
    
    # 错误分析
    error_analysis = evaluator.analyze_errors(y_test, y_pred, X_test)
    
    # 保存结果
    evaluator.save_results(args.output_path)

if __name__ == "__main__":
    main() 