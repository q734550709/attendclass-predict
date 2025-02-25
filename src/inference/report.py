import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")

import sys
import os
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

logger = logging.getLogger(__name__)
class ModelReport:
    def __init__(self, config_path: str) -> None:
        # 获取当前脚本文件所在的目录
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
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

            relative_log_path = log_config['save_path']
            log_path = os.path.normpath(os.path.join(self.current_dir, relative_log_path))
            os.makedirs(log_path, exist_ok=True)
            
            file_handler = logging.FileHandler(
                os.path.join(log_path, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            )
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logging.getLogger().addHandler(file_handler)

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """加载测试数据"""
        relative_data_paths = self.config['data']['paths']

        # 读取测试数据
        X_test = pd.read_csv(os.path.normpath(os.path.join(self.current_dir, relative_data_paths['X_test'])))
        # 保存用户id和截止日期
        user_ids = X_test[['用户id','起始日期时间戳']]
        user_ids['截止日期'] = pd.to_datetime(user_ids['起始日期时间戳'], unit='s')
        
        # 删除用户id后的数据作为特征
        X_test = X_test.drop(columns=['用户id'])

        # 读取标签
        y_test = pd.read_csv(os.path.normpath(os.path.join(self.current_dir, relative_data_paths['y_test']))).squeeze()

        # 创建用户表，用于后续分析
        user_df = user_ids.drop(columns=['起始日期时间戳'])
        # 用户表添加标签
        user_df['标签'] = y_test
        
        logger.info(f"Test data loaded")
        # 返回测试数据
        return X_test, y_test, user_df
    
    def evaluate(self, y_test: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray, model_name: str) -> pd.DataFrame:
        """评估模型"""
        self.metrics = {
            'model_name': [model_name],
            'accuracy': [accuracy_score(y_test, y_pred)],
            'precision': [precision_score(y_test, y_pred)],
            'recall': [recall_score(y_test, y_pred)],
            'f1': [f1_score(y_test, y_pred)],
            'auc_roc': [roc_auc_score(y_test, y_prob)]
        }
        
        metrics_result = pd.DataFrame(self.metrics)
        
        logger.info(f"Test evaluation results: {metrics_result}")

        return metrics_result

    def calculate_prediction_statistics(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """计算预测统计信息"""
        stats = {
            'total_predictions': len(predictions),
            'attendance_count': predictions['预测标签'].sum(),
            'attendance_rate': predictions['预测标签'].mean(),
            'average_confidence': predictions['预测概率'].mean(),
            'confidence_std': predictions['预测概率'].std(),
            'high_confidence_predictions': (
            (predictions['预测概率'] > 0.9).sum() +
            (predictions['预测概率'] < 0.1).sum()
            ),
            'low_confidence_predictions': (
            ((predictions['预测概率'] >= 0.4) &
            (predictions['预测概率'] <= 0.6)).sum()
            )
        }
        stats_result = pd.DataFrame([stats])
        stats_result['attendance_rate'] = stats_result['attendance_rate'].apply(lambda x: f"{x*100:.2f}%")
        stats_result['average_confidence'] = stats_result['average_confidence'].apply(lambda x: f"{x*100:.2f}%")
        stats_result['confidence_std'] = stats_result['confidence_std'].apply(lambda x: f"{x*100:.2f}%")

        logger.info(f"Prediction statistics: {stats_result}")

        return stats_result
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, output_path: str, model_name: str) -> None:
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 保存图片
        output_file = os.path.join(output_path, f'{model_name}_confusion_matrix.png')
        plt.savefig(output_file)
        plt.close()

        logger.info(f"Confusion matrix saved to {output_file}")
        
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                      output_path: str, model_name: str) -> None:
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
        output_file = os.path.join(output_path, f'{model_name}_roc_curve.png')
        plt.savefig(output_file)
        plt.close()

        logger.info(f"ROC curve saved to {output_file}")

    def merge_model_predictions(self, file_list: List[str], model_name_list: List[str]) -> pd.DataFrame:
        """读取多个模型预测的CSV文件并合并成一个DataFrame"""
        df_list = []

        for filedir, model_name in zip(file_list, model_name_list):
            file = filedir + f'/{model_name}_tune_user_shap.csv'
            # 读取CSV文件，读取前几列（假设前5列）
            df = pd.read_csv(file, usecols=[
                '用户id', '截止日期', '标签', '预测标签', '预测概率'
            ], encoding='utf-8')
            
            # 为 标签, 预测标签, 预测概率 列名添加模型类型名称
            df.rename(columns={
                '标签': f'标签_{model_name}',
                '预测标签': f'预测标签_{model_name}',
                '预测概率': f'预测概率_{model_name}'
            }, inplace=True)
            
            df_list.append(df)
        
        # 按照 用户id 和 截止日期 进行合并
        from functools import reduce
        merged_df = reduce(lambda left, right: pd.merge(left, right, on=['用户id', '截止日期']), df_list)
        
        # 计算标签，取所有标签的众数
        label_cols = [f'标签_{model}' for model in model_name_list]
        merged_df['标签'] = merged_df[label_cols].mode(axis=1).iloc[:, 0]

        # 计算综合预测标签，如果预测标签列中有至少3个值为1，则返回1，否则返回0
        # 获取所有预测标签列的列名
        pred_label_cols = [f'预测标签_{model}' for model in model_name_list]
        # 计算每行预测标签为1的数量
        merged_df['预测标签_sum'] = merged_df[pred_label_cols].sum(axis=1)
        # 如果预测标签sum大于等于3，则综合预测标签为1，否则为0
        merged_df['预测标签'] = merged_df['预测标签_sum'].apply(lambda x: 1 if x >= 3 else 0)
        # 删除辅助的预测标签_sum列
        merged_df.drop(columns=['预测标签_sum'], inplace=True)

        # 计算综合预测概率，取所有预测概率的平均值
        pred_prob_cols = [f'预测概率_{model}' for model in model_name_list]
        merged_df['预测概率'] = merged_df[pred_prob_cols].mean(axis=1)
        
        return merged_df
    
def main(config_path: str) -> None:
    """主函数"""
    # 初始化评估器
    report = ModelReport(config_path)
    
    # 获取当前脚本文件所在的目录
    current_dir = report.current_dir
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 获取实验名称和模型目录
    output_dir = config['output']['output_dir']
    csv_dir = config['output']['csv_dir']
    model_name_list = config['model']['model_name']
    model_dir_list = config['model']['model_dir']

    # 创建输出目录
    output_path = os.path.join(current_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(current_dir, csv_dir)
    os.makedirs(csv_path, exist_ok=True)

    # 加载测试数据
    X_test, y_test, user_df = report.load_data()

    # 定义输出表格
    report_df = pd.DataFrame()

    # 按照模型列表加载模型并预测
    for model_name, model_dir in zip(model_name_list, model_dir_list):
        # 加载模型
        model_filename = f'{model_name}_tune_model.joblib'
        model_path = os.path.join(current_dir, model_dir, model_filename)
        model = joblib.load(model_path)
        
        # 预测并记录预测时间
        prediction_start_time = datetime.now()
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        prediction_time = (datetime.now()-prediction_start_time).total_seconds()

        # 用户表添加预测标签和概率
        user_df['预测标签'] = y_pred
        user_df['预测概率'] = y_prob

        # 评估模型
        metrics_result = report.evaluate(y_test, y_pred, y_prob, model_name)
        metrics_result['prediction_time'] = prediction_time

        # 保存评估结果
        stats_result = report.calculate_prediction_statistics(user_df)
        metrics_result = pd.concat([metrics_result, stats_result], axis=1)
        report_df = pd.concat([report_df, metrics_result], ignore_index=True)
        
        # 绘制混淆矩阵
        report.plot_confusion_matrix(y_test, y_pred, output_path, model_name)
        
        # 绘制ROC曲线
        report.plot_roc_curve(y_test, y_prob, output_path, model_name)
    
    # 获取多个模型预测的CSV文件
    model_prediction_list = config['prediction']['prediction_dir']

    # 多模型预测结果合并
    ensemble_start_time = datetime.now()
    merge_df = report.merge_model_predictions(model_prediction_list, model_name_list[1:])
    merge_df_eval = report.evaluate(merge_df['标签'], merge_df['预测标签'], merge_df['预测概率'], 'ensemble')
    ensemble_time = (datetime.now()-ensemble_start_time).total_seconds()
    merge_df_eval['prediction_time'] = ensemble_time

    # 计算合并结果的统计信息
    stats_result = report.calculate_prediction_statistics(merge_df)
    merge_df_eval = pd.concat([merge_df_eval, stats_result], axis=1)

    # 保存合并结果
    report_df = pd.concat([report_df, merge_df_eval], ignore_index=True)
    report_df.to_csv(os.path.join(csv_path, 'report.csv'), index=False)

    # 保存合并结果的混淆矩阵
    report.plot_confusion_matrix(merge_df['标签'], merge_df['预测标签'], output_path, 'ensemble')
    # 保存合并结果的ROC曲线
    report.plot_roc_curve(merge_df['标签'], merge_df['预测概率'], output_path, 'ensemble')

    logger.info("Evaluation completed")

if __name__ == "__main__":
    # 加载数据配置
    config_path = "/home/qikunlyu/文档/attendclass_predict_project/configs/base/report.yaml"
    main(config_path)