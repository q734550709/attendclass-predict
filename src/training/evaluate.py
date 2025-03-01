import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")

import sys
import os
import shap
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
    def __init__(self, config_path: str, model_type: str) -> None:
        # 获取当前脚本文件所在的目录
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        """初始化评估器"""
        self.config = self._load_config(config_path)
        self.metrics = {}
                
        # 设置日志
        self.setup_logging(model_type)

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self, model_type: str) -> None:
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
                os.path.join(log_path, f"evaluate_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            )
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logging.getLogger().addHandler(file_handler)

    def _standardize_feature_names(self, X: pd.DataFrame) -> pd.DataFrame:
        """标准化特征名称
        
        将特征名称中的特殊字符替换为下划线，确保特征名称的一致性
        
        Args:
            X: 输入的特征数据框
            
        Returns:
            pd.DataFrame: 处理后的特征数据框
        """
        # 创建一个新的DataFrame，避免修改原始数据
        X = X.copy()
        
        # 特征名称映射字典
        rename_dict = {}
        for col in X.columns:
            # 将特殊字符替换为下划线
            new_name = col.replace(' ', '_')
            # 确保不会产生重复的列名
            if new_name in rename_dict.values():
                i = 1
                while f"{new_name}_{i}" in rename_dict.values():
                    i += 1
                new_name = f"{new_name}_{i}"
            rename_dict[col] = new_name
        
        # 重命名列
        X = X.rename(columns=rename_dict)
        
        # 记录特征名称的变化
        if rename_dict:
            changes = [f"{old} -> {new}" for old, new in rename_dict.items() if old != new]
            if changes:
                logger.info("Feature names standardized:")
                for change in changes:
                    logger.info(change)
        
        return X
    
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

        # 标准化特征名称
        X_test = self._standardize_feature_names(X_test)

        # 读取标签
        y_test = pd.read_csv(os.path.normpath(os.path.join(self.current_dir, relative_data_paths['y_test']))).squeeze()

        # 创建用户表，用于后续分析
        user_df = user_ids.drop(columns=['起始日期时间戳'])
        # 用户表添加标签
        user_df['标签'] = y_test
        
        logger.info(f"Test data loaded")
        # 返回测试数据
        return X_test, y_test, user_df

    
    def calculate_shap_values(self, model: Any, X_test: pd.DataFrame) -> pd.DataFrame:
        # 根据模型类型创建解释器 
        if self.config['experiment']['name'] in ['decision_tree', 'gbdt', 'random_forest', 'xgboost', 'lightgbm']:
            explainer = shap.TreeExplainer(model, X_test)
            explainer_test = explainer(X_test, check_additivity=False)
        else:
            # 使用 shap.sample 或 shap.kmeans 减少背景数据样本
            background = shap.sample(X_test, 100)  # 选择 100 个样本作为背景数据
            explainer = shap.KernelExplainer(model.predict, background)
            explainer_test = explainer.shap_values(X_test)

        shap_values = explainer_test.values
        base_values = explainer_test.base_values

        # 获取特征名称
        feature_names = X_test.columns
        feature_len = len(feature_names)

        # 计算每个样本的前10个特征的shap值贡献
        shap_results = []
        for i in range(len(X_test)):
            # 处理 base_values，根据维度判断
            if base_values.ndim == 1:
                # base_values 形状为 (n_samples,)
                base_value = base_values[i]
            elif base_values.ndim == 2:
                # base_values 形状为 (n_samples, n_classes)
                base_value = base_values[i][1]
            else:
                raise ValueError("Unexpected number of dimensions in base_values")

            # 处理 shap_values，根据维度判断
            if shap_values.ndim == 2:
                # shap_values 形状为 (n_samples, n_features)
                shap_value = shap_values[i]
            elif shap_values.ndim == 3:
                # shap_values 形状为 (n_samples, n_classes, n_features)
                shap_value = shap_values[i][1]
            else:
                raise ValueError("Unexpected number of dimensions in shap_values")

            # 将 shap_value 与 base_value 相加
            shap_value = shap_value + base_value/feature_len

            # 将特征名称和shap值对应起来
            shap_feature_importance = dict(zip(feature_names, shap_value))
            # 根据绝对值排序，获取前10个特征
            sorted_features = sorted(shap_feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            # 创建一个字典来保存结果，包括前10个特征的贡献值
            instance_result = {}
            # 保存前10个特征的名称和贡献值
            for idx, (feature, contribution) in enumerate(sorted_features):
                instance_result[f'Feature_{idx+1}_Name'] = feature
                instance_result[f'Feature_{idx+1}_Contribution'] = contribution
            shap_results.append(instance_result)
        
        # 将结果转换为DataFrame
        shap_df = pd.DataFrame(shap_results)
        
        logger.info(f"SHAP values calculated")

        return shap_df


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
                            output_path: str, exp_name: str, model_type: str) -> None:
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 保存图片
        output_file = os.path.join(output_path, f'{exp_name}_{model_type}_confusion_matrix.png')
        plt.savefig(output_file)
        plt.close()

        logger.info(f"Confusion matrix saved to {output_file}")
        
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                      output_path: str, exp_name: str, model_type: str) -> None:
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
        output_file = os.path.join(output_path, f'{exp_name}_{model_type}_roc_curve.png')
        plt.savefig(output_file)
        plt.close()

        logger.info(f"ROC curve saved to {output_file}")
    
    def plot_feature_importance(self, model: Any, feature_names: List[str], 
                                output_path: str, exp_name: str, model_type: str) -> None:
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
        output_file = os.path.join(output_path, f'{exp_name}_{model_type}_feature_importance.png')
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        
        # 保存特征重要性数据
        importance_df.to_csv(
            os.path.join(output_path, f'{exp_name}_{model_type}_feature_importance.csv'),
            index=False
        )

        logger.info(f"Feature importance saved to {output_file}")

def main(config_path: str, model_type: str) -> None:
    """主函数"""
    # 初始化评估器
    evaluator = ModelEvaluator(config_path, model_type)

    # 获取当前脚本文件所在的目录
    current_dir = evaluator.current_dir
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 获取实验名称和模型目录
    exp_name = config['experiment']['name']
    exp_dir = config['output']['exp_dir']
    exp_model_dir = config['output']['exp_model_dir']
    
    # 创建实验目录
    exp_path = os.path.join(current_dir, exp_dir, f'{model_type}_evaluate')
    os.makedirs(exp_path, exist_ok=True)
    
    # 创建模型目录
    exp_model_path = os.path.join(current_dir, exp_model_dir)
    os.makedirs(exp_model_path, exist_ok=True)

    # 根据模型类型加载模型
    model_filename = f'{exp_name}_{model_type}_model.joblib'
    model_path = os.path.join(exp_model_path, model_filename)
    model = joblib.load(model_path)
    
    # 加载测试数据
    X_test, y_test, user_df = evaluator.load_data()

    # 获取预测结果
    y_pred = model.predict(X_test)
    predict_proba_result = model.predict_proba(X_test)
    y_prob = predict_proba_result[:, 1]

    # 如果model是树模型，计算shap值
    if exp_name in ['decision_tree', 'gbdt', 'random_forest', 'xgboost', 'lightgbm']:
        # 计算shap值
        shap_df = evaluator.calculate_shap_values(model, X_test)

        # 用户表添加预测标签和概率
        user_df['预测标签'] = y_pred
        user_df['预测概率'] = y_prob

        # 用户表和shap值表按照列拼接
        user_shap_df = pd.concat([user_df, shap_df], axis=1)

        # 保存用户表和shap值表
        user_shap_path = os.path.join(exp_path, f'{exp_name}_{model_type}_user_shap.csv')
        user_shap_df.to_csv(user_shap_path, index=False)
        logger.info(f"User and SHAP values saved to {user_shap_path}")

    # 评估模型
    metrics_result = evaluator.evaluate(y_test, y_pred, y_prob)
    
    # 保存评估结果
    evaluation_path = os.path.join(exp_path, f'{exp_name}_{model_type}_test_evaluation.csv')
    metrics_result.to_csv(evaluation_path, index=False)
    logger.info(f"Evaluation results saved to {evaluation_path}")

    # 生成可视化
    evaluator.plot_confusion_matrix(y_test, y_pred, exp_path, exp_name, model_type)
    evaluator.plot_roc_curve(y_test, y_prob, exp_path, exp_name, model_type)
    evaluator.plot_feature_importance(model, X_test.columns, exp_path, exp_name, model_type)    
    logger.info("Evaluation visualizations saved")

    # 移除控制台处理程序
    logging.getLogger().handlers = []

if __name__ == "__main__":
    # 加载数据配置
    config_path = "/home/qikunlyu/文档/attendclass_predict_project/configs/experiments/target1/model_stack/train_config.yaml"
    main(config_path, 'train')
