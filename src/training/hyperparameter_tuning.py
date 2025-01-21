import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")

import sys
import os
# 添加src目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import pandas as pd
from datetime import datetime
from models.factory import ModelFactory
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import yaml
import logging
import joblib
from typing import Dict, List, Tuple, Any


# 设置日志
logger = logging.getLogger(__name__)
class HyperparameterTuning:
    def __init__(self, config_path: str):
        """初始化超参数调优器"""
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
                os.path.join(log_path, f"tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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

    def grid_search_cv(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict, float]:
        """执行网格搜索"""
        grid_search_params = self.config['grid_search_params']
        grid_search = GridSearchCV(estimator=self.model, **grid_search_params)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_, grid_search.cv_results_

    def random_search_cv(self,  X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict, float]:
        """执行随机搜索"""
        random_search_params = self.config['random_search_params']
        random_search = RandomizedSearchCV(estimator=self.model, **random_search_params)
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_, random_search.best_params_, random_search.best_score_, random_search.cv_results_
        
    def run(self, X_train: pd.DataFrame, y_train: pd.Series, if_random_search: bool = True) -> Tuple[Dict, float]:
        """运行超参数搜索"""
        if if_random_search:
            logger.info("Performing Randomized Search...")
            best_estimator, best_params, best_score, cv_results = self.random_search_cv(X_train, y_train)
        else:
            logger.info("Performing Grid Search...")
            best_estimator, best_params, best_score, cv_results = self.grid_search_cv(X_train, y_train)

        logger.info(f"Best score: {best_score}")
        logger.info(f"Best params: {best_params}")
        
        return best_estimator, best_params, best_score, cv_results
    
    def save_tuner(self, best_estimator: Dict, best_params: Dict, best_score: float, cv_results: pd.DataFrame) -> None:
        """保存调优器"""
        # 保存最佳模型
        model_path = os.path.join(self.exp_model_path, f"{self.exp_name}_tune_model.joblib")
        joblib.dump(best_estimator, model_path)
        logger.info(f"Best model saved at: {model_path}")

        # 保存最佳超参数
        best_params_path = os.path.join(self.exp_path, f"{self.exp_name}_tune_params.yaml")
        with open(best_params_path, 'w') as f:
            yaml.safe_dump(best_params, f)
        logger.info(f"Best params saved at: {best_params_path}")

        # 保存最佳分数
        best_score_path = os.path.join(self.exp_path, f"{self.exp_name}_tune_score.txt")
        with open(best_score_path, 'w') as f:
            f.write(str(best_score))
        logger.info(f"Best score saved at: {best_score_path}")

        # 保存超参数搜索结果
        cv_results_path = os.path.join(self.exp_path, f"{self.exp_name}_tune_cv_results.csv")
        cv_results_df = pd.DataFrame(cv_results)
        cv_results_df.to_csv(cv_results_path, index=False)
        logger.info(f"CV results saved at: {cv_results_path}")


def main(config_path: str, random_search: bool = True) -> Tuple[Dict, Dict, float, pd.DataFrame]:
    """主函数"""
    # 初始化超参数调优器
    tuner = HyperparameterTuning(config_path)
    
    # 加载数据
    X_train, y_train, user_ids = tuner.load_data()

    # 运行超参数搜索
    best_estimator, best_params, best_score, cv_results = tuner.run(X_train, y_train, if_random_search=random_search)

    # 保存调优器
    tuner.save_tuner(best_estimator, best_params, best_score, cv_results)

    logger.info("Hyperparameter tuning completed.")

    return best_estimator, best_params, best_score, cv_results


if __name__ == "__main__":
    # 加载数据配置
    config_path = "../../configs/experiments/exp_preview.yaml"
    main(config_path, random_search=True)
