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
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV, HalvingGridSearchCV
import yaml
import logging
import joblib
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

import optuna
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
from optuna.importance import get_param_importances
import optuna.visualization as vis
import hiplot as hip
import plotly.io as pio


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
        self.exp_path = os.path.join(self.current_dir, self.exp_dir, 'tune_result')
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
        
        logger.info("Data loaded successfully")
        return X_train, y_train, user_ids

    def random_search_cv(self,  X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict, float]:
        """执行随机搜索"""
        random_search_params = self.config['random_search_params']
        # 处理param_distributions中的range字符串
        param_distributions = random_search_params['param_distributions']
        param_distributions = {k: (list(eval(v)) if isinstance(v, str) and v.startswith('range') else v) for k, v in param_distributions.items()}
        random_search_params['param_distributions'] = param_distributions

        random_search = RandomizedSearchCV(estimator=self.model, **random_search_params)
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_, random_search.best_params_, random_search.best_score_, random_search.cv_results_
    
    def grid_search_cv(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict, float]:
        """执行网格搜索"""
        grid_search_params = self.config['grid_search_params']
        # 处理param_grid中的range字符串
        param_grid = grid_search_params['param_grid']
        param_grid = {k: (list(eval(v)) if isinstance(v, str) and v.startswith('range') else v) for k, v in param_grid.items()}
        grid_search_params['param_grid'] = param_grid

        grid_search = GridSearchCV(estimator=self.model, **grid_search_params)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_, grid_search.cv_results_

    def halving_random_search_cv(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict, float]:
        """执行随机搜索"""
        random_search_params = self.config['halving_random_search_params']
        # 处理param_distributions中的range字符串
        param_distributions = random_search_params['param_distributions']
        param_distributions = {k: (list(eval(v)) if isinstance(v, str) and v.startswith('range') else v) for k, v in param_distributions.items()}
        random_search_params['param_distributions'] = param_distributions
        random_search = HalvingRandomSearchCV(estimator=self.model, **random_search_params)
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_, random_search.best_params_, random_search.best_score_, random_search.cv_results_

    def halving_grid_search_cv(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict, float]:
        """执行网格搜索"""
        grid_search_params = self.config['halving_grid_search_params']
        # 处理param_grid中的range字符串
        param_grid = grid_search_params['param_grid']
        param_grid = {k: (list(eval(v)) if isinstance(v, str) and v.startswith('range') else v) for k, v in param_grid.items()}
        grid_search_params['param_grid'] = param_grid
        
        grid_search = HalvingGridSearchCV(estimator=self.model, **grid_search_params)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_, grid_search.cv_results

    def convert_to_optuna_distributions(self, search_params: Dict) -> Dict:
        """将随机搜索参数转换为Optuna分布"""
        param_distributions = {}
        for param_name, param_values in search_params.items():
            if isinstance(param_values, np.ndarray) or isinstance(param_values, list):
                values_type = type(param_values[0]) if len(param_values) > 0 else None
                if values_type == int:
                    param_distributions[param_name] = IntDistribution(min(param_values), max(param_values))
                elif values_type == float:
                    param_distributions[param_name] = FloatDistribution(min(param_values), max(param_values))
                else:
                    param_distributions[param_name] = CategoricalDistribution(param_values)
            else:
                raise ValueError(f"Unsupported type for parameter values: {param_name}")
        return param_distributions
        
    def run(self, X_train: pd.DataFrame, y_train: pd.Series, tune_type: str = 'random') -> Tuple[Dict, Dict, float, pd.DataFrame]:
        """运行超参数搜索"""
        if tune_type == 'random':
            logger.info("Performing Randomized Search...")
            best_estimator, best_params, best_score, cv_results = self.random_search_cv(X_train, y_train)
        elif tune_type == 'grid':
            logger.info("Performing Grid Search...")
            best_estimator, best_params, best_score, cv_results = self.grid_search_cv(X_train, y_train)
        elif tune_type == 'halving_random':
            logger.info("Performing Halving Randomized Search...")
            best_estimator, best_params, best_score, cv_results = self.halving_random_search_cv(X_train, y_train)
        elif tune_type == 'halving_grid':
            logger.info("Performing Halving Grid Search...")
            best_estimator, best_params, best_score, cv_results = self.halving_grid_search_cv(X_train, y_train)
        else:
            raise ValueError(f"Unsupported tuning type: {tune_type}")

        logger.info(f"Best score: {best_score}")
        logger.info(f"Best params: {best_params}")
        
        return best_estimator, best_params, best_score, cv_results
    
    def optuna_study(self, cv_results: pd.DataFrame, tune_type: str = 'random') -> Dict:
        """使用Optuna重要性评估器调优"""
        # 创建Optuna Study
        study = optuna.create_study(direction='maximize')

        # 将试验数据添加到Study
        trials_data = []
        for i in range(len(cv_results['params'])):
            trial_data = {
                'params': cv_results['params'][i],
                'value': cv_results['mean_test_score'][i]
            }
            trials_data.append(trial_data)

        # 定义参数分布
        if tune_type == 'random':
            param_distributions = self.convert_to_optuna_distributions(self.config['random_search_params']['param_distributions'])
        elif tune_type == 'grid':
            param_distributions = self.convert_to_optuna_distributions(self.config['grid_search_params']['param_grid'])
        elif tune_type == 'halving_random':
            param_distributions = self.convert_to_optuna_distributions(self.config['halving_random_search_params']['param_distributions'])
        elif tune_type == 'halving_grid':
            param_distributions = self.convert_to_optuna_distributions(self.config['halving_grid_search_params']['param_grid'])
        else:
            raise ValueError(f"Unsupported tuning type: {tune_type}")

        # 添加试验数据到Study
        for trial_data in trials_data:
            if not np.isnan(trial_data['value']):  # 检查 value 是否为 NaN
                trial = optuna.trial.create_trial(
                    params=trial_data['params'],
                    distributions=param_distributions,
                    value=trial_data['value']
                )
                study.add_trial(trial)
        logger.info("Optuna study created successfully")
        return study

    def plot_and_save_optuna_result(self, study: optuna.study.Study) -> None:
        """绘制Optuna结果并保存"""
        # 计算参数重要性
        importances = get_param_importances(study)

        # 输出参数重要性
        logger.info("Parameter importances:")
        for param, importance in importances.items():
            logger.info(f"{param}: {importance:.4f}")

        # 优化历史图（Optimization History Plot）
        fig_optimization_history = vis.plot_optimization_history(study)
        # 参数重要性图（Parameter Importance Plot）
        fig_param_importances = vis.plot_param_importances(study)
        # 平行坐标图（Parallel Coordinates Plot）
        fig_parallel_coordinates = vis.plot_parallel_coordinate(study)
        # 轮廓图（Contour Plot）
        fig_contour = vis.plot_contour(study)
        # 参数相关图（Parameter Correlation Plot）
        fig_param_correlations = vis.plot_slice(study)
        # 参数分布箱线图（Parameter Distribution Box Plot）
        trials = study.trials_dataframe()
        params = trials.filter(like='params')

        
        # 创建保存路径
        tune_box_plot_path = os.path.join(self.exp_path, 'tune_box_plot')
        os.makedirs(tune_box_plot_path, exist_ok=True)
        # 保存参数分布箱线图
        for col in params.columns:
            plt.figure()
            plt.hist(params[col], bins=20)
            plt.title(f'Histogram of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(tune_box_plot_path, f"{self.exp_name}_param_distribution_{col}.png"))
            plt.close()

        logger.info(f"Optuna box plot saved at: {tune_box_plot_path}")

        # 创建保存路径
        tune_plot_html_path = os.path.join(self.exp_path, 'tune_plot_html')
        os.makedirs(tune_plot_html_path, exist_ok=True)
        # 保存Optuna的html图
        pio.write_html(fig_optimization_history, os.path.join(tune_plot_html_path, f"{self.exp_name}_opt_history.html"))
        pio.write_html(fig_param_importances, os.path.join(tune_plot_html_path, f"{self.exp_name}_param_importances.html"))
        pio.write_html(fig_parallel_coordinates, os.path.join(tune_plot_html_path, f"{self.exp_name}_parallel_coordinates.html"))
        pio.write_html(fig_contour, os.path.join(tune_plot_html_path, f"{self.exp_name}_contour.html"))
        pio.write_html(fig_param_correlations, os.path.join(tune_plot_html_path, f"{self.exp_name}_param_correlations.html"))

        logger.info(f"Optuna html plots saved at: {tune_plot_html_path}")
        
        # 创建保存路径
        tune_plot_png_path = os.path.join(self.exp_path, 'tune_plot_png')
        os.makedirs(tune_plot_png_path, exist_ok=True)
        # 保存Optuna的png图
        pio.write_image(fig_optimization_history, os.path.join(tune_plot_png_path, f"{self.exp_name}_opt_history.png"))
        pio.write_image(fig_param_importances, os.path.join(tune_plot_png_path, f"{self.exp_name}_param_importances.png"))
        pio.write_image(fig_parallel_coordinates, os.path.join(tune_plot_png_path, f"{self.exp_name}_parallel_coordinates.png"))
        pio.write_image(fig_contour, os.path.join(tune_plot_png_path, f"{self.exp_name}_contour.png"))
        pio.write_image(fig_param_correlations, os.path.join(tune_plot_png_path, f"{self.exp_name}_param_correlations.png"))

        logger.info(f"Optuna png plots saved at: {tune_plot_png_path}")
        
        # 创建保存路径
        tune_plot_hiplot_path = os.path.join(self.exp_path, 'tune_plot_png')
        os.makedirs(tune_plot_hiplot_path, exist_ok=True)
        # 保存HiPlot图
        hip_experiment = hip.Experiment.from_optuna(study)
        hip_experiment.to_html(os.path.join(tune_plot_hiplot_path, f"{self.exp_name}_hip_plot.html"))

        logger.info(f"Optuna hiplot plots saved at: {tune_plot_hiplot_path}")

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


def main(config_path: str, tune_type: str = 'random') -> Tuple[Dict, Dict, float, pd.DataFrame]:
    """主函数"""
    # 初始化超参数调优器
    tuner = HyperparameterTuning(config_path)
    
    # 加载数据
    X_train, y_train, user_ids = tuner.load_data()

    # 运行超参数搜索
    best_estimator, best_params, best_score, cv_results = tuner.run(X_train, y_train, tune_type=tune_type)

    # 使用Optuna重要性评估器调优
    optuna_study = tuner.optuna_study(cv_results, tune_type=tune_type)

    # 绘制并保存Optuna结果
    tuner.plot_and_save_optuna_result(optuna_study)

    # 保存调优器
    tuner.save_tuner(best_estimator, best_params, best_score, cv_results)

    logger.info("Hyperparameter tuning completed.")

    # 移除控制台处理程序
    logging.getLogger().handlers = []

    return best_estimator, best_params, best_score, cv_results


if __name__ == "__main__":
    # 加载数据配置
    config_path = "../../configs/experiments/exp_preview.yaml"
    main(config_path, tune_type='random')
