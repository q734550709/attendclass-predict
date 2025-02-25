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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV, HalvingGridSearchCV, cross_val_score
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

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import StackingClassifier

# 添加贝叶斯优化相关的导入
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver
from skopt.plots import plot_convergence, plot_objective, plot_evaluations

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
        
        if model_config.get('is_stacking', False):
            # 构建特征选择器（如果启用）
            steps = []
            if model_config.get('feature_selector', {}).get('enabled', False):
                selector_config = model_config['feature_selector']
                feature_selector = SelectFromModel(
                    ModelFactory.create_model(selector_config['type'], selector_config['params']),
                    threshold=selector_config.get('threshold', 'mean')
                )
                steps.append(('feature_selection', feature_selector))
            
            # 构建基础模型
            estimators = []
            for base_model in model_config['base_models']:
                model = ModelFactory.create_model(base_model['type'], base_model['params'])
                estimators.append((base_model['name'], model))
            
            # 构建元学习器
            meta_config = model_config['meta_learner']
            final_estimator = ModelFactory.create_model(meta_config['type'], meta_config['params'])
            
            # 构建Stacking模型
            stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                cv=self.config.get('cross_validation', {}).get('n_splits', 5)
            )
            steps.append(('stacking', stacking))
            
            return Pipeline(steps)
        else:
            return ModelFactory.create_model(model_config['type'], model_config['params'])
    
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
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """加载训练数据"""
        relative_data_path = self.config['data']['paths']
        
        # 读取数据
        X_train = pd.read_csv(os.path.normpath(os.path.join(self.current_dir, relative_data_path['X_train'])))
        # 保存用户id
        user_ids = X_train['用户id']
        # 删除用户id后的数据作为特征
        X_train = X_train.drop(columns=['用户id'])

        # 标准化特征名称
        X_train = self._standardize_feature_names(X_train)
        
        # 读取标签
        y_train = pd.read_csv(os.path.normpath(os.path.join(self.current_dir, relative_data_path['y_train']))).squeeze()
        
        logger.info("Data loaded successfully")
        return X_train, y_train, user_ids

    def _prepare_param_grid_for_stacking(self, param_config: dict, is_distribution: bool = False) -> dict:
        """准备多层模型的参数网格"""
        param_grid = {}
        
        # 如果启用了特征选择器
        if self.config['model']['feature_selector']['enabled']:
            # 添加特征选择器的模型参数
            feature_selector_params = param_config['feature_selector_param_grid' if not is_distribution 
                                                else 'feature_selector_param_distributions']
            for param, values in feature_selector_params.items():
                param_grid[f'feature_selection__estimator__{param}'] = values
            
            # 添加特征选择阈值
            threshold_values = param_config['feature_selector_threshold']
            param_grid['feature_selection__threshold'] = threshold_values
        
        # 为每个基础模型添加参数
        base_models_params = param_config['base_models_param_grid' if not is_distribution 
                                        else 'base_models_param_distributions']
        for model_name, model_params in base_models_params.items():
            for param, values in model_params.items():
                param_grid[f'stacking__{model_name}__{param}'] = values
        
        # 为元学习器添加参数
        meta_params = param_config['meta_learner_param_grid' if not is_distribution 
                                else 'meta_learner_param_distributions']
        for param, values in meta_params.items():
            param_grid[f'stacking__final_estimator__{param}'] = values
        
        return param_grid

    def random_search_cv(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict, float]:
        """执行随机搜索"""
        random_search_params = self.config['random_search_params']
        
        if self.config['model']['is_stacking']:
            # 处理多层模型参数
            param_distributions = self._prepare_param_grid_for_stacking(random_search_params, is_distribution=True)
        else:
            # 处理单一模型参数
            param_distributions = random_search_params['param_distributions']
        
        search_params = {
            'estimator': self.model,
            'param_distributions': param_distributions,
            'n_iter': random_search_params['n_iter'],
            'scoring': random_search_params['scoring'],
            'cv': random_search_params['cv'],
            'n_jobs': random_search_params['n_jobs'],
            'random_state': random_search_params['random_state'],
            'return_train_score': random_search_params['return_train_score']
        }
        
        random_search = RandomizedSearchCV(**search_params)
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_, random_search.best_params_, random_search.best_score_, random_search.cv_results_

    def grid_search_cv(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict, float]:
        """执行网格搜索"""
        grid_search_params = self.config['grid_search_params']
        
        if self.config['model']['is_stacking']:
            # 处理多层模型参数
            param_grid = self._prepare_param_grid_for_stacking(grid_search_params)
        else:
            # 处理单一模型参数
            param_grid = grid_search_params['param_grid']
        
        search_params = {
            'estimator': self.model,
            'param_grid': param_grid,
            'scoring': grid_search_params['scoring'],
            'cv': grid_search_params['cv'],
            'n_jobs': grid_search_params['n_jobs'],
            'return_train_score': grid_search_params['return_train_score']
        }
        
        grid_search = GridSearchCV(**search_params)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_, grid_search.cv_results_

    def halving_random_search_cv(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict, float]:
        """执行Halving随机搜索"""
        halving_random_params = self.config['halving_random_search_params']
        
        if self.config['model']['is_stacking']:
            # 处理多层模型参数
            param_distributions = self._prepare_param_grid_for_stacking(halving_random_params, is_distribution=True)
        else:
            # 处理单一模型参数
            param_distributions = halving_random_params['param_distributions']
        
        search_params = {
            'estimator': self.model,
            'param_distributions': param_distributions,
            'n_candidates': halving_random_params['n_candidates'],
            'factor': halving_random_params['factor'],
            'resource': halving_random_params['resource'],
            'scoring': halving_random_params['scoring'],
            'cv': halving_random_params['cv'],
            'refit': halving_random_params['refit'],
            'n_jobs': halving_random_params['n_jobs'],
            'random_state': halving_random_params['random_state'],
            'return_train_score': halving_random_params['return_train_score']
        }
        
        halving_random_search = HalvingRandomSearchCV(**search_params)
        halving_random_search.fit(X_train, y_train)
        return halving_random_search.best_estimator_, halving_random_search.best_params_, halving_random_search.best_score_, halving_random_search.cv_results_

    def halving_grid_search_cv(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict, float]:
        """执行Halving网格搜索"""
        halving_grid_params = self.config['halving_grid_search_params']
        
        if self.config['model']['is_stacking']:
            # 处理多层模型参数
            param_grid = self._prepare_param_grid_for_stacking(halving_grid_params)
        else:
            # 处理单一模型参数
            param_grid = halving_grid_params['param_grid']
        
        search_params = {
            'estimator': self.model,
            'param_grid': param_grid,
            'factor': halving_grid_params['factor'],
            'resource': halving_grid_params['resource'],
            'scoring': halving_grid_params['scoring'],
            'cv': halving_grid_params['cv'],
            'refit': halving_grid_params['refit'],
            'n_jobs': halving_grid_params['n_jobs'],
            'random_state': halving_grid_params['random_state'],
            'return_train_score': halving_grid_params['return_train_score']
        }
        
        halving_grid_search = HalvingGridSearchCV(**search_params)
        halving_grid_search.fit(X_train, y_train)
        return halving_grid_search.best_estimator_, halving_grid_search.best_params_, halving_grid_search.best_score_, halving_grid_search.cv_results_

    def convert_to_optuna_distributions(self, search_params: Dict) -> Dict:
        """将搜索参数转换为Optuna分布
        
        支持以下格式的参数:
        1. 列表/数组格式 (随机/网格搜索)
        2. 字典格式 (贝叶斯优化), 包含type, low, high等键
        
        Args:
            search_params: 搜索参数配置字典
            
        Returns:
            转换后的Optuna分布字典
        """
        param_distributions = {}
        
        # 处理嵌套的参数空间(用于stacking模型)
        if any(key in search_params for key in ['feature_selector_param_space', 'base_models_param_space', 'meta_learner_param_space']):
            # 处理特征选择器参数
            if 'feature_selector_param_space' in search_params:
                feature_selector_distributions = self._convert_param_space(search_params['feature_selector_param_space'])
                param_distributions.update({f'feature_selection__estimator__{k}': v 
                                        for k, v in feature_selector_distributions.items()})
            
            # 处理特征选择阈值
            if 'feature_selector_threshold' in search_params:
                threshold_distribution = self._convert_param_space({'threshold': search_params['feature_selector_threshold']})
                param_distributions['feature_selection__threshold'] = threshold_distribution['threshold']
            
            # 处理基础模型参数
            if 'base_models_param_space' in search_params:
                for model_name, model_params in search_params['base_models_param_space'].items():
                    model_distributions = self._convert_param_space(model_params)
                    param_distributions.update({f'stacking__{model_name}__{k}': v 
                                            for k, v in model_distributions.items()})
            
            # 处理元学习器参数
            if 'meta_learner_param_space' in search_params:
                meta_distributions = self._convert_param_space(search_params['meta_learner_param_space'])
                param_distributions.update({f'stacking__final_estimator__{k}': v 
                                        for k, v in meta_distributions.items()})
                
            return param_distributions
        
        # 处理普通参数空间
        return self._convert_param_space(search_params)

    def _convert_param_space(self, param_space: Dict) -> Dict:
        """转换单个参数空间为Optuna分布
        
        Args:
            param_space: 参数空间配置
            
        Returns:
            转换后的Optuna分布字典
        """
        distributions = {}
        
        for param_name, param_config in param_space.items():
            # 处理贝叶斯优化格式
            if isinstance(param_config, dict) and 'type' in param_config:
                param_type = param_config['type'].lower()
                
                if param_type == 'int':
                    distributions[param_name] = IntDistribution(
                        low=param_config['low'],
                        high=param_config['high'],
                        log=param_config.get('prior', '') == 'log-uniform'
                    )
                elif param_type == 'real':
                    distributions[param_name] = FloatDistribution(
                        low=param_config['low'],
                        high=param_config['high'],
                        log=param_config.get('prior', '') == 'log-uniform'
                    )
                elif param_type == 'categorical':
                    distributions[param_name] = CategoricalDistribution(
                        choices=param_config['choices']
                    )
                else:
                    raise ValueError(f"Unsupported parameter type: {param_type}")
                    
            # 处理列表/数组格式(随机/网格搜索)
            elif isinstance(param_config, (list, np.ndarray)):
                if len(param_config) == 0:
                    raise ValueError(f"Parameter values for {param_name} are empty")
                    
                values_type = type(param_config[0])
                if all(isinstance(v, values_type) for v in param_config):
                    if values_type == int:
                        distributions[param_name] = IntDistribution(
                            low=min(param_config),
                            high=max(param_config)
                        )
                    elif values_type == float:
                        distributions[param_name] = FloatDistribution(
                            low=min(param_config),
                            high=max(param_config)
                        )
                    elif values_type == str:
                        distributions[param_name] = CategoricalDistribution(
                            choices=param_config
                        )
                    else:
                        raise ValueError(f"Unsupported value type for parameter: {param_name}")
                else:
                    raise ValueError(f"Inconsistent value types in parameter: {param_name}")
            else:
                raise ValueError(f"Unsupported parameter configuration format: {param_name}")
                
        return distributions
        
    def _convert_to_skopt_space(self, param_config: Dict) -> Dict:
        """将配置参数转换为scikit-optimize空间"""
        space = {}
        
        for param_name, param_info in param_config.items():
            if param_info['type'] == 'real':
                space[param_name] = Real(
                    low=param_info['low'],
                    high=param_info['high'],
                    prior=param_info.get('prior', 'uniform')
                )
            elif param_info['type'] == 'int':
                space[param_name] = Integer(
                    low=param_info['low'],
                    high=param_info['high']
                )
            elif param_info['type'] == 'categorical':
                space[param_name] = Categorical(
                    categories=param_info['choices']
                )
        return space

    def _prepare_bayesian_space_for_stacking(self, param_config: Dict) -> Dict:
        """准备多层模型的贝叶斯优化空间"""
        search_space = {}
        
        # 如果启用了特征选择器
        if self.config['model']['feature_selector']['enabled']:
            # 添加特征选择器的模型参数
            feature_selector_space = self._convert_to_skopt_space(param_config['feature_selector_param_space'])
            for param, space in feature_selector_space.items():
                search_space[f'feature_selection__estimator__{param}'] = space
            
            # 添加特征选择阈值
            threshold_space = self._convert_to_skopt_space({'threshold': param_config['feature_selector_threshold']})
            search_space['feature_selection__threshold'] = threshold_space['threshold']
        
        # 为每个基础模型添加参数
        base_models_space = param_config['base_models_param_space']
        for model_name, model_params in base_models_space.items():
            model_space = self._convert_to_skopt_space(model_params)
            for param, space in model_space.items():
                search_space[f'stacking__{model_name}__{param}'] = space
        
        # 为元学习器添加参数
        meta_space = self._convert_to_skopt_space(param_config['meta_learner_param_space'])
        for param, space in meta_space.items():
            search_space[f'stacking__final_estimator__{param}'] = space
        
        return search_space

    def bayesian_search_cv(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict, float]:
        """执行贝叶斯优化搜索"""
        bayesian_params = self.config['bayesian_search_params']
        
        # 验证采集函数类型
        valid_acq_funcs = ['EI', 'PI', 'LCB', 'gp_hedge']  # 更新支持的采集函数
        acq_func = bayesian_params['acquisition_function'].upper()  # 转换为大写
        if acq_func not in valid_acq_funcs:
            logger.warning(f"Invalid acquisition function: {acq_func}. Using 'EI' instead.")
            acq_func = 'EI'
        
        # 验证和调整其他参数
        n_trials = max(1, bayesian_params.get('n_trials', 100))
        n_startup_trials = min(bayesian_params.get('n_startup_trials', 10), n_trials)
        n_points = max(1, bayesian_params.get('n_points', 1))
        
        if self.config['model']['is_stacking']:
            # 处理多层模型参数
            search_space = self._prepare_bayesian_space_for_stacking(bayesian_params)
        else:
            # 处理单一模型参数
            search_space = self._convert_to_skopt_space(bayesian_params['param_space'])
        
        # 创建检查点保存器
        checkpoint_path = os.path.join(self.exp_path, 'bayesian_checkpoint')
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_path, f"{self.exp_name}_bayesian_checkpoint.pkl")
        checkpoint_saver = CheckpointSaver(checkpoint_file, compress=9)
        
        # 配置贝叶斯搜索
        search_params = {
            'estimator': self.model,
            'search_spaces': search_space,
            'n_iter': n_trials,
            'n_points': n_points,
            'scoring': bayesian_params['scoring'],
            'cv': bayesian_params['cv'],
            'n_jobs': bayesian_params['n_jobs'],
            'random_state': bayesian_params['random_state'],
            'return_train_score': bayesian_params['return_train_score'],
            'optimizer_kwargs': {
                'base_estimator': 'GP',
                'acq_func': acq_func,
                'n_initial_points': n_startup_trials
            }
        }
        
        bayes_search = BayesSearchCV(**search_params)
        
        # 执行优化
        logger.info("Starting Bayesian optimization...")
        bayes_search.fit(X_train, y_train, callback=[checkpoint_saver])
        
        # 记录优化结果
        logger.info(f"Number of completed trials: {len(bayes_search.cv_results_['mean_test_score'])}")
        
        # 保存优化过程的可视化结果
        self._save_bayesian_plots(bayes_search)
        
        return bayes_search.best_estimator_, bayes_search.best_params_, bayes_search.best_score_, bayes_search.cv_results_

    def _save_bayesian_plots(self, bayes_search: BayesSearchCV) -> None:
        """保存贝叶斯优化的可视化结果"""
        # 创建保存路径
        plots_path = os.path.join(self.exp_path, 'bayesian_plots')
        os.makedirs(plots_path, exist_ok=True)
        
        try:
            # 收敛图
            plt.figure(figsize=(12, 8))
            plot_convergence(bayes_search.optimizer_results_[0])
            plt.title('Convergence Plot')
            plt.grid(True)
            plt.savefig(os.path.join(plots_path, f"{self.exp_name}_convergence.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 目标函数图
            plt.figure(figsize=(12, 8))
            plot_objective(bayes_search.optimizer_results_[0])
            plt.title('Objective Function Plot')
            plt.grid(True)
            plt.savefig(os.path.join(plots_path, f"{self.exp_name}_objective.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 评估图
            plt.figure(figsize=(12, 8))
            plot_evaluations(bayes_search.optimizer_results_[0])
            plt.title('Parameter Evaluations Plot')
            plt.grid(True)
            plt.savefig(os.path.join(plots_path, f"{self.exp_name}_evaluations.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 保存优化历史数据
            history_df = pd.DataFrame({
                'iteration': range(len(bayes_search.cv_results_['mean_test_score'])),
                'mean_test_score': bayes_search.cv_results_['mean_test_score'],
                'std_test_score': bayes_search.cv_results_['std_test_score']
            })
            history_df.to_csv(os.path.join(plots_path, f"{self.exp_name}_optimization_history.csv"), index=False)
            
            logger.info(f"Bayesian optimization plots and history saved at: {plots_path}")
            
        except Exception as e:
            logger.warning(f"Error while saving plots: {str(e)}")

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
        elif tune_type == 'bayesian':
            logger.info("Performing Bayesian Optimization...")
            best_estimator, best_params, best_score, cv_results = self.bayesian_search_cv(X_train, y_train)
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
        elif tune_type == 'bayesian':
            param_distributions = self.convert_to_optuna_distributions(self.config['bayesian_search_params'])
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
            # 过滤掉非数值类型的数据
            numeric_data = [x for x in params[col] if isinstance(x, (int, float))]
            plt.hist(numeric_data, bins=20)
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
            # 将 OrderedDict 转换为普通字典
            params_dict = dict(best_params)
            yaml.safe_dump(params_dict, f, default_flow_style=False)
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

    # 保存调优器
    # 如果可以调用save_tuner就进行Optuna重要性评估，并且使用HiPlot绘制并保存Optuna结果
    try:
        # 尝试调用 tuner.save_tuner 方法
        tuner.save_tuner(best_estimator, best_params, best_score, cv_results)
         # 使用Optuna重要性评估器调优
        optuna_study = tuner.optuna_study(cv_results, tune_type=tune_type)

        # 绘制并保存Optuna结果
        tuner.plot_and_save_optuna_result(optuna_study)

        logger.info("Hyperparameter tuning completed.")
        # 移除控制台处理程序
        logging.getLogger().handlers = []
    except Exception as e:
        # 如果发生任何错误，记录日志并继续执行
        logger.info(f"Error occurred: {e}, skipping Optuna importance evaluation.")
        # 移除控制台处理程序
        logging.getLogger().handlers = []

    return best_estimator, best_params, best_score, cv_results


if __name__ == "__main__":
    # 加载数据配置
    config_path = "../../configs/experiments/exp_preview.yaml"
    main(config_path, tune_type='random')
