import numpy as np
import yaml
import pandas as pd
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class HyperparameterTuning:
    def __init__(self, config_file: str):
        self.config = self._load_config(config_file)
        self.X_train, self.X_test, self.y_train, self.y_test = self._load_data()

    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载训练数据"""
        data_path = self.config['data']['paths']['processed']
        
        X_train = pd.read_csv(f"{data_path}/{self.config['data']['files']['X_train']}")
        X_test = pd.read_csv(f"{data_path}/{self.config['data']['files']['X_test']}")
        y_train = pd.read_csv(f"{data_path}/{self.config['data']['files']['y_train']}")
        y_test = pd.read_csv(f"{data_path}/{self.config['data']['files']['y_test']}")
        
        return X_train, X_test, y_train, y_test

    def grid_search_cv(self) -> Tuple[Dict, float]:
        """执行网格搜索"""
        param_grid = self.config['grid_search_params']
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        return grid_search.best_params_, grid_search.best_score_

    def random_search_cv(self) -> Tuple[Dict, float]:
        """执行随机搜索"""
        param_dist = self._get_param_distribution(self.config['random_search_params'])
        rf = RandomForestClassifier(random_state=42)
        random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, cv=3, random_state=42, n_jobs=-1, scoring='accuracy')
        random_search.fit(self.X_train, self.y_train)
        return random_search.best_params_, random_search.best_score_

    @staticmethod
    def _get_param_distribution(params: Dict) -> Dict:
        """获取参数分布"""
        param_dist = {}
        for param, values in params.items():
            if isinstance(values, dict) and 'start' in values and 'stop' in values and 'num' in values:
                param_dist[param] = [int(x) for x in np.linspace(start=values['start'], stop=values['stop'], num=values['num'])]
            else:
                param_dist[param] = values
        return param_dist
        
    def run(self, if_random_search=True) -> Tuple[Dict, float]:
        """运行超参数搜索"""
        if if_random_search:
            print("Performing Randomized Search...")
            best_params, best_score = self.random_search_cv()
            print("Best parameters (Randomized Search):", best_params)
            print("Best cross-validation score (Randomized Search):", best_score)
        else:
            print("Performing Grid Search...")
            best_params, best_score = self.grid_search_cv()
            print("Best parameters (Grid Search):", best_params)
            print("Best cross-validation score (Grid Search):", best_score)
        return best_params, best_score

def main(config_path: str) -> None:
    """主函数"""
    # 初始化超参数调优器
    tuner = HyperparameterTuning(config_path)
    
    # 运行超参数搜索
    best_params, best_score = tuner.run()
    
    # 打印结果
    print("Best parameters:", best_params)
    print("Best cross-validation score:", best_score)


if __name__ == "__main__":
    main()
