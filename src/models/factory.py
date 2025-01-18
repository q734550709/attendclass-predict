from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

class ModelFactory:
    @staticmethod
    def create_model(model_type, config):
        if model_type == "decision_tree":  # 决策树模型
            return DecisionTreeClassifier(
                criterion=config.get('criterion', 'gini'),
                splitter=config.get('splitter', 'best'),
                max_depth=config.get('max_depth'),
                random_state=config.get('random_state', 42),
                **{k: v for k, v in config.items() if k not in ['criterion', 'splitter', 'max_depth', 'random_state']}
            )
        elif model_type == "gbdt":  # GBDT模型
            return GradientBoostingClassifier(
                loss=config.get('loss', 'deviance'),
                learning_rate=config.get('learning_rate', 0.1),
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', 3),
                random_state=config.get('random_state', 42)
            )
        elif model_type == "logistic_reg":  # 逻辑回归模型
            return LogisticRegression(
                penalty=config.get('penalty', 'l2'),
                C=config.get('C', 1.0),
                solver=config.get('solver', 'lbfgs'),
                max_iter=config.get('max_iter', 100),
                random_state=config.get('random_state', 42),
                **{k: v for k, v in config.items() if k not in ['penalty', 'C', 'solver', 'max_iter', 'random_state']}
            )
        elif model_type == "mlp":  # 多层感知机模型
            return MLPClassifier(
                hidden_layer_sizes=config.get('hidden_layer_sizes', (100,)),
                activation=config.get('activation', 'relu'),
                solver=config.get('solver', 'adam'),
                alpha=config.get('alpha', 0.0001),
                batch_size=config.get('batch_size', 'auto'),
                learning_rate=config.get('learning_rate', 'constant'),
                max_iter=config.get('max_iter', 200),
                random_state=config.get('random_state', 42)
            )
        elif model_type == "random_forest":  # 随机森林模型
            return RandomForestClassifier(
                n_estimators=config.get('n_estimators', 100),
                criterion=config.get('criterion', 'gini'),
                max_depth=config.get('max_depth'),
                random_state=config.get('random_state', 42)
            )
        elif model_type == "xgboost":  # XGBoost模型
            return xgb.XGBClassifier(
                max_depth=config.get('max_depth', 3),
                learning_rate=config.get('learning_rate', 0.1),
                n_estimators=config.get('n_estimators', 100),
                random_state=config.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")  # 未知的模型类型