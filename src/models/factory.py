from .decision_tree import DecisionTreeModel  # 决策树模型
from .gbdt_model import GBDTModel  # GBDT模型
from .lstm_model import LSTMModel  # LSTM模型
from .logistic_reg import LogisticRegModel  # 逻辑回归模型
from .mlp_model import MLPModel  # 多层感知机模型
from .random_forest import RandomForestModel  # 随机森林模型
from .xgboost_model import XGBoostModel  # XGBoost模型


class ModelFactory:
    @staticmethod
    def create_model(model_type, config):
        if model_type == "decision_tree":  # 决策树模型
            return DecisionTreeModel(config)
        elif model_type == "gbdt":  # GBDT模型
            return GBDTModel(config)
        elif model_type == "lstm":  # LSTM模型
            return LSTMModel(config)
        elif model_type == "logistic_reg":  # 逻辑回归模型
            return LogisticRegModel(config)
        elif model_type == "mlp":  # 多层感知机模型
            return MLPModel(config)
        elif model_type == "random_forest":  # 随机森林模型
            return RandomForestModel(config)
        elif model_type == "xgboost":  # XGBoost模型
            return XGBoostModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")  # 未知的模型类型