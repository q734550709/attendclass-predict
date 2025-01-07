from .gbdt_model import GBDTModel
from .xgboost_model import XGBoostModel
from .lstm_model import LSTMModel
from .random_forest import RandomForestModel

class ModelFactory:
    @staticmethod
    def create_model(model_type, config):
        if model_type == "random_forest":
            return RandomForestModel(config)
        elif model_type == "gbdt":
            return GBDTModel(config)
        elif model_type == "xgboost":
            return XGBoostModel(config)
        elif model_type == "lstm":
            return LSTMModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}") 