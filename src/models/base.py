import joblib
import pandas as pd

class BaseModel:
    def train(self, X, y):
        raise NotImplementedError("train method not implemented")
    
    def predict(self, X):
        raise NotImplementedError("predict method not implemented")
    
    def predict_proba(self, X):
        raise NotImplementedError("predict_proba method not implemented")
    
    def evaluate(self, X, y):
        raise NotImplementedError("evaluate method not implemented")
    
    def save_model(self, filepath):
        """保存模型到文件
        
        Args:
            filepath (str): 模型保存路径
        """
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """从文件加载模型
        
        Args:
            filepath (str): 模型文件路径
        """
        self.model = joblib.load(filepath)
    
    @staticmethod
    def load_data(filepath):
        """从文件加载数据
        
        Args:
            filepath (str): 数据文件路径
            
        Returns:
            DataFrame: 加载的数据
        """
        return pd.read_csv(filepath)
    
    @staticmethod
    def save_data(data, filepath):
        """保存数据到文件
        
        Args:
            data (DataFrame): 要保存的数据
            filepath (str): 数据保存路径
        """
        data.to_csv(filepath, index=False)