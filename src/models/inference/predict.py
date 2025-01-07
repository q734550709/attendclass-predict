import pandas as pd
import numpy as np
import joblib
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Union
from datetime import datetime

from src.data.preprocess import DataPreprocessor
from src.features.build_features import FeatureBuilder

logger = logging.getLogger(__name__)

class AttendancePredictor:
    def __init__(self, model_path: str, config_path: str):
        """初始化预测器"""
        self.config = self._load_config(config_path)
        self.model = self._load_model(model_path)
        self.preprocessor = self._load_preprocessor(model_path)
        self.feature_builder = FeatureBuilder(self.config)
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_model(self, model_path: str):
        """加载模型"""
        model_file = Path(model_path) / 'model.joblib'
        return joblib.load(model_file)
    
    def _load_preprocessor(self, model_path: str) -> DataPreprocessor:
        """加载预处理器"""
        preprocessor_file = Path(model_path) / 'preprocessor.joblib'
        return joblib.load(preprocessor_file)
    
    def predict(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """进行预测"""
        # 将输入数据转换为DataFrame
        if isinstance(data, (dict, list)):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
            
        # 特征工程
        df = self.feature_builder.build_features(df)
        
        # 数据预处理
        df = self.preprocessor.transform(df)
        
        # 模型预测
        predictions = self.model.predict_proba(df)[:, 1]
        
        # 根据阈值转换为二分类结果
        threshold = self.config['model']['output']['threshold']
        binary_predictions = (predictions >= threshold).astype(int)
        
        return binary_predictions
    
    def predict_proba(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """预测概率"""
        # 将输入数据转换为DataFrame
        if isinstance(data, (dict, list)):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
            
        # 特征工程
        df = self.feature_builder.build_features(df)
        
        # 数据预处理
        df = self.preprocessor.transform(df)
        
        # 返回预测概率
        return self.model.predict_proba(df)[:, 1]
    
    def predict_batch(self, data_path: str, output_path: str):
        """批量预测"""
        # 读取数据
        df = pd.read_csv(data_path)
        
        # 进行预测
        predictions = self.predict(df)
        probabilities = self.predict_proba(df)
        
        # 保存结果
        results_df = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities
        })
        
        # 如果原始数据有ID列，保留它
        if 'id' in df.columns:
            results_df['id'] = df['id']
            
        # 保存预测结果
        output_file = Path(output_path) / f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
        
        return results_df

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='课堂出勤预测')
    parser.add_argument('--model-path', type=str, required=True,
                      help='模型路径')
    parser.add_argument('--config-path', type=str, required=True,
                      help='配置文件路径')
    parser.add_argument('--input-path', type=str, required=True,
                      help='输入数据路径')
    parser.add_argument('--output-path', type=str, required=True,
                      help='输出结果路径')
    
    args = parser.parse_args()
    
    # 初始化预测器
    predictor = AttendancePredictor(args.model_path, args.config_path)
    
    # 进行批量预测
    predictor.predict_batch(args.input_path, args.output_path)

if __name__ == "__main__":
    main() 