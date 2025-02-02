import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")

import sys
import os
# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from typing import Dict, List, Tuple, Any

from training.train import main as train_model
from training.evaluate import main as evaluate_model
from training.hyperparameter_tuning import main as tune_hyperparameters


def run_pipeline(args):
    """运行完整的数据处理和训练流程"""
    
    steps = args.step.split(',')
    for step in steps:
        if step == 'train':
            train_model(args.config)
            evaluate_model(args.config, 'train')
        elif step == 'tune':
            tune_hyperparameters(args.config, args.tune_type)
            evaluate_model(args.config, 'tune')

def main(config_path: str) -> None:
    """主函数"""
    parser = argparse.ArgumentParser(description='课堂出勤预测系统')
    parser.add_argument('--config', type=str, default=config_path,
                      help='配置文件路径')
    parser.add_argument('--step', type=str, choices=['train', 'tune', 'all'],
                      default='all', help='运行的步骤')
    parser.add_argument('--tune_type', type=str, choices=['random', 'grid', 'halving_random', 'halving_grid'], 
                      default='grid', help='使用的搜参方法')
    
    args = parser.parse_args()
    
    if args.step == 'all':
        args.step = 'train,tune'
    
    run_pipeline(args)

if __name__ == "__main__":
    config_path = "/home/qikunlyu/文档/attendclass_predict_project/configs/experiments/target1/model3/tune003_config.yaml"
    main(config_path)