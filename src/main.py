import argparse
import logging
from pathlib import Path
import yaml
import sys

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent))

from training.train import main as train_model
from training.evaluate import main as evaluate_model
from training.hyperparameter_tuning import main as tune_hyperparameters

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_environment(config: dict):
    """设置运行环境"""
    # 创建必要的目录
    for path in config['data']['paths'].values():
        Path(path).mkdir(parents=True, exist_ok=True)
    
    # 创建实验目录
    Path('experiments').mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)

def run_pipeline(args):
    """运行完整的数据处理和训练流程"""
    # 加载配置
    config = load_config(args.config)
    
    # 设置环境
    setup_environment(config)
    
    steps = args.step.split(',')
    for step in steps:
        if step == 'train':
            logger.info("Starting model training step")
            train_model(args.config)
        elif step == 'evaluate':
            logger.info("Starting model evaluation step")
            evaluate_model(args.config)
        elif step == 'tune':
            logger.info("Starting hyperparameter tuning step")
            tune_hyperparameters(args.config)

def main():
    parser = argparse.ArgumentParser(description='课堂出勤预测系统')
    parser.add_argument('--config', type=str, default='configs/base/model.yaml',
                      help='配置文件路径')
    parser.add_argument('--step', type=str, choices=['train', 'evaluate', 'tune', 'all'],
                      required=True, help='运行的步骤')
    
    args = parser.parse_args()
    
    if args.step == 'all':
        args.step = 'train,tune,evaluate'
    
    run_pipeline(args)

if __name__ == "__main__":
    main()