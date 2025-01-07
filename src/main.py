import argparse
import logging
from pathlib import Path
import yaml
import sys

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.data.make_dataset import main as make_dataset
from src.models.training.train import main as train_model
from src.models.inference.predict import main as predict

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
    
    if args.step == 'all' or args.step == 'data':
        logger.info("Starting data processing step")
        make_dataset()
    
    if args.step == 'all' or args.step == 'train':
        logger.info("Starting model training step")
        train_model()
    
    if args.step == 'all' or args.step == 'predict':
        logger.info("Starting prediction step")
        predict()

def main():
    parser = argparse.ArgumentParser(description='课堂出勤预测系统')
    parser.add_argument('--config', type=str, default='configs/base/model.yaml',
                      help='配置文件路径')
    parser.add_argument('--step', type=str, choices=['all', 'data', 'train', 'predict'],
                      default='all', help='运行的步骤')
    
    args = parser.parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main() 