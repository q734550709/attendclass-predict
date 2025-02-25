# 课堂出勤预测系统

## 项目概述
这个项目旨在通过机器学习方法预测学生的课堂出勤情况，帮助教育机构更好地管理和改善学生的出勤率。系统采用多种机器学习算法，包括传统机器学习方法和深度学习方法，通过分析历史数据来预测学生的出勤行为。

## 技术栈
- **Python**: 3.12.2+
- **核心数据科学库**: 
  - NumPy 1.26.4
  - Pandas 2.2.3
  - Scikit-learn 1.5.2
- **深度学习框架**: 
  - PyTorch 2.5.1
  - TorchVision 0.20.1
- **集成学习框架**:
  - XGBoost 2.0.3
  - LightGBM 4.3.0
- **数据可视化**:
  - Matplotlib 3.9.2
  - Seaborn 0.13.2
  - Plotly 5.24.1
- **模型解释性**:
  - SHAP 0.44.1

## 项目结构
```
attendclass_predict_project/
├── data/                      # 数据集相关
│   ├── raw/                  # 原始数据
│   ├── processed/            # 预处理后的数据
│   └── features/             # 特征工程后的数据
├── src/                      # 源代码
│   ├── data/                # 数据处理模块
│   ├── models/              # 模型定义和实现
│   ├── training/            # 训练相关代码
│   ├── inference/           # 推理相关代码
│   ├── main.py             # 主程序入口
│   └── run_experiments.sh   # 实验运行脚本
├── configs/                  # 配置文件
├── notebooks/               # Jupyter notebooks
├── docs/                    # 项目文档
├── experiments/             # 实验记录和结果
├── models/                  # 模型文件
└── outputs/                 # 输出结果
```

## 主要功能
1. **数据预处理和特征工程**
   - 数据清洗和标准化
   - 特征提取和选择
   - 数据增强和平衡

2. **模型训练和评估**
   - 支持多种机器学习算法
   - 交叉验证和模型选择
   - 超参数优化
   - 模型性能评估

3. **预测和可视化**
   - 批量预测功能
   - 预测结果可视化
   - 模型解释性分析

4. **实验管理**
   - 实验配置管理
   - 训练过程监控
   - 结果记录和对比

## 安装和使用

### 环境要求
- Python 3.12.2 或更高版本
- CUDA支持（用于GPU训练，可选）

### 安装步骤
1. 克隆项目
```bash
git clone [项目地址]
cd attendclass_predict_project
```

2. 创建并激活虚拟环境（推荐）
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

### 使用说明

1. **数据准备**
```bash
python src/data/prepare_data.py
```

2. **训练模型**
```bash
python src/main.py --mode train --config configs/experiments/exp001.yaml
```

3. **运行预测**
```bash
python src/main.py --mode predict --model-path models/trained/model.pkl
```

4. **运行实验**
```bash
bash src/run_experiments.sh
```

## 配置说明
- 所有配置文件位于 `configs/` 目录下
- 使用YAML格式定义配置
- 支持环境特定配置覆盖

## 实验记录
实验结果和模型性能指标保存在 `experiments/` 目录下，包括：
- 训练日志
- 评估指标
- 性能图表
- 模型参数

## 贡献指南
1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 发起 Pull Request

## 问题反馈
如有问题或建议，请在项目的Issues页面提出。

## 许可证
本项目采用 MIT License 开源许可证。 