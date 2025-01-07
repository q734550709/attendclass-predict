# 课堂出勤预测系统

## 项目概述
这个项目旨在通过机器学习方法预测学生的课堂出勤情况，帮助教育机构更好地管理和改善学生的出勤率。

## 项目结构
```
attendclass_predict_project/
├── data/                      # 数据集相关
│   ├── raw/                  # 原始数据
│   ├── processed/            # 预处理后的数据
│   └── features/             # 特征工程后的数据
├── src/                      # 源代码
│   ├── data/                # 数据处理相关代码
│   ├── features/            # 特征工程相关代码
│   ├── models/              # 模型定义和训练代码
│   └── visualization/       # 可视化相关代码
├── configs/                  # 配置文件
│   ├── base/               # 基础配置
│   ├── environments/       # 环境配置
│   └── experiments/        # 实验配置
├── notebooks/               # Jupyter notebooks用于探索性分析
├── docs/                    # 项目文档
├── experiments/             # 实验记录和结果
├── models/                  # 保存的模型文件
│   ├── trained/            # 训练好的模型
│   └── pretrained/         # 预训练模型
└── outputs/                 # 输出结果
    ├── predictions/        # 预测结果
    ├── figures/           # 数据可视化、模型性能图表等
    └── metrics/           # 模型评估指标记录（如准确率、召回率、F1分数等）
```

## 主要功能
1. 数据预处理和特征工程
2. 模型训练和评估
3. 预测结果可视化
4. 实验管理和记录

## 安装和使用
1. 克隆项目
```bash
git clone [项目地址]
```

2. 安装依赖
```bash
pip install -r requirements.txt
cd attendclass_predict_project
```

3. 运行训练
```bash
python src/main.py
```

## 文档
- 详细的数据说明请参考 `docs/data_dict.md`
- 模型架构说明请参考 `docs/model_desc.md`
- 实验记录请参考 `docs/experiments.md`

## 添加新模型
1. 在 `models/` 下创建新的模型实现文件
2. 继承 `base.py` 中的基类实现必要方法
3. 在 `factory.py` 中注册新模型
4. 在 `configs/experiments/` 下添加对应配置文件

## 配置说明
- 基础配置文件位于 `configs/base/`
- 环境特定配置文件位于 `configs/environments/`
- 实验配置文件位于 `configs/experiments/`

## 贡献指南
1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

## 许可证
MIT License 