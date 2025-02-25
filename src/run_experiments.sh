#!/bin/bash

# 激活 conda 环境
source ~/.bashrc
conda activate qikun

# 切换到项目的 src 目录
cd "/home/qikunlyu/文档/attendclass_predict_project/src"

# 定义目标列表和模型编号范围
targets=("target1" "target2")
models=$(seq 2 6)

# 标记是否已经开始运行
# start_running=false

# 先运行所有的训练步骤
# for target in "${targets[@]}"; do
#   for model_num in $models; do
#     model="model${model_num}"

    # 检查是否需要跳过
    # if [[ "${target}" == "target2" && "${model}" == "model7" ]]; then
    #   start_running=true
    # fi

    # 如果还未到达指定的起始点，则继续跳过
    # if [[ "${start_running}" == false ]]; then
    #   continue
    # fi

    # 检查是否需要跳过
#     if [[ ! ("${target}" == "target1" && "${model}" == "model7") && ! ("${target}" == "target2" && "${model}" == "model7") ]]; then
#       continue
#     fi

#     # 构建训练配置文件路径
#     train_config="/home/qikunlyu/文档/attendclass_predict_project/configs/experiments/${target}/${model}/train_config.yaml"

#     # 检查训练配置文件是否存在
#     if [[ -f "${train_config}" ]]; then
#       echo "正在运行训练：${target}/${model}"
#       python main.py --config "${train_config}" --step train
#     else
#       echo "训练配置文件不存在：${train_config}"
#     fi

#   done
# done

# 重置标记
start_running=false

# 再运行所有的调优步骤
for target in "${targets[@]}"; do
  for model_num in $models; do
    model="model${model_num}"

    # 检查是否需要跳过
    if [[ "${target}" == "target1" && "${model}" == "model4" ]]; then
      start_running=true
    fi

    # 如果还未到达指定的起始点，则继续跳过
    if [[ "${start_running}" == false ]]; then
      continue
    fi

    # 构建调优配置文件路径
    tune_config="/home/qikunlyu/文档/attendclass_predict_project/configs/experiments/${target}/${model}/tune004_config.yaml"

    # 检查调优配置文件是否存在
    if [[ -f "${tune_config}" ]]; then
      echo "正在运行调优：${target}/${model}"
      python main.py --config "${tune_config}" --step tune
    else
      echo "调优配置文件不存在：${tune_config}"
    fi

  done
done