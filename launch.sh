#!/bin/bash

# tmux 参数设置
session_name="cogvideo_sparse"
experiment_name="cogvideo_original"
# 基本路径设置
BASE_OUT_PATH="evaluate/results"
MODEL_PATH="evaluate/models_dict/CogVideoX-2b_0.06_0.07.pt"
EXECUTE_FILE="evaluate/cogvideo_example.py"

gpu_assignments=("0")

num_assignments=${#gpu_assignments[@]} # 获取列表中的元素数量
assignment_idx=0 # 当前使用的 GPU 分配方案的索引

# threshold 范围
threshold_values=$(seq 1.011 0.001 1.011)

echo "将在 ${num_assignments} 个 tmux 窗口/GPU配置 (${gpu_assignments[*]}) 上分配任务..."

# 分配任务
for num_layer in $(seq 1 1); do
    for threshold in $threshold_values; do
        # 获取当前的 GPU 分配方案 (例如 "0" 或 "1,2")
        current_gpu_setting=${gpu_assignments[$assignment_idx]}
        # 获取要发送命令的 tmux 窗口/窗格 ID (使用索引)
        target_pane_id=$assignment_idx

        out_path="${BASE_OUT_PATH}/${experiment_name}/_threshold_${threshold}_layer_${num_layer}_gpus_${current_gpu_setting//,/}" # 文件路径中使用 GPU 信息 (替换逗号)
        mkdir -p $out_path

        # 构建 python 命令，使用 CUDA_VISIBLE_DEVICES 指定 GPU
        command="CUDA_VISIBLE_DEVICES=$current_gpu_setting python $EXECUTE_FILE \
            --use_spas_sage_attn \
            --model_out_path '$MODEL_PATH' \
            --out_path '$out_path'"

        echo "发送任务到 tmux $session_name:$target_pane_id (使用 GPUs: $current_gpu_setting)"
        # 发送命令到对应的 tmux 窗口/窗格
        tmux send-keys -t "$session_name:$target_pane_id" "$command" C-m

        # 更新索引，轮换到下一个 GPU 分配方案
        assignment_idx=$(( (assignment_idx + 1) % num_assignments ))
    done
done

echo "所有任务已分配完毕。"