#!/bin/bash

# tmux 参数设置
session_name="cogvideo"

# 基本路径设置
BASE_OUT_PATH="evaluate/datasets/video/cogvideo_sparge"
MODEL_PATH="evaluate/models_dict/CogVideoX-2b_0.06_0.07.pt"
EXECUTE_FILE="evaluate/cogvideo_example.py"

# GPU 初始配置
gpu=0
num_gpus=4

# threshold 范围
threshold_values=$(seq 1.01 0.01 1.1)

# 创建 tmux 会话
first_gpu=0

# 分配任务到 GPU
for num_layer in $(seq 1 30); do
    for threshold in $threshold_values; do
        out_path="${BASE_OUT_PATH}_threshold_${threshold}_layer_${num_layer}"

        # 创建窗口（如果需要）
        if ! tmux list-windows -t "$session_name" | grep -q "^$gpu:"; then
            tmux new-window -t "$session_name" -n "$gpu"
            tmux send-keys -t "$session_name:$gpu" "export CUDA_VISIBLE_DEVICES=$gpu" C-m
        fi

        tmux send-keys -t "$session_name:$gpu" \
        "python $EXECUTE_FILE \
            --use_spas_sage_attn \
            --model_out_path '$MODEL_PATH' \
            --use_kv_sparse \
            --out_path '$out_path' \
            --num_evaluate_layer $num_layer \
            --kv_sparse_threshold $threshold" C-m

        gpu=$(( (gpu + 1) % num_gpus ))
    done
done

# 最终提示
echo "Tmux session '$session_name' created with $num_gpus GPU windows."
echo "To attach to the session, use: tmux attach -t $session_name"
