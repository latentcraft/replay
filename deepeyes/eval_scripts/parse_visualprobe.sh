#!/bin/bash

# 检查是否提供了参数
if [ -z "$1" ]; then
    echo "Usage: bash parse_visualprobe.sh <model> <prefix>"
    echo "Example: bash parse_visualprobe.sh DeepEyes-7B visualprobe_easy"
    echo "Example: bash parse_visualprobe.sh DeepEyes-7B visualprobe_medium"
    echo "Example: bash parse_visualprobe.sh DeepEyes-7B visualprobe_hard"
    exit 1
fi

model=$1
prefix=$2
files="../mirror/DeepEyes/outputs/${model}/${prefix}_*_cuda.jsonl"

# 初始化统计变量
total_samples=0
total_acc_sum=0
declare -a position_sums
declare -a position_counts
max_positions=0

echo "Processing files matching: ${files}"
echo "========================================"
echo ""

# 处理所有匹配的文件
for file in $files; do
    if [ -f "$file" ]; then
        echo "Processing: $file"
        
        # 逐行读取jsonl文件
        while IFS= read -r line; do
            # 提取acc数组内容
            acc_array=$(echo "$line" | grep -o '"acc": \[[^]]*\]' | sed 's/"acc": \[\(.*\)\]/\1/')
            
            if [ ! -z "$acc_array" ]; then
                # 增加样本计数
                total_samples=$((total_samples + 1))
                
                # 按逗号分割acc数组
                IFS=',' read -ra values <<< "$acc_array"
                num_values=${#values[@]}
                
                # 更新最大位置数
                if [ $num_values -gt $max_positions ]; then
                    max_positions=$num_values
                fi
                
                # 累加每个位置的acc值
                for i in "${!values[@]}"; do
                    val=$(echo "${values[$i]}" | tr -d ' ')
                    if [ ! -z "$val" ]; then
                        position_sums[$i]=$(echo "${position_sums[$i]:-0} + $val" | bc)
                        position_counts[$i]=$((${position_counts[$i]:-0} + 1))
                        total_acc_sum=$(echo "$total_acc_sum + $val" | bc)
                    fi
                done
            fi
        done < "$file"
    fi
done

# 输出结果
if [ $total_samples -gt 0 ]; then
    echo ""
    echo "========================================"
    echo "Summary for ${model} - ${prefix}"
    echo "========================================"
    echo "Total samples processed: $total_samples"
    echo ""
    
    echo "----------------------------------------"
    echo "Accuracy by Position"
    echo "----------------------------------------"
    
    # 计算并显示每个位置的平均准确率
    for i in $(seq 0 $((max_positions - 1))); do
        if [ ${position_counts[$i]:-0} -gt 0 ]; then
            sum=${position_sums[$i]:-0}
            count=${position_counts[$i]}
            avg=$(echo "scale=4; $sum / $count * 100" | bc)
            position=$((i + 1))
            echo "  Position $position: ${avg}% (${count} samples)"
        fi
    done
    
    echo ""
    echo "========================================"
    echo "Overall Average Accuracy"
    echo "========================================"
    
    # 计算总体平均准确率
    if [ $total_samples -gt 0 ] && [ $max_positions -gt 0 ]; then
        overall_avg=$(echo "scale=4; $total_acc_sum / $total_samples * 100" | bc)
        echo "Average accuracy across all positions: ${overall_avg}%"
    fi
    echo ""
    
else
    echo "No data found for prefix: $prefix"
    echo "Make sure files exist at: ${files}"
fi