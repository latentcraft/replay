#!/bin/bash
# Check if argument is provided
if [ -z "$1" ]; then
    echo "Usage: bash acc.sh <prefix>"
    echo "Example: bash eval_scripts/acc.sh DeepEyes-7B vstar"
    exit 1
fi

model=$1
prefix=$2
files="../mirror/DeepEyes/outputs/${model}/${prefix}_*_cuda.jsonl"

# Initialize counters
total=0
declare -a sums  # Array to store sums for each position
max_values=0     # Track the maximum number of values found

# Process all matching files
for file in $files; do
    if [ -f "$file" ]; then
        # Extract acc arrays and process them
        while IFS= read -r line; do
            # Extract the array content between brackets
            acc_array=$(echo "$line" | grep -o '"acc": \[[^]]*\]' | sed 's/"acc": \[\(.*\)\]/\1/')
            
            if [ ! -z "$acc_array" ]; then
                total=$((total + 1))
                
                # Split by comma and iterate through values
                IFS=',' read -ra values <<< "$acc_array"
                num_values=${#values[@]}
                
                # Update max_values if needed
                if [ $num_values -gt $max_values ]; then
                    max_values=$num_values
                fi
                
                # Add each value to its corresponding sum
                for i in "${!values[@]}"; do
                    val=$(echo "${values[$i]}" | tr -d ' ')
                    if [ ! -z "$val" ]; then
                        sums[$i]=$(echo "${sums[$i]:-0} + $val" | bc)
                    fi
                done
            fi
        done < "$file"
    fi
done

# Calculate and display accuracy
if [ $total -gt 0 ]; then
    echo "Total samples: $total"
    echo ""
    
    for i in $(seq 0 $((max_values - 1))); do
        sum=${sums[$i]:-0}
        avg=$(echo "scale=4; $sum / $total * 100" | bc)
        position=$((i + 1))
        echo "Position $position:"
        echo "  Sum: $sum"
        echo "  Average: ${avg}%"
    done
else
    echo "No data found for prefix: $prefix"
fi