#!/bin/bash
# Check if argument is provided
if [ -z "$1" ]; then
    echo "Usage: bash acc.sh <model> <prefix>"
    echo "Example: bash eval_scripts/acc.sh DeepEyes-7B vstar"
    exit 1
fi

model=$1
prefix=$2
files="../mirror/DeepEyes/outputs/${model}/${prefix}_*_cuda.jsonl"

# Declare associative arrays for subset statistics
declare -A subset_totals      # Total samples per subset
declare -A subset_sums        # Sums for each subset and position (key: subset_position)
declare -A subset_max_values  # Max values per subset

# Process all matching files
for file in $files; do
    if [ -f "$file" ]; then
        # Extract acc arrays and process them
        while IFS= read -r line; do
            # Extract subset from image_path
            subset=$(echo "$line" | grep -o '"image_path": "[^"]*"' | sed 's/"image_path": ".*\/\([^\/]*\)\/[^"]*"/\1/')
            
            # Extract the acc array content between brackets
            acc_array=$(echo "$line" | grep -o '"acc": \[[^]]*\]' | sed 's/"acc": \[\(.*\)\]/\1/')
           
            if [ ! -z "$acc_array" ] && [ ! -z "$subset" ]; then
                # Increment total for this subset
                subset_totals[$subset]=$((${subset_totals[$subset]:-0} + 1))
               
                # Split by comma and iterate through values
                IFS=',' read -ra values <<< "$acc_array"
                num_values=${#values[@]}
               
                # Update max_values for this subset if needed
                if [ $num_values -gt ${subset_max_values[$subset]:-0} ]; then
                    subset_max_values[$subset]=$num_values
                fi
               
                # Add each value to its corresponding sum for this subset
                for i in "${!values[@]}"; do
                    val=$(echo "${values[$i]}" | tr -d ' ')
                    if [ ! -z "$val" ]; then
                        key="${subset}_${i}"
                        subset_sums[$key]=$(echo "${subset_sums[$key]:-0} + $val" | bc)
                    fi
                done
            fi
        done < "$file"
    fi
done

# Calculate and display accuracy for each subset
if [ ${#subset_totals[@]} -gt 0 ]; then
    echo "========================================"
    echo "Accuracy by Subset"
    echo "========================================"
    echo ""
    
    # Overall statistics
    overall_total=0
    declare -a overall_sums
    overall_max=0
    
    # Sort subsets alphabetically for consistent output
    for subset in $(echo "${!subset_totals[@]}" | tr ' ' '\n' | sort); do
        total=${subset_totals[$subset]}
        max_vals=${subset_max_values[$subset]}
        
        echo "----------------------------------------"
        echo "Subset: $subset"
        echo "Total samples: $total"
        echo ""
        
        for i in $(seq 0 $((max_vals - 1))); do
            key="${subset}_${i}"
            sum=${subset_sums[$key]:-0}
            avg=$(echo "scale=4; $sum / $total * 100" | bc)
            position=$((i + 1))
            echo "  Position $position: ${avg}%"
            
            # Accumulate for overall statistics
            overall_sums[$i]=$(echo "${overall_sums[$i]:-0} + $sum" | bc)
        done
        
        overall_total=$((overall_total + total))
        if [ $max_vals -gt $overall_max ]; then
            overall_max=$max_vals
        fi
        
        echo ""
    done
    
    # Display overall statistics
    echo "========================================"
    echo "Overall Statistics"
    echo "========================================"
    echo "Total samples: $overall_total"
    echo ""
    
    for i in $(seq 0 $((overall_max - 1))); do
        sum=${overall_sums[$i]:-0}
        avg=$(echo "scale=4; $sum / $overall_total * 100" | bc)
        position=$((i + 1))
        echo "  Position $position: ${avg}%"
    done
    
else
    echo "No data found for prefix: $prefix"
fi