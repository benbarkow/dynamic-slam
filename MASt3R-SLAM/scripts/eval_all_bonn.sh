#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="$ROOT_DIR/datasets/bonn"
LOG_PREFIX="$ROOT_DIR/logs/bonn"
ATE_OUT_DIR="$ROOT_DIR/logs/ate_bonn"
CONFIG_DIR="$ROOT_DIR/config"

no_calib=false
print_only=false

# Parse options
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --no-calib)
            no_calib=true
            ;;
        --print)
            print_only=true
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

mkdir -p "$ATE_OUT_DIR"

datasets=()
for d in "$DATASET_DIR"/rgbd_bonn_*; do
    [ -d "$d" ] && datasets+=("$(basename "$d")")
done

if [ ${#datasets[@]} -eq 0 ]; then
    echo "[!] No dataset folders found in $DATASET_DIR"
    exit 1
fi

selected_indices=(0 1)  # <--- Change these to the indices you want to run with main.py

# Run MASt3R-SLAM if not skipping
if [ "$print_only" = false ]; then
    for idx in "${!datasets[@]}"; do
        if [[ " ${selected_indices[@]} " =~ " ${idx} " ]]; then
            dataset="${datasets[$idx]}"
            dataset_path="$DATASET_DIR/$dataset"

            echo "[*] Running main.py for: $dataset (index $idx)"

            if [ "$no_calib" = true ]; then
                python "$ROOT_DIR/main.py" --dataset "$dataset_path" --no-viz --save-as bonn/no_calib/"$dataset" --config "$CONFIG_DIR/eval_no_calib.yaml"
            else
                python "$ROOT_DIR/main.py" --dataset "$dataset_path" --no-viz --save-as bonn/calib/"$dataset" --config "$CONFIG_DIR/eval_calib_bonn.yaml"
            fi
        else
            echo "[!] Skipping dataset at index $idx: ${datasets[$idx]}"
        fi
    done
fi

# Evaluate with evo_ape using aligned timestamps
for dataset in "${datasets[@]}"; do
    echo "[*] Evaluating: $dataset"
    dataset_dir="$DATASET_DIR/$dataset"
    gt_file="$dataset_dir/groundtruth.txt"

    if [ "$no_calib" = true ]; then
        traj_file="$ROOT_DIR/logs/bonn/no_calib/$dataset/$dataset.txt"
        out_file="$ATE_OUT_DIR/${dataset}_no_calib.txt"
        shifted_file="$ROOT_DIR/logs/bonn/no_calib/$dataset/${dataset}_shifted.txt"
    else
        traj_file="$ROOT_DIR/logs/bonn/calib/$dataset/$dataset.txt"
        out_file="$ATE_OUT_DIR/${dataset}_calib.txt"
        shifted_file="$ROOT_DIR/logs/bonn/calib/$dataset/${dataset}_shifted.txt"
    fi

    if [ -f "$traj_file" ] && [ -f "$gt_file" ]; then
        # Extract the first ground truth timestamp
        first_gt_timestamp=$(grep -v '^#' "$gt_file" | head -n 1 | awk '{print $1}')

        if [[ -z "$first_gt_timestamp" ]]; then
            echo "[!] Could not extract first groundtruth timestamp for $dataset"
            continue
        fi

        # Shift SLAM trajectory timestamps
        # Extract the first ground truth timestamp
        first_gt_timestamp=$(grep -v '^#' "$gt_file" | head -n 1 | awk '{print $1}')
        traj_start=$(head -n 1 "$traj_file" | awk '{print $1}')
        offset=$(awk "BEGIN {printf \"%.6f\", $first_gt_timestamp - $traj_start}")

        # Shift SLAM trajectory timestamps
        shifted_file="${traj_file%.txt}_shifted.txt"
        python "$ROOT_DIR/scripts/shift_traj_timestamps.py" bonn "$dataset" "$offset"

        # Evaluate with shifted file
        evo_ape tum "$gt_file" "$shifted_file" --t_max_diff 0.05 --align -as > "$out_file"



        echo "[✓] Saved ATE results to $out_file"
    else
        echo "[!] Missing file(s):"
        [ ! -f "$traj_file" ] && echo "    ✗ Trajectory: $traj_file"
        [ ! -f "$gt_file" ] && echo "    ✗ Groundtruth: $gt_file"
    fi
done