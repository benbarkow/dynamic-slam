#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="$ROOT_DIR/datasets/tum"
LOG_PREFIX="$ROOT_DIR/logs/tum"
ATE_OUT_DIR="$ROOT_DIR/logs/ate_tum"
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

# Prepare output folder
mkdir -p "$ATE_OUT_DIR"

# Collect all dataset folders
datasets=()
for d in "$DATASET_DIR"/rgbd_dataset_freiburg3*; do
    [ -d "$d" ] && datasets+=("$(basename "$d")")
done

if [ ${#datasets[@]} -eq 0 ]; then
    echo "[!] No dataset folders found in $DATASET_DIR"
    exit 1
fi

# Run MASt3R-SLAM if not skipping
if [ "$print_only" = false ]; then
    for dataset in "${datasets[@]}"; do
        dataset_name="$DATASET_DIR/$dataset"

        if [ "$no_calib" = true ]; then
            python "$ROOT_DIR/main.py" --dataset "$dataset_name" --no-viz --save-as tum/no_calib/"$dataset" --config "$CONFIG_DIR/eval_no_calib.yaml"
        else
            python "$ROOT_DIR/main.py" --dataset "$dataset_name" --no-viz --save-as tum/calib/"$dataset" --config "$CONFIG_DIR/eval_calib.yaml"
        fi
    done
fi

# Evaluate with evo_ape and save results
for dataset in "${datasets[@]}"; do
    dataset_name="$DATASET_DIR/$dataset"
    echo "[*] Evaluating: $dataset"

    if [ "$no_calib" = true ]; then
        traj_file="$ROOT_DIR/logs/tum/no_calib/$dataset/$dataset.txt"
        out_file="$ATE_OUT_DIR/${dataset}_no_calib.txt"
    else
        traj_file="$ROOT_DIR/logs/tum/calib/$dataset/$dataset.txt"
        out_file="$ATE_OUT_DIR/${dataset}_calib.txt"
    fi

    if [ -f "$traj_file" ]; then
        evo_ape tum "$dataset_name/groundtruth.txt" "$traj_file" -as > "$out_file"
        echo "[✓] Saved ATE results to $out_file"
    else
        echo "[!] Missing trajectory file: $traj_file"
    fi
done