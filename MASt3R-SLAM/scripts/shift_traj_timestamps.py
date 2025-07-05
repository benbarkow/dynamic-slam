#!/usr/bin/env python3

import os
import sys
import glob

def shift_trajectory_timestamps(input_file, output_file, offset):
    """Shift timestamps in trajectory file by given offset"""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if line.strip().startswith('#') or len(line.strip()) == 0:
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                parts[0] = f"{float(parts[0]) + offset:.6f}"
                f_out.write(' '.join(parts) + '\n')

def process_tum_datasets(project_dir, offset, specific_dataset=None):
    """Process TUM datasets"""
    tum_calib_dir = os.path.join(project_dir, "logs", "tum", "calib")
    
    if not os.path.exists(tum_calib_dir):
        print(f"Warning: TUM calibration directory not found at {tum_calib_dir}")
        return 0
    
    print(f"Processing TUM datasets with offset: {offset}")
    processed_count = 0
    
    search_pattern = specific_dataset if specific_dataset else "rgbd_dataset_freiburg*"
    
    for dataset_dir in glob.glob(os.path.join(tum_calib_dir, search_pattern)):
        if os.path.isdir(dataset_dir):
            dataset_name = os.path.basename(dataset_dir)
            trajectory_file = os.path.join(dataset_dir, f"{dataset_name}.txt")
            shifted_file = os.path.join(dataset_dir, f"{dataset_name}_shifted.txt")
            
            if os.path.exists(trajectory_file):
                try:
                    shift_trajectory_timestamps(trajectory_file, shifted_file, offset)
                    print(f"  ✓ Processed: {dataset_name}")
                    processed_count += 1
                except Exception as e:
                    print(f"  ✗ Error processing {dataset_name}: {e}")
            else:
                print(f"  ⚠ Trajectory file not found: {trajectory_file}")
    
    return processed_count


def process_bonn_datasets(project_dir, offset, specific_dataset=None):
    """Process Bonn datasets"""
    bonn_calib_dir = os.path.join(project_dir, "logs", "bonn", "no_calib")
    
    if not os.path.exists(bonn_calib_dir):
        print(f"Warning: Bonn calibration directory not found at {bonn_calib_dir}")
        return 0
    
    print(f"Processing Bonn datasets with offset: {offset}")
    processed_count = 0
    
    search_pattern = specific_dataset if specific_dataset else "rgbd_bonn_*"
    
    for dataset_dir in glob.glob(os.path.join(bonn_calib_dir, search_pattern)):
        if os.path.isdir(dataset_dir):
            dataset_name = os.path.basename(dataset_dir)
            trajectory_file = os.path.join(dataset_dir, f"{dataset_name}.txt")
            shifted_file = os.path.join(dataset_dir, f"{dataset_name}_shifted.txt")
            
            if os.path.exists(trajectory_file):
                try:
                    shift_trajectory_timestamps(trajectory_file, shifted_file, offset)
                    print(f"  ✓ Processed: {dataset_name}")
                    processed_count += 1
                except Exception as e:
                    print(f"  ✗ Error processing {dataset_name}: {e}")
            else:
                print(f"  ⚠ Trajectory file not found: {trajectory_file}")
    
    return processed_count


def show_usage():
    print("Usage: python shift_all_trajectories.py [dataset_type] [dataset_name] [offset]")
    print("")
    print("Parameters:")
    print("  dataset_type: tum, bonn, or all (default: all)")
    print("  dataset_name: optional - specific dataset folder name (e.g., rgbd_bonn_01)")
    print("  offset: time offset in seconds (default: tum=0.0, bonn=0.035)")
    print("")
    print("Examples:")
    print("  python shift_all_trajectories.py tum 0.02")
    print("  python shift_all_trajectories.py bonn rgbd_bonn_01 0.04")
    print("  python shift_all_trajectories.py bonn rgbd_bonn_02")
    print("  python shift_all_trajectories.py all")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    dataset_type = "all"
    dataset_name = None
    tum_offset = 0.0
    bonn_offset = 0.035

    args = sys.argv[1:]

    if not args or args[0] in ["help", "-h", "--help"]:
        show_usage()
        return

    dataset_type = args[0].lower()

    if len(args) >= 2:
        try:
            # If it's a float, it's offset; else, it's dataset name
            float(args[1])  # just to check
            offset = float(args[1])
            if dataset_type == "tum":
                tum_offset = offset
            elif dataset_type == "bonn":
                bonn_offset = offset
            else:
                tum_offset = bonn_offset = offset
        except ValueError:
            dataset_name = args[1]
    
    if len(args) == 3:
        try:
            offset = float(args[2])
            if dataset_type == "tum":
                tum_offset = offset
            elif dataset_type == "bonn":
                bonn_offset = offset
            else:
                tum_offset = bonn_offset = offset
        except ValueError:
            print("Error: Invalid offset value. Using defaults.")
    
    if dataset_type not in ["tum", "bonn", "all"]:
        print(f"Error: Invalid dataset type '{dataset_type}'")
        show_usage()
        return

    print("Shifting trajectory timestamps...")
    print("="*50)

    total_processed = 0

    if dataset_type in ["tum", "all"]:
        total_processed += process_tum_datasets(project_dir, tum_offset, dataset_name if dataset_type == "tum" else None)
        print()

    if dataset_type in ["bonn", "all"]:
        total_processed += process_bonn_datasets(project_dir, bonn_offset, dataset_name if dataset_type == "bonn" else None)
        print()

    print("="*50)
    print(f"Summary: {total_processed} datasets processed successfully")

    if total_processed > 0:
        print("\nShifted trajectory files created with '_shifted.txt' suffix")
        print("You can now run evaluation scripts on the shifted trajectories")

if __name__ == "__main__":
    main()