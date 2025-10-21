#!/usr/bin/env python3
"""
Debug script to check label values and identify issues.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


def analyze_labels_csv(labels_csv_path: str):
    """Analyze the labels CSV file."""
    print("=" * 60)
    print("Analyzing Labels CSV")
    print("=" * 60)
    
    df = pd.read_csv(labels_csv_path, encoding='utf-8')
    print(f"\nTotal rows: {len(df)}")
    print(f"\nColumns: {list(df.columns)}")
    
    if 'channel_combination' in df.columns:
        print(f"\n--- Channel Combinations ---")
        channel_combos = df['channel_combination'].value_counts()
        print(f"Total unique combinations: {len(channel_combos)}")
        print(f"\nTop 10 combinations:")
        for combo, count in channel_combos.head(10).items():
            print(f"  {combo}: {count}")
        
        # Check for problematic characters
        print(f"\n--- Checking for issues ---")
        for idx, combo in enumerate(df['channel_combination'].unique()):
            if not isinstance(combo, str):
                print(f"  ⚠ Non-string value at index {idx}: {combo} (type: {type(combo)})")
    
    return df


def analyze_encoded_labels(labels_csv_path: str, features_root: str):
    """Analyze encoded label values."""
    print("\n" + "=" * 60)
    print("Analyzing Encoded Label Values")
    print("=" * 60)
    
    from training.datasets import load_labels_csv, discover_patient_segments_from_csv, ConnectivityGraphDataset
    
    labels_df = load_labels_csv(labels_csv_path)
    patient_to_files = discover_patient_segments_from_csv(labels_csv_path, features_root)
    
    # Get all files
    all_files = []
    for files in patient_to_files.values():
        all_files.extend(files)
    
    print(f"\nTotal NPZ files: {len(all_files)}")
    
    # Create dataset (just to get labels)
    print("\nCreating dataset to extract labels...")
    try:
        dataset = ConnectivityGraphDataset(
            all_files[:100],  # Just check first 100 files
            labels_df,
            matrix_keys=['plv_alpha'],
            fusion_method='weighted'
        )
        
        labels = []
        for i in range(min(len(dataset), 100)):
            try:
                sample = dataset[i]
                label = int(sample['y'])
                labels.append(label)
            except Exception as e:
                print(f"  ⚠ Error loading sample {i}: {e}")
        
        if labels:
            labels_array = np.array(labels)
            print(f"\n--- Label Statistics ---")
            print(f"Min label: {labels_array.min()}")
            print(f"Max label: {labels_array.max()}")
            print(f"Mean label: {labels_array.mean():.2f}")
            print(f"Unique labels: {len(np.unique(labels_array))}")
            
            print(f"\n--- Label Distribution ---")
            counter = Counter(labels)
            for label, count in sorted(counter.items()):
                print(f"  Label {label}: {count} samples")
            
            # Check if any label is too large
            if labels_array.max() > 1000:
                print(f"\n⚠ WARNING: Maximum label value ({labels_array.max()}) is very large!")
                print(f"  This might cause issues with classification.")
                print(f"  Consider using a better label encoding method.")
        
        return labels
        
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_label_encoding():
    """Test different label encoding methods."""
    print("\n" + "=" * 60)
    print("Testing Label Encoding Methods")
    print("=" * 60)
    
    from training.label_encoder import encode_channel_combination
    
    test_combos = [
        "Fp1-Fp2-F3-F4",
        "C3-C4-P3-P4",
        "O1-O2",
        "F7-F8-T3-T4",
    ]
    
    encoding_types = ['frequency', 'count', 'binary', 'hash']
    
    for enc_type in encoding_types:
        print(f"\n--- {enc_type.upper()} encoding ---")
        labels = []
        for combo in test_combos:
            try:
                label = encode_channel_combination(combo, encoding_type=enc_type)
                labels.append(label)
                print(f"  {combo[:30]:30s} -> {label}")
            except Exception as e:
                print(f"  {combo[:30]:30s} -> ERROR: {e}")
        
        if labels:
            print(f"  Range: {min(labels)} - {max(labels)}")
            print(f"  Unique: {len(set(labels))}/{len(labels)}")


def recommend_num_classes(labels_csv_path: str):
    """Recommend the number of classes based on unique channel combinations."""
    print("\n" + "=" * 60)
    print("Recommended Number of Classes")
    print("=" * 60)
    
    df = pd.read_csv(labels_csv_path, encoding='utf-8')
    
    if 'channel_combination' in df.columns:
        unique_combos = df['channel_combination'].nunique()
        print(f"\nUnique channel combinations: {unique_combos}")
        print(f"\n✓ RECOMMENDATION:")
        print(f"  Set --num_classes {unique_combos} in your finetune script")
        print(f"\nOr use a simpler encoding:")
        print(f"  - If binary (seizure vs non-seizure): --num_classes 2")
        print(f"  - If channel region (frontal/temporal/etc): --num_classes 5-10")
        return unique_combos
    
    return None


def main():
    """Run all diagnostics."""
    print("\n" + "🔍 Label Diagnostics Tool".center(60))
    print()
    
    # Paths
    labels_csv = r'E:\output\connectivity_features\labels.csv'
    features_root = r'E:\output\connectivity_features'
    
    # Check if files exist
    if not Path(labels_csv).exists():
        print(f"✗ Labels CSV not found: {labels_csv}")
        return 1
    
    if not Path(features_root).exists():
        print(f"✗ Features root not found: {features_root}")
        return 1
    
    # Run diagnostics
    df = analyze_labels_csv(labels_csv)
    labels = analyze_encoded_labels(labels_csv, features_root)
    check_label_encoding()
    num_classes = recommend_num_classes(labels_csv)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary & Recommendations")
    print("=" * 60)
    
    if labels and len(labels) > 0:
        max_label = max(labels)
        unique_labels = len(set(labels))
        
        print(f"\n📊 Current label encoding:")
        print(f"  - Max label value: {max_label}")
        print(f"  - Unique labels: {unique_labels}")
        
        if max_label >= 1000:
            print(f"\n⚠ PROBLEM DETECTED:")
            print(f"  Hash-based encoding produces very large label values!")
            print(f"  This will cause CUDA errors with CrossEntropyLoss.")
            
            print(f"\n✅ SOLUTIONS:")
            print(f"  1. Use 'frequency' encoding (recommended):")
            print(f"     - Modify label_encoder.py")
            print(f"     - Or create a label mapping file")
            
            print(f"  2. Set correct num_classes:")
            print(f"     - In finetune script: --num_classes {unique_labels}")
            
            print(f"\n  3. Use simpler task:")
            print(f"     - Binary classification: seizure vs non-seizure")
            print(f"     - Region classification: frontal/temporal/parietal/occipital")
    
    print("\n" + "=" * 60)
    print("✓ Diagnostics complete!")
    print("\nNext steps:")
    print("  1. Review the output above")
    print("  2. Fix label encoding if needed")
    print("  3. Set correct --num_classes parameter")
    print("  4. Run training again")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
