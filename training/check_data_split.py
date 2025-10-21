#!/usr/bin/env python3
"""
检查数据分割情况的诊断脚本
"""

import sys
from pathlib import Path

from datasets_multilabel import (
    load_labels_csv,
    discover_patient_segments_from_csv,
    make_patient_splits
)


def check_data_split(labels_csv_path, features_root):
    """检查数据分割情况"""
    print("=" * 60)
    print("Data Split Diagnostic Tool")
    print("=" * 60)
    
    # 1. 检查文件是否存在
    print("\n1. Checking file paths...")
    if not Path(labels_csv_path).exists():
        print(f"  ✗ Labels CSV not found: {labels_csv_path}")
        return False
    print(f"  ✓ Labels CSV exists: {labels_csv_path}")
    
    if not Path(features_root).exists():
        print(f"  ✗ Features root not found: {features_root}")
        return False
    print(f"  ✓ Features root exists: {features_root}")
    
    # 2. 加载labels
    print("\n2. Loading labels...")
    labels_df = load_labels_csv(labels_csv_path)
    print(f"  ✓ Loaded {len(labels_df)} rows from labels.csv")
    print(f"  Columns: {list(labels_df.columns)}")
    
    # 3. 发现患者文件
    print("\n3. Discovering patient files...")
    patient_to_files = discover_patient_segments_from_csv(labels_csv_path, features_root)
    
    print(f"\n  Found {len(patient_to_files)} patients:")
    for patient, files in sorted(patient_to_files.items()):
        print(f"    {patient}: {len(files)} files")
    
    total_files = sum(len(files) for files in patient_to_files.values())
    print(f"\n  Total NPZ files: {total_files}")
    
    # 4. 检查数据分割
    print("\n4. Testing data splits...")
    
    # 测试不同的分割比例
    split_configs = [
        (0.2, 0.1, "20% test, 10% val (default)"),
        (0.2, 0.0, "20% test, 0% val (no validation)"),
        (0.3, 0.15, "30% test, 15% val (larger validation)"),
    ]
    
    for test_ratio, val_ratio, desc in split_configs:
        print(f"\n  Testing: {desc}")
        try:
            splits = make_patient_splits(
                patient_to_files,
                test_ratio=test_ratio,
                val_ratio=val_ratio,
                seed=42
            )
            
            print(f"    Train files: {len(splits['train'])}")
            print(f"    Val files: {len(splits['val'])}")
            print(f"    Test files: {len(splits['test'])}")
            
            if len(splits['val']) == 0 and val_ratio > 0:
                print(f"    ⚠ Warning: Val set is empty despite val_ratio={val_ratio}")
            elif len(splits['val']) > 0:
                print(f"    ✓ All sets have data")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    # 5. 建议
    print("\n" + "=" * 60)
    print("Recommendations:")
    print("=" * 60)
    
    n_patients = len(patient_to_files)
    
    if n_patients < 3:
        print("\n✗ CRITICAL: Too few patients!")
        print(f"  You have {n_patients} patients, need at least 3")
        print("  Cannot create train/val/test split")
        return False
    elif n_patients < 10:
        print(f"\n⚠ WARNING: Only {n_patients} patients")
        print("  Recommendations:")
        print("    - Use train/test split only (no validation)")
        print("    - Or use cross-validation")
        print("    - Suggested: --val_ratio 0.0")
    elif n_patients < 30:
        print(f"\n✓ {n_patients} patients - acceptable")
        print("  Recommendations:")
        print("    - Use smaller val_ratio")
        print("    - Suggested: --val_ratio 0.05 or 0.1")
    else:
        print(f"\n✓ {n_patients} patients - good!")
        print("  Can use standard split ratios")
        print("  Suggested: --val_ratio 0.1, --test_ratio 0.2")
    
    # 6. 检查数据质量
    print("\n6. Checking data quality...")
    
    # 检查前几个文件
    print("\n  Checking first few NPZ files...")
    checked = 0
    for patient, files in patient_to_files.items():
        for npz_file in files[:2]:  # 每个患者检查2个文件
            try:
                import numpy as np
                data = np.load(npz_file, allow_pickle=True)
                keys = list(data.keys())
                print(f"    ✓ {Path(npz_file).name}: {len(keys)} arrays")
                checked += 1
                if checked >= 5:
                    break
            except Exception as e:
                print(f"    ✗ {Path(npz_file).name}: Error - {e}")
        if checked >= 5:
            break
    
    return True


def main():
    """运行诊断"""
    labels_csv = r'E:\output\connectivity_features\labels.csv'
    features_root = r'E:\output\connectivity_features'
    
    # 允许命令行参数
    if len(sys.argv) > 1:
        labels_csv = sys.argv[1]
    if len(sys.argv) > 2:
        features_root = sys.argv[2]
    
    success = check_data_split(labels_csv, features_root)
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Diagnostic complete!")
        print("\nYou can now run training with appropriate parameters.")
        return 0
    else:
        print("\n" + "=" * 60)
        print("✗ Issues found!")
        print("\nPlease fix the issues above before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

