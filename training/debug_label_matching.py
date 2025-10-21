#!/usr/bin/env python3
"""
诊断标签匹配问题 - 检查NPZ文件是否能正确匹配到标签
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path


def debug_label_matching(labels_csv_path, features_root):
    """诊断标签匹配"""
    print("=" * 80)
    print("Label Matching Diagnostic Tool")
    print("=" * 80)
    
    # 1. 加载labels.csv
    print("\n1. Loading labels.csv...")
    df = pd.read_csv(labels_csv_path, encoding='utf-8')
    print(f"  ✓ Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    
    # 2. 检查字段
    if 'features_dir_path' not in df.columns:
        print("  ✗ Missing 'features_dir_path' column!")
        return False
    
    if 'channel_combination' not in df.columns:
        print("  ✗ Missing 'channel_combination' column!")
        return False
    
    print("\n2. Analyzing features_dir_path...")
    print(f"  Sample paths:")
    for path in df['features_dir_path'].head(5):
        print(f"    - {path}")
    
    print("\n3. Analyzing channel_combination...")
    print(f"  Sample combinations:")
    for combo in df['channel_combination'].head(10):
        print(f"    - '{combo}' (type: {type(combo).__name__})")
    
    # 统计
    unique_combos = df['channel_combination'].unique()
    print(f"\n  Unique combinations: {len(unique_combos)}")
    
    # 检查空值
    empty_count = df['channel_combination'].isna().sum()
    empty_str_count = (df['channel_combination'] == '').sum()
    bracket_count = (df['channel_combination'] == '[]').sum()
    
    print(f"\n  Empty/Invalid values:")
    print(f"    NaN: {empty_count}")
    print(f"    Empty string: {empty_str_count}")
    print(f"    '[]': {bracket_count}")
    
    # 4. 实际匹配测试
    print("\n4. Testing file-to-label matching...")
    
    from datasets_multilabel import discover_patient_segments_from_csv
    
    patient_to_files = discover_patient_segments_from_csv(labels_csv_path, features_root)
    
    # 获取一些文件来测试
    test_files = []
    for patient, files in list(patient_to_files.items())[:3]:  # 前3个患者
        test_files.extend(files[:2])  # 每个患者2个文件
    
    print(f"\n  Testing {len(test_files)} files...")
    
    matched = 0
    unmatched = 0
    
    for npz_file in test_files:
        npz_file_normalized = npz_file.replace('\\', '/')
        
        # 尝试匹配
        found = False
        matched_combo = None
        
        for _, row in df.iterrows():
            features_dir_path = str(row['features_dir_path']).replace('\\', '/')
            
            if features_dir_path in npz_file_normalized:
                found = True
                matched_combo = row['channel_combination']
                break
        
        if found:
            matched += 1
            print(f"    ✓ {Path(npz_file).name}")
            print(f"      → '{matched_combo}'")
        else:
            unmatched += 1
            print(f"    ✗ {Path(npz_file).name}")
            print(f"      → NO MATCH FOUND")
            
            # 显示可能的匹配
            print(f"      File path parts: {Path(npz_file).parts[-3:]}")
            print(f"      CSV paths (first 3):")
            for path in df['features_dir_path'].head(3):
                print(f"        - {path}")
    
    print(f"\n  Match rate: {matched}/{len(test_files)} ({matched/len(test_files)*100:.1f}%)")
    
    # 5. 解析通道组合测试
    print("\n5. Testing channel parsing...")
    
    test_combos = df['channel_combination'].head(10)
    
    for combo in test_combos:
        # 使用相同的解析逻辑
        if not isinstance(combo, str):
            parsed = []
        else:
            # 清理
            cleaned = combo.strip()
            cleaned = cleaned.replace('[', '').replace(']', '')
            cleaned = cleaned.replace("'", '').replace('"', '')
            cleaned = cleaned.replace('(', '').replace(')', '')
            
            # 分割
            channels = []
            for sep in ['-', ',', ';', ' ', '|']:
                if sep in cleaned:
                    channels = [ch.strip() for ch in cleaned.split(sep) if ch.strip()]
                    break
            else:
                channels = [cleaned.strip()] if cleaned.strip() else []
            
            # 清理
            parsed = []
            for ch in channels:
                ch = ch.strip()
                ch = ''.join(c for c in ch if c.isalnum() or c in ['-', '_'])
                if ch and any(c.isalpha() for c in ch):
                    parsed.append(ch)
        
        print(f"  '{combo}' → {parsed}")
        
        if not parsed:
            print(f"    ⚠ WARNING: Parsed to empty list!")
    
    # 6. 建议
    print("\n" + "=" * 80)
    print("Summary and Recommendations:")
    print("=" * 80)
    
    if unmatched > 0:
        print(f"\n⚠ WARNING: {unmatched} files could not be matched to labels!")
        print("\nPossible issues:")
        print("  1. features_dir_path in CSV doesn't match actual file paths")
        print("  2. Path separator differences (\\  vs /)")
        print("  3. Relative vs absolute paths")
        
        print("\nSuggestions:")
        print("  1. Check your labels.csv file")
        print("  2. Verify features_dir_path format")
        print("  3. Run: python dataset_maker/scripts/generate_labels_csv.py")
    
    if empty_count + empty_str_count + bracket_count > 0:
        print(f"\n⚠ WARNING: Found {empty_count + empty_str_count + bracket_count} empty/invalid labels!")
        print("\nThis will result in all-zero label vectors!")
        print("\nSuggestions:")
        print("  1. Clean your labels.csv file")
        print("  2. Remove rows with empty channel_combination")
        print("  3. Or regenerate labels.csv")
    
    if matched == len(test_files) and empty_count + empty_str_count + bracket_count == 0:
        print("\n✓ Everything looks good!")
        print("  All files can be matched to labels")
        print("  No empty labels found")
    
    return matched > 0


def main():
    """运行诊断"""
    labels_csv = r'E:\output\connectivity_features\labels.csv'
    features_root = r'E:\output\connectivity_features'
    
    if len(sys.argv) > 1:
        labels_csv = sys.argv[1]
    if len(sys.argv) > 2:
        features_root = sys.argv[2]
    
    if not Path(labels_csv).exists():
        print(f"✗ Labels CSV not found: {labels_csv}")
        return 1
    
    success = debug_label_matching(labels_csv, features_root)
    
    if success:
        print("\n" + "=" * 80)
        print("✓ Diagnostic complete!")
        return 0
    else:
        print("\n" + "=" * 80)
        print("✗ Critical issues found!")
        print("\nPlease fix the issues above before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

