#!/usr/bin/env python3
"""
测试通道解析和清理功能
"""

import sys
from pathlib import Path

# 测试用例
TEST_CASES = [
    # (输入, 期望输出)
    ("Fp1-F3-C3", ["Fp1", "F3", "C3"]),
    ("Fp1, F3, C3", ["Fp1", "F3", "C3"]),
    ("Fp1; F3; C3", ["Fp1", "F3", "C3"]),
    ("['Fp1', 'F3', 'C3']", ["Fp1", "F3", "C3"]),
    ("['Fp1','F3','C3']", ["Fp1", "F3", "C3"]),
    ('["Fp1", "F3", "C3"]', ["Fp1", "F3", "C3"]),
    ("Fp1-F3-C3-P4", ["Fp1", "F3", "C3", "P4"]),
    ("[]", []),
    ("", []),
    ("   ", []),
    ("Fp1", ["Fp1"]),
    ("  Fp1  ", ["Fp1"]),
    ("Fp1-,,-F3", ["Fp1", "F3"]),
    ("[Fp1, F3, C3]", ["Fp1", "F3", "C3"]),
    ("(Fp1-F3)", ["Fp1", "F3"]),
    # 带特殊字符的
    ("Fp1*-F3#-C3@", ["Fp1", "F3", "C3"]),
    # 空元素
    ("Fp1--F3", ["Fp1", "F3"]),
    ("Fp1,,F3", ["Fp1", "F3"]),
]


def clean_channel_string(combo_str: str):
    """
    复制 datasets_multilabel.py 中的清理逻辑
    """
    if not isinstance(combo_str, str):
        return []
    
    # 1. 移除方括号、引号等无关字符
    cleaned = combo_str.strip()
    cleaned = cleaned.replace('[', '').replace(']', '')
    cleaned = cleaned.replace("'", '').replace('"', '')
    cleaned = cleaned.replace('(', '').replace(')', '')
    
    # 2. 尝试不同的分隔符
    channels = []
    for sep in ['-', ',', ';', ' ', '|']:
        if sep in cleaned:
            channels = [ch.strip() for ch in cleaned.split(sep) if ch.strip()]
            break
    else:
        # 没有分隔符，可能是单个通道
        channels = [cleaned.strip()] if cleaned.strip() else []
    
    # 3. 进一步清理每个通道名称
    cleaned_channels = []
    for ch in channels:
        # 移除可能的前后缀
        ch = ch.strip()
        # 移除数字前后的特殊字符（但保留通道名中的数字）
        ch = ''.join(c for c in ch if c.isalnum() or c in ['-', '_'])
        
        # 只保留有效的通道名（字母+数字组合）
        if ch and any(c.isalpha() for c in ch):
            cleaned_channels.append(ch)
    
    return cleaned_channels


def test_channel_parsing():
    """测试通道解析"""
    print("=" * 60)
    print("Channel Parsing Test")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for input_str, expected in TEST_CASES:
        result = clean_channel_string(input_str)
        
        # 检查结果
        if result == expected:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1
        
        # 显示结果
        print(f"\n{status}")
        print(f"  Input:    '{input_str}'")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")
    
    # 总结
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


def test_with_real_data():
    """使用真实数据测试"""
    print("\n" + "=" * 60)
    print("Testing with Real Data")
    print("=" * 60)
    
    labels_csv = r'E:\output\connectivity_features\labels.csv'
    
    if not Path(labels_csv).exists():
        print(f"Labels CSV not found: {labels_csv}")
        print("Skipping real data test")
        return True
    
    import pandas as pd
    
    print(f"\nLoading: {labels_csv}")
    df = pd.read_csv(labels_csv, encoding='utf-8')
    
    print(f"Total rows: {len(df)}")
    
    if 'channel_combination' not in df.columns:
        print("Error: 'channel_combination' column not found")
        return False
    
    # 显示一些示例
    print("\nSample channel combinations (before cleaning):")
    unique_combos = df['channel_combination'].unique()[:10]
    for combo in unique_combos:
        print(f"  '{combo}'")
    
    # 清理并显示结果
    print("\nAfter cleaning:")
    for combo in unique_combos:
        cleaned = clean_channel_string(combo)
        print(f"  '{combo}' → {cleaned}")
    
    # 统计
    print("\n" + "=" * 60)
    print("Statistics:")
    print("=" * 60)
    
    all_channels = set()
    invalid_count = 0
    
    for combo in df['channel_combination']:
        channels = clean_channel_string(combo)
        if not channels:
            invalid_count += 1
            print(f"  Warning: Empty result for '{combo}'")
        all_channels.update(channels)
    
    print(f"\nTotal unique channels discovered: {len(all_channels)}")
    print(f"Invalid combinations: {invalid_count}")
    print(f"\nAll channels: {sorted(all_channels)}")
    
    # 检查是否有异常的通道名
    print("\nChecking for unusual channel names...")
    unusual = []
    for ch in sorted(all_channels):
        # 检查长度
        if len(ch) > 10:
            unusual.append(f"{ch} (too long)")
        # 检查是否只包含数字
        elif ch.isdigit():
            unusual.append(f"{ch} (only digits)")
        # 检查是否有非ASCII字符
        elif not ch.isascii():
            unusual.append(f"{ch} (non-ASCII)")
    
    if unusual:
        print("  Found unusual channel names:")
        for u in unusual:
            print(f"    - {u}")
    else:
        print("  ✓ All channel names look normal")
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "🔍 Channel Parsing and Cleaning Test".center(60))
    print()
    
    # 测试1: 单元测试
    test1_pass = test_channel_parsing()
    
    # 测试2: 真实数据测试
    test2_pass = test_with_real_data()
    
    # 总结
    print("\n" + "=" * 60)
    if test1_pass and test2_pass:
        print("✓ ALL TESTS PASSED!")
        print("\nChannel parsing is working correctly.")
        print("Invalid characters will be automatically cleaned.")
        return 0
    else:
        print("✗ SOME TESTS FAILED!")
        print("\nPlease review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

