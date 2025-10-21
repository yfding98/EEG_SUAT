#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_marked_files.py

收集所有含有 marked_abnormal_segments.csv 的结果目录，
将对应的 .set 文件和整个结果目录复制到新目录并打包成 zip 文件。

使用方法:
    python collect_marked_files.py --root_dir "E:\DataSet\EEG\EEG dataset_SUAT_processed" --out_dir "E:\output"
"""

import os
import shutil
import zipfile
import argparse
import re
from pathlib import Path


def find_marked_results(root_dir):
    """
    递归查找所有包含 marked_abnormal_segments.csv 的目录
    
    返回: list of dict, 每个dict包含:
        - csv_path: marked_abnormal_segments.csv 的路径
        - result_dir: 结果目录路径
        - set_file: 对应的 .set 文件路径
        - relative_path: 相对于root_dir的路径（用于保持目录结构）
    """
    marked_results = []
    
    for root, dirs, files in os.walk(root_dir):
        if 'marked_abnormal_segments.csv' in files:
            csv_path = os.path.join(root, 'marked_abnormal_segments.csv')
            result_dir = root
            
            # 结果目录名应该以 _results 结尾
            if not result_dir.endswith('_results'):
                print(f"Warning: 找到 marked_abnormal_segments.csv 但目录不是以 _results 结尾: {result_dir}")
                continue
            
            # 获取对应的 .set 文件
            # 例如: SZ2_preICA_reject_1_postICA_results -> SZ2_preICA_reject_1_postICA.set
            result_dir_name = os.path.basename(result_dir)
            set_name = result_dir_name.replace('_results', '.set')
            parent_dir = os.path.dirname(result_dir)
            set_file = os.path.join(parent_dir, set_name)
            
            # 检查 .set 文件是否存在
            if not os.path.exists(set_file):
                print(f"Warning: 找不到对应的 .set 文件: {set_file}")
                continue
            
            # 计算相对路径（从root_dir到父目录）
            try:
                relative_path = os.path.relpath(parent_dir, root_dir)
            except ValueError:
                # 如果路径不在同一驱动器，使用绝对路径的最后几级
                parts = Path(parent_dir).parts
                # 找到root_dir中的最后一级目录，作为起点
                root_parts = Path(root_dir).parts
                if len(root_parts) > 0:
                    root_last = root_parts[-1]
                    try:
                        idx = parts.index(root_last)
                        relative_path = os.path.join(*parts[idx+1:])
                    except ValueError:
                        relative_path = os.path.join(*parts[-2:])  # 取最后两级
                else:
                    relative_path = os.path.join(*parts[-2:])
            
            marked_results.append({
                'csv_path': csv_path,
                'result_dir': result_dir,
                'set_file': set_file,
                'relative_path': relative_path,
                'set_name': set_name,
                'result_dir_name': result_dir_name
            })
            
            print(f"Found: {relative_path}/{set_name}")
    
    return marked_results


def copy_files_to_output(marked_results, out_dir, file_types=['ica', 'reject', 'raw'], create_zip=True):
    """
    将找到的文件复制到输出目录，保持目录结构
    
    复制内容：
    1. ica (postICA文件): SZ1_preICA_reject_1_postICA.set + .fdt
    2. reject (reject文件): SZ1_preICA_reject_1.set + .fdt
    3. raw (原始文件): SZ1_preICA.set + .fdt
    4. 结果目录中只复制 marked_abnormal_segments.csv
    
    参数:
        marked_results: find_marked_results 返回的列表
        out_dir: 输出目录
        file_types: 要复制的文件类型列表，可选 'raw', 'reject', 'ica'
        create_zip: 是否创建 zip 文件
    """
    os.makedirs(out_dir, exist_ok=True)
    
    copied_count = 0
    
    # 显示要复制的文件类型
    types_display = {
        'raw': '原始数据',
        'reject': 'Reject处理后数据',
        'ica': 'ICA处理后数据'
    }
    selected_types = [types_display.get(t, t) for t in file_types]
    print(f"\n📋 将复制以下类型的文件: {', '.join(selected_types)}")
    print("=" * 80)
    
    for item in marked_results:
        print(f"\n处理: {item['relative_path']}/{item['set_name']}")
        print("-" * 60)
        
        # 创建目标目录（保持原有的目录结构）
        target_parent = os.path.join(out_dir, item['relative_path'])
        os.makedirs(target_parent, exist_ok=True)
        
        parent_dir = os.path.dirname(item['set_file'])
        
        # 辅助函数：复制单个文件及其对应的 .fdt
        def copy_set_and_fdt(set_file, set_name, label):
            if not os.path.exists(set_file):
                print(f"⚠ Not found: {set_name}")
                return False
            
            target_set = os.path.join(target_parent, set_name)
            try:
                shutil.copy2(set_file, target_set)
                print(f"✓ Copied [{label}]: {set_name}")
            except Exception as e:
                print(f"✗ Error copying {set_name}: {e}")
                return False
            
            # 复制对应的 .fdt 文件
            fdt_name = set_name.replace('.set', '.fdt')
            fdt_file = os.path.join(parent_dir, fdt_name)
            if os.path.exists(fdt_file):
                target_fdt = os.path.join(target_parent, fdt_name)
                try:
                    shutil.copy2(fdt_file, target_fdt)
                    print(f"✓ Copied [{label}]: {fdt_name}")
                except Exception as e:
                    print(f"✗ Error copying {fdt_name}: {e}")
            else:
                print(f"⚠ Not found: {fdt_name}")
            
            return True
        
        # 1. 复制 ICA 处理后的文件（postICA）
        if 'ica' in file_types:
            copy_set_and_fdt(item['set_file'], item['set_name'], 'ICA')
        
        # 2. 复制 reject 文件（去掉 _postICA）
        if 'reject' in file_types and '_postICA' in item['set_name']:
            reject_set_name = item['set_name'].replace('_postICA', '')
            reject_set_file = os.path.join(parent_dir, reject_set_name)
            copy_set_and_fdt(reject_set_file, reject_set_name, 'Reject')
        
        # 3. 复制原始文件（去掉 _reject_N 和 _postICA）
        if 'raw' in file_types:
            # 从 postICA 文件名推导原始文件名
            original_set_name = item['set_name'].replace('_postICA', '')
            original_set_name = re.sub(r'_reject_\d+', '', original_set_name)
            original_set_file = os.path.join(parent_dir, original_set_name)
            copy_set_and_fdt(original_set_file, original_set_name, 'Raw')
        
        # 4. 只复制 marked_abnormal_segments.csv
        target_result_dir = os.path.join(target_parent, item['result_dir_name'])
        os.makedirs(target_result_dir, exist_ok=True)
        
        target_csv = os.path.join(target_result_dir, 'marked_abnormal_segments.csv')
        try:
            shutil.copy2(item['csv_path'], target_csv)
            print(f"✓ Copied [Result]: {item['result_dir_name']}/marked_abnormal_segments.csv")
            copied_count += 1
        except Exception as e:
            print(f"✗ Error copying marked_abnormal_segments.csv: {e}")
    
    print("\n" + "=" * 80)
    print(f"✅ 总共处理了 {copied_count} 个标记结果")
    
    # 创建 zip 文件
    if create_zip and copied_count > 0:
        zip_path = os.path.join(os.path.dirname(out_dir), f"{os.path.basename(out_dir)}.zip")
        print(f"\n正在创建 ZIP 文件: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(out_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, out_dir)
                    zipf.write(file_path, arcname)
                    
        print(f"✅ ZIP 文件已创建: {zip_path}")
        
        # 显示 ZIP 文件大小
        zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"   文件大小: {zip_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="收集所有标记的 EEG 结果文件并打包",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 复制所有类型的文件
  python collect_marked_files.py --root_dir "E:\\data" --out_dir "E:\\output"
  
  # 只复制 ICA 处理后的文件
  python collect_marked_files.py --root_dir "E:\\data" --out_dir "E:\\output" --types ica
  
  # 复制原始数据和 Reject 处理后的数据
  python collect_marked_files.py --root_dir "E:\\data" --out_dir "E:\\output" --types raw reject
  
  # 复制所有数据但不打包
  python collect_marked_files.py --root_dir "E:\\data" --out_dir "E:\\output" --no-zip
        """
    )
    parser.add_argument(
        '--root_dir',
        required=True,
        help="数据集根目录，例如: E:\\DataSet\\EEG\\EEG dataset_SUAT_processed"
    )
    parser.add_argument(
        '--out_dir',
        required=True,
        help="输出目录，例如: E:\\output\\marked_results"
    )
    parser.add_argument(
        '--types',
        nargs='+',
        choices=['raw', 'reject', 'ica'],
        default=[ 'ica'],
        help="要复制的文件类型，可选: raw(原始数据), reject(Reject处理后), ica(ICA处理后)。默认复制所有类型。"
    )
    parser.add_argument(
        '--no-zip',
        action='store_true',
        help="不创建 ZIP 文件，仅复制文件"
    )
    
    args = parser.parse_args()
    
    print(f"正在扫描目录: {args.root_dir}")
    print("=" * 80)
    
    # 查找所有标记的结果
    marked_results = find_marked_results(args.root_dir)
    
    if not marked_results:
        print("\n❌ 没有找到任何包含 marked_abnormal_segments.csv 的目录")
        return
    
    print(f"\n找到 {len(marked_results)} 个标记结果")
    print("=" * 80)
    
    # 复制文件到输出目录
    copy_files_to_output(marked_results, args.out_dir, file_types=args.types, create_zip=not args.no_zip)
    
    print("\n✅ 完成！")


if __name__ == "__main__":
    # 如果不提供命令行参数，使用默认值
    import sys
    if len(sys.argv) == 1:
        print("使用默认参数运行...")
        sys.argv.extend([
            '--root_dir', r'E:\DataSet\EEG\EEG dataset_SUAT_processed',
            '--out_dir', r'E:\output\marked_results'
        ])
    
    main()

