#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动应用加速方案到 extract_connectivity_features.py
使用方法: python auto_apply_acceleration.py [1|2|3]
    1 = joblib并行
    2 = numba JIT
    3 = multiprocess
"""

import sys
import os


def apply_acceleration(method_id):
    """应用加速方案"""
    
    target_file = 'extract_connectivity_features.py'
    
    if not os.path.exists(target_file):
        print(f"错误: 找不到 {target_file}")
        return False
    
    # 读取原文件
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 备份
    backup_file = target_file + '.backup'
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ 已备份原文件到: {backup_file}")
    
    # 根据选择的方案修改
    if method_id == 1:
        # joblib方案
        import_line = """from connectivity_acceleration_joblib import (
    compute_granger_causality_pairwise_parallel,
    compute_transfer_entropy_pairwise_parallel
)
"""
        gc_call = "compute_granger_causality_pairwise_parallel(data, sfreq, n_jobs=-1)"
        te_call = "compute_transfer_entropy_pairwise_parallel(data, n_jobs=-1)"
        method_name = "joblib并行"
        
    elif method_id == 2:
        # numba方案
        import_line = """from connectivity_acceleration_numba import (
    compute_granger_causality_pairwise_threaded,
    compute_transfer_entropy_pairwise_numba
)
"""
        gc_call = "compute_granger_causality_pairwise_threaded(data, sfreq)"
        te_call = "compute_transfer_entropy_pairwise_numba(data)"
        method_name = "numba JIT"
        
    elif method_id == 3:
        # multiprocess方案
        import_line = """from connectivity_acceleration_multiprocess import (
    compute_granger_causality_pairwise_batch,
    compute_transfer_entropy_pairwise_batch
)
"""
        gc_call = "compute_granger_causality_pairwise_batch(data, sfreq, batch_size=10)"
        te_call = "compute_transfer_entropy_pairwise_batch(data, batch_size=10)"
        method_name = "multiprocess批量"
        
    else:
        print("错误: 无效的方案编号")
        return False
    
    # 在import部分添加导入
    import_marker = "from itertools import combinations"
    if import_marker in content:
        content = content.replace(
            import_marker,
            import_marker + "\n\n" + import_line
        )
    else:
        print("警告: 找不到导入位置，请手动添加导入语句")
    
    # 替换Granger调用
    old_gc = "compute_granger_causality_pairwise(data, sfreq)"
    content = content.replace(old_gc, gc_call)
    
    # 替换Transfer Entropy调用
    old_te = "compute_transfer_entropy_pairwise(data)"
    content = content.replace(old_te, te_call)
    
    # 写回文件
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ 已应用 {method_name} 加速方案")
    print(f"✓ 修改的文件: {target_file}")
    print(f"\n如需恢复原文件，运行: cp {backup_file} {target_file}")
    
    return True


if __name__ == '__main__':
    print("使用方法: python auto_apply_acceleration.py [1|2|3]")
    print("  1 = joblib并行（推荐）")
    print("  2 = numba JIT（最快）")
    print("  3 = multiprocess（无额外依赖）")
    
    try:
        if len(sys.argv) == 1:
            method_id = 2
        elif len(sys.argv) > 2:
            raise ValueError()
        else:
            method_id = int(sys.argv[1])
        if method_id not in [1, 2, 3]:
            raise ValueError()
    except:
        print("错误: 请输入1、2或3")
        sys.exit(1)
    
    success = apply_acceleration(method_id)
    sys.exit(0 if success else 1)

