#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_connectivity_setup.py

测试EEG连接性特征提取工具的安装和基本功能

使用方法:
    python test_connectivity_setup.py
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """测试所有必要的包是否正确安装"""
    print("Testing package imports...")
    print("-" * 60)
    
    packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'scipy': 'SciPy',
        'mne': 'MNE-Python',
        'sklearn': 'scikit-learn',
        'statsmodels': 'statsmodels',
        'networkx': 'NetworkX',
        'matplotlib': 'Matplotlib'
    }
    
    failed = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name:<20} - OK")
        except ImportError as e:
            print(f"✗ {name:<20} - FAILED")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Please install with: pip install -r connectivity_requirements.txt")
        return False
    else:
        print("\n✓ All packages installed successfully!")
        return True


def test_basic_functions():
    """测试基本函数是否可以运行"""
    print("\n" + "="*60)
    print("Testing basic functions...")
    print("-" * 60)
    
    try:
        import numpy as np
        from scipy import signal
        from scipy.signal import butter, filtfilt, hilbert
        
        # 生成测试数据
        print("Generating test data...")
        sfreq = 250  # 250 Hz
        duration = 5  # 5 seconds
        n_channels = 4
        t = np.linspace(0, duration, int(sfreq * duration))
        
        # 模拟EEG信号（多个频率成分）
        data = np.zeros((n_channels, len(t)))
        for ch in range(n_channels):
            # 添加不同频率成分
            data[ch] = (np.sin(2 * np.pi * 10 * t + ch * 0.5) +  # 10 Hz (alpha)
                       0.5 * np.sin(2 * np.pi * 5 * t + ch * 0.3) +  # 5 Hz (theta)
                       0.3 * np.random.randn(len(t)))  # 噪声
        
        print(f"Test data shape: {data.shape}")
        
        # 测试带通滤波
        print("Testing bandpass filter...")
        nyq = 0.5 * sfreq
        low = 8 / nyq
        high = 13 / nyq
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, data, axis=1)
        print(f"✓ Bandpass filter works")
        
        # 测试Hilbert变换
        print("Testing Hilbert transform...")
        analytic = hilbert(filtered, axis=1)
        phase = np.angle(analytic)
        envelope = np.abs(analytic)
        print(f"✓ Hilbert transform works")
        
        # 测试相关计算
        print("Testing correlation...")
        corr = np.corrcoef(data)
        print(f"✓ Correlation works (shape: {corr.shape})")
        
        # 测试PLV计算
        print("Testing PLV calculation...")
        plv_matrix = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                phase_diff = phase[i] - phase[j]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv_matrix[i, j] = plv
                plv_matrix[j, i] = plv
        np.fill_diagonal(plv_matrix, 1.0)
        print(f"✓ PLV calculation works (mean PLV: {np.mean(plv_matrix[np.triu_indices(n_channels, k=1)]):.3f})")
        
        # 测试图网络功能
        print("Testing graph metrics...")
        import networkx as nx
        G = nx.from_numpy_array(np.abs(corr))
        degree = dict(G.degree())
        clustering = nx.clustering(G)
        print(f"✓ Graph metrics work (mean degree: {np.mean(list(degree.values())):.3f})")
        
        print("\n✓ All basic functions work correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mne_io():
    """测试MNE读取功能（使用示例数据）"""
    print("\n" + "="*60)
    print("Testing MNE-Python I/O...")
    print("-" * 60)
    
    try:
        import mne
        import numpy as np
        
        # 创建模拟Raw对象
        print("Creating synthetic EEG data...")
        sfreq = 250
        n_channels = 10
        duration = 10  # 10 seconds
        
        # 创建info对象
        ch_names = [f'Ch{i+1}' for i in range(n_channels)]
        ch_types = ['eeg'] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        
        # 生成数据
        t = np.linspace(0, duration, int(sfreq * duration))
        data = np.zeros((n_channels, len(t)))
        for ch in range(n_channels):
            data[ch] = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
        
        # 创建Raw对象
        raw = mne.io.RawArray(data, info, verbose='ERROR')
        print(f"✓ Created MNE Raw object: {raw}")
        print(f"  - Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  - Duration: {raw.times[-1]:.2f} seconds")
        print(f"  - Channels: {len(raw.ch_names)}")
        
        # 测试数据提取
        print("Testing data extraction...")
        segment = raw.get_data(start=0, stop=int(5*sfreq))
        print(f"✓ Extracted segment shape: {segment.shape}")
        
        print("\n✓ MNE-Python I/O works correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ MNE test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extraction_module():
    """测试特征提取模块是否可以导入"""
    print("\n" + "="*60)
    print("Testing feature extraction module...")
    print("-" * 60)
    
    try:
        # 尝试导入主脚本中的函数
        import sys
        import os
        
        # 添加当前目录到路径
        sys.path.insert(0, os.path.dirname(__file__))
        
        print("Attempting to import extract_connectivity_features...")
        # 注意：这里只测试能否被Python解析，不实际运行
        with open('extract_connectivity_features.py', 'r') as f:
            code = f.read()
            compile(code, 'extract_connectivity_features.py', 'exec')
        
        print("✓ extract_connectivity_features.py is valid Python code")
        
        print("Attempting to import analyze_connectivity_features...")
        with open('analyze_connectivity_features.py', 'r') as f:
            code = f.read()
            compile(code, 'analyze_connectivity_features.py', 'exec')
        
        print("✓ analyze_connectivity_features.py is valid Python code")
        
        print("\n✓ All modules are valid!")
        return True
        
    except Exception as e:
        print(f"\n❌ Module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_system_info():
    """打印系统信息"""
    print("\n" + "="*60)
    print("System Information")
    print("="*60)
    
    import platform
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        
        import scipy
        print(f"SciPy version: {scipy.__version__}")
        
        import pandas as pd
        print(f"Pandas version: {pd.__version__}")
        
        import mne
        print(f"MNE-Python version: {mne.__version__}")
        
        import sklearn
        print(f"scikit-learn version: {sklearn.__version__}")
        
        import networkx as nx
        print(f"NetworkX version: {nx.__version__}")
    except:
        pass


def main():
    """运行所有测试"""
    print("="*60)
    print("EEG Connectivity Feature Extraction - Setup Test")
    print("="*60)
    
    results = []
    
    # 测试1: 包导入
    results.append(("Package imports", test_imports()))
    
    # 测试2: 基本函数
    results.append(("Basic functions", test_basic_functions()))
    
    # 测试3: MNE I/O
    results.append(("MNE-Python I/O", test_mne_io()))
    
    # 测试4: 模块导入
    results.append(("Module validation", test_feature_extraction_module()))
    
    # 打印系统信息
    print_system_info()
    
    # 汇总结果
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:<25} {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All tests passed! Your setup is ready to use.")
        print("\nNext steps:")
        print("1. Prepare your .set files")
        print("2. Run: python extract_connectivity_features.py --help")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install missing packages: pip install -r connectivity_requirements.txt")
        print("2. Update your packages: pip install --upgrade -r connectivity_requirements.txt")
        print("3. Check Python version (requires Python 3.7+)")
        return 1


if __name__ == "__main__":
    sys.exit(main())

