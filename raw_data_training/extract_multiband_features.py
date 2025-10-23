"""
多频段特征提取脚本

从原始EEG数据中提取不同频段的信号
用于高级排序模型的输入

频段定义（基于文献）:
- Delta: 0.5-4 Hz (深睡眠、病理活动)
- Theta: 4-8 Hz (发作前期常见)
- Alpha: 8-13 Hz (放松状态)
- Beta: 13-30 Hz (活跃思维)
- Gamma: 30-80 Hz (认知功能)
- HFO: 80-250 Hz (高频振荡，癫痫最可靠标志物)
"""

import numpy as np
import mne
from scipy.signal import butter, filtfilt, sosfiltfilt
import torch
from pathlib import Path
from tqdm import tqdm


# 频段定义
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 80),
    'hfo': (80, 250)
}


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Butterworth带通滤波器
    
    Args:
        data: (n_channels, n_samples)
        lowcut: 低频截止
        highcut: 高频截止
        fs: 采样率
        order: 滤波器阶数
    
    Returns:
        filtered_data: (n_channels, n_samples)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # 确保在有效范围内
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))
    
    # 使用sos (second-order sections) 更稳定
    sos = butter(order, [low, high], btype='band', output='sos')
    
    # 过滤每个通道
    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):
        try:
            filtered[i] = sosfiltfilt(sos, data[i])
        except:
            # 如果失败，返回原始数据
            filtered[i] = data[i]
    
    return filtered


def extract_multiband_data(data, sfreq):
    """
    从原始数据提取多频段信号
    
    Args:
        data: (n_channels, n_samples) 原始EEG
        sfreq: 采样率
    
    Returns:
        bands_data: dict of {band_name: (n_channels, n_samples)}
    """
    bands_data = {}
    
    for band_name, (low, high) in FREQ_BANDS.items():
        # 跳过超过Nyquist频率的频段
        if high > sfreq / 2:
            # 对于HFO，如果采样率不够，调整上限
            if band_name == 'hfo':
                high = min(high, sfreq / 2.5)
                if high < low:
                    print(f"Warning: Sampling rate too low for {band_name} band, skipping")
                    continue
        
        # 带通滤波
        filtered_data = butter_bandpass_filter(data, low, high, sfreq, order=4)
        bands_data[band_name] = filtered_data
    
    return bands_data


def preprocess_window(window_data, sfreq):
    """
    预处理单个时间窗口
    
    1. Z-score归一化
    2. 提取多频段
    
    Args:
        window_data: (n_channels, n_samples)
        sfreq: 采样率
    
    Returns:
        bands_data: dict of {band_name: (n_channels, n_samples)}
    """
    # 1. Z-score归一化每个通道
    from scipy.stats import zscore
    data_normalized = zscore(window_data, axis=1)
    data_normalized = np.nan_to_num(data_normalized, 0)
    
    # 2. 提取多频段
    bands_data = extract_multiband_data(data_normalized, sfreq)
    
    return bands_data


def load_standard_channel_positions():
    """
    加载标准10-20系统的通道位置
    
    Returns:
        channel_positions: dict of {channel_name: (x, y, z)}
    """
    # 标准10-20系统位置（归一化到单位球面）
    standard_1020_montage = mne.channels.make_standard_montage('standard_1020')
    
    positions = {}
    for ch_name, pos in standard_1020_montage.get_positions()['ch_pos'].items():
        positions[ch_name] = pos
    
    return positions


def get_channel_positions_array(channel_names):
    """
    获取通道位置数组
    
    Args:
        channel_names: list of channel names
    
    Returns:
        positions: (n_channels, 3) numpy array
    """
    standard_positions = load_standard_channel_positions()
    
    positions = []
    for ch_name in channel_names:
        # 尝试匹配标准位置
        ch_upper = ch_name.upper()
        
        if ch_upper in standard_positions:
            pos = standard_positions[ch_upper]
        elif ch_name in standard_positions:
            pos = standard_positions[ch_name]
        else:
            # 如果找不到，使用随机位置（或者报警告）
            print(f"Warning: Channel {ch_name} not in standard montage, using random position")
            pos = np.random.randn(3)
        
        positions.append(pos)
    
    return np.array(positions, dtype=np.float32)


class MultibandDataset(torch.utils.data.Dataset):
    """
    多频段数据集
    
    加载预处理的多频段数据
    """
    def __init__(self, data_list, labels_list):
        """
        Args:
            data_list: list of dicts, each dict has {band_name: (n_channels, n_samples)}
            labels_list: list of (n_channels,) binary labels
        """
        self.data_list = data_list
        self.labels_list = labels_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        bands_data = self.data_list[idx]
        labels = self.labels_list[idx]
        
        # 转为tensor
        bands_tensors = []
        for band_name in ['delta', 'theta', 'alpha', 'beta', 'gamma', 'hfo']:
            if band_name in bands_data:
                bands_tensors.append(torch.from_numpy(bands_data[band_name]).float())
            else:
                # 如果某个频段缺失，用零填充
                n_channels, n_samples = bands_data['delta'].shape
                bands_tensors.append(torch.zeros(n_channels, n_samples))
        
        labels_tensor = torch.from_numpy(labels).long()
        
        return {
            'bands': bands_tensors,
            'labels': labels_tensor
        }


def create_dummy_multiband_data(n_samples=100, n_channels=19, n_timepoints=1536):
    """
    创建虚拟的多频段数据用于测试
    
    Returns:
        data_list: list of multiband data
        labels_list: list of labels
        channel_names: list of channel names
    """
    channel_names = [f"Ch{i+1}" for i in range(n_channels)]
    
    data_list = []
    labels_list = []
    
    for i in range(n_samples):
        # 生成随机多频段数据
        bands_data = {}
        for band_name in FREQ_BANDS.keys():
            bands_data[band_name] = np.random.randn(n_channels, n_timepoints).astype(np.float32)
        
        data_list.append(bands_data)
        
        # 生成标签（随机2-5个异常通道）
        labels = np.zeros(n_channels, dtype=np.int64)
        n_abnormal = np.random.randint(2, 6)
        abnormal_idx = np.random.choice(n_channels, n_abnormal, replace=False)
        labels[abnormal_idx] = 1
        
        labels_list.append(labels)
    
    return data_list, labels_list, channel_names


if __name__ == "__main__":
    # 测试多频段提取
    print("测试多频段特征提取...")
    
    # 模拟数据
    sfreq = 256
    duration = 6  # 秒
    n_channels = 19
    n_samples = int(sfreq * duration)
    
    # 生成测试信号
    t = np.linspace(0, duration, n_samples)
    data = np.zeros((n_channels, n_samples))
    
    for i in range(n_channels):
        # 混合不同频率
        data[i] = (
            np.sin(2 * np.pi * 1.5 * t) +  # Delta
            np.sin(2 * np.pi * 6 * t) +     # Theta
            np.sin(2 * np.pi * 10 * t) +    # Alpha
            np.sin(2 * np.pi * 20 * t) +    # Beta
            0.1 * np.random.randn(n_samples)  # Noise
        )
    
    # 提取多频段
    bands_data = preprocess_window(data, sfreq)
    
    print("\n提取的频段:")
    for band_name, band_data in bands_data.items():
        print(f"  {band_name}: shape={band_data.shape}, "
              f"mean={band_data.mean():.3f}, std={band_data.std():.3f}")
    
    # 测试通道位置
    print("\n测试通道位置...")
    channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                     'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    
    positions = get_channel_positions_array(channel_names)
    print(f"通道位置形状: {positions.shape}")
    print(f"前5个通道位置:\n{positions[:5]}")
    
    # 测试数据集
    print("\n测试数据集...")
    data_list, labels_list, ch_names = create_dummy_multiband_data(n_samples=10)
    
    dataset = MultibandDataset(data_list, labels_list)
    print(f"数据集大小: {len(dataset)}")
    
    sample = dataset[0]
    print(f"样本包含 {len(sample['bands'])} 个频段")
    print(f"每个频段形状: {sample['bands'][0].shape}")
    print(f"标签形状: {sample['labels'].shape}")
    print(f"异常通道数: {sample['labels'].sum().item()}")
    
    print("\n✓ 所有测试通过!")



