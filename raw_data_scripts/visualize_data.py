
from dataset_maker.shock.utils.eegUtils import preprocessing_edf
import matplotlib.pyplot as plt
import numpy as np


def comprehensive_eeg_inspection(edf_file_path):
    """
    完整的EEG数据查看和分析
    """

    # 处理数据
    eeg_data, channels = preprocessing_edf(edf_file_path)
    
    if eeg_data is None:
        print("处理失败")
        return
    
    # 基本信息
    print("=== EEG数据基本信息 ===")
    print(f"数据形状: {eeg_data.shape}")
    print(f"通道数量: {eeg_data.shape[0]}")
    print(f"采样点数: {eeg_data.shape[1]}")
    print(f"时长: {eeg_data.shape[1] / 200:.2f} 秒")
    print(f"数据范围: {eeg_data.min():.2f} ~ {eeg_data.max():.2f} μV")
    
    # 可视化
    time_axis = np.arange(eeg_data.shape[1]) / 200
    
    plt.figure(figsize=(15, 8))
    # for i in range(len(channels)):
    for i in range(min(3, len(channels))):
        # plt.subplot(len(channels), 1, i+1)
        plt.subplot(3, 1, i + 1)
        plt.plot(time_axis, eeg_data[i, :], linewidth=0.5)
        plt.ylim(-300, 300)
        plt.xlim(0, 100)
        plt.title(f'{channels[i]}')
        plt.ylabel('μV')
        if i == 4:
        # if i==len(channels)-1:
            plt.xlabel('Time (seconds)')
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('EEG time series')
    plt.tight_layout()
    plt.show()
    
    return eeg_data, channels

# 使用示例
# eeg_data, channels = comprehensive_eeg_inspection("your_file.edf")
edf_file = r"E:\DataSet\EEG\EEG dataset_SUAT\头皮数据-6例\江仁坤\SZ1.edf"
eeg_data, channels = comprehensive_eeg_inspection(edf_file)

def visualize_eeg_with_mne_noninteractive(edf_file_path):
    """
    使用MNE但保存为图片，避免交互式窗口问题
    """
    print(f"正在处理文件: {edf_file_path}")
    
    # 处理EDF文件
    eeg_data, channels = preprocessing_edf(edf_file_path)
    
    if eeg_data is None:
        print("处理失败")
        return
    
    # 创建MNE的Raw对象
    info = mne.create_info(
        ch_names=channels,
        sfreq=200,
        ch_types=['eeg'] * len(channels)
    )
    
    raw = mne.io.RawArray(eeg_data, info)
    
    # 1. 保存原始数据图为PNG
    fig = raw.plot(duration=10, n_channels=min(8, len(channels)), scalings='auto', show=False)
    fig.savefig('eeg_raw_data.png', dpi=300, bbox_inches='tight')
    print("原始数据图已保存为: eeg_raw_data.png")
    
    # 2. 保存功率谱密度图
    fig = raw.plot_psd(fmax=50, show=False)
    fig.savefig('eeg_power_spectrum.png', dpi=300, bbox_inches='tight')
    print("功率谱密度图已保存为: eeg_power_spectrum.png")
    
    # 关闭图形以释放内存
    plt.close('all')
    
    return eeg_data, channels

import mne


visualize_eeg_with_mne_noninteractive(edf_file)

# # 创建MNE的Raw对象进行可视化
# info = mne.create_info(
#     ch_names=channels,
#     sfreq=200,
#     ch_types=['eeg'] * len(channels)
# )

# raw = mne.io.RawArray(eeg_data, info)

# # 1. 绘制原始数据
# raw.plot(duration=10, n_channels=min(8, len(channels)), scalings='auto')

# # 2. 绘制功率谱密度
# raw.plot_psd(fmax=50)

# 3. 绘制数据分布
# raw.plot_psd_topomap()

# 4. 绘制通道位置（如果有montage信息）
# raw.set_montage('standard_1020')  # 如果有标准电极位置
# raw.plot_sensors()