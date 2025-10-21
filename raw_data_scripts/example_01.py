import mne

# 加载 .edf 文件
file_path = r'E:\DataSet\EEG\EEG dataset_SUAT\头皮数据-6例\江仁坤\SZ1.edf'
raw = mne.io.read_raw_edf(file_path, preload=True)

print(raw.info)           # 采样率、通道数、设备类型等
raw.plot_psd(fmax=50)     # 查看功率谱密度
raw.plot(n_channels=20, duration=30, scalings='auto')  # 可视化原始信号


# # 重命名通道（例如 EDF 中的 A1-A20 改为 Fp1, F3...）
# mapping = {f'A{i}': f'EEG{i:03d}' for i in range(1, 21)}
# raw.rename_channels(mapping)

# 设置标准 10-20 系统位置（如果没有自动识别）
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

# 设置重参考 （均值）
raw.set_eeg_reference('average', projection=True)

# 带通滤波：0.5 - 40 Hz（保留 delta, theta, alpha, beta, low gamma）
raw.filter(0.5, 40., fir_design='firwin')

# 去除工频干扰（50Hz 或 60Hz）
raw.notch_filter([50., 100.])  # 若是 50Hz 国家；美国用 [60, 120]

"""
方法一：使用 ICA 去眼电（EOG）、肌电（EMG）
"""
# 找出 EOG 通道（或模拟一个垂直方向差分）
eog_epochs = mne.preprocessing.create_eog_epochs(raw)
eog_evoked = eog_epochs.average()

# 使用 ICA 分解
ica = mne.preprocessing.ICA(n_components=15, random_state=97)
ica.fit(raw)

# 自动或手动识别并移除眼电成分
ica.detect_artifacts(eog_epochs)
ica.exclude = [0]  # 示例：排除第一个成分（需根据 topomap 判断）

# 应用 ICA 修正
ica.apply(raw)

"""
方法二：手动标记坏段（适合癫痫发作间期尖波/棘波）
"""
# 使用交互式工具标记坏段
raw.plot(duration=30, n_channels=20)
# 在弹出窗口中按 'b' 键标记坏段，保存为 annotations


"""
从 Annotations 获取事件（如果有标注）
"""
# 假设你的文件已有“Interictal”、“Preictal”、“Ictal”等标签
events, event_id = mne.events_from_annotations(raw)

# 创建 epochs（例如每段 ±2 秒）
tmin, tmax = -1.0, 2.0
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                    baseline=None, preload=True, verbose=False)

"""
若无标注 → 手动检测异常事件（如尖波）
"""
# 示例：检测 >3 SD 的 beta/gamma 活动
from scipy import stats
import numpy as np

# 提取 beta 波段 (13-30Hz) 包络
raw_beta = raw.copy().filter(13, 30)
power = mne.time_frequency.tfr_morlet(raw_beta, freqs=[20], n_cycles=5, return_itc=False)
beta_power = power[0].data.mean(axis=0)  # 平均所有通道

# 检测异常高功率段
z_score = np.abs(stats.zscore(beta_power))
spike_mask = z_score > 3  # 超过 3 个标准差

# 转换为事件序列（可用于创建 Epochs）
onsets = np.where(np.diff(spike_mask.astype(int)) == 1)[0] + 1
durations = np.diff(np.where(np.diff(spike_mask.astype(int)) == -1)[0] + 1, prepend=0)
annotations = mne.Annotations(onsets / raw.info['sfreq'], durations, description='spike_candidate')
raw.set_annotations(annotations)


freqs = np.logspace(*np.log10([1, 40]), num=8)
power, itc = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=2, return_itc=True)



