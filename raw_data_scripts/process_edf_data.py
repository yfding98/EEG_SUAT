"""
医院EDF数据处理和癫痫发作前期信号源定位
基于LaBraM项目的EEG处理流程
"""

import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from scipy import signal
from scipy.signal import butter, filtfilt
import warnings

warnings.filterwarnings('ignore')
# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 或者使用以下方法设置中文字体
try:
    import matplotlib.font_manager as fm
    # 查找系统中的中文字体
    chinese_fonts = [f.name for f in fm.fontManager.ttflist if 'SimHei' in f.name or 'Microsoft YaHei' in f.name or 'WenQuanYi' in f.name]
    if chinese_fonts:
        plt.rcParams['font.sans-serif'] = chinese_fonts[0]
        print(f"使用中文字体: {chinese_fonts[0]}")
    else:
        print("未找到中文字体，使用默认字体")
except:
    print("字体设置失败，使用默认字体")

class HospitalEEGProcessor:
    """医院EDF数据处理器，专门用于癫痫发作前期信号分析"""

    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.channels = None
        self.sfreq = 200  # 目标采样频率
        self.original_sfreq = None

        # 医院数据通道配置
        self.hospital_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                                  'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                                  'A2', 'Fz', 'Cz', 'Pz', 'Sph-R', 'Sph-L', 'ECG',
                                  'EMG1', 'EMG2', '27', '28', '32','29', '30', '31', 'DC', 'OSat', 'PR']

        # 需要删除的通道（对应EEGLAB的pop_select操作）
        self.drop_channels = ['A1', 'A2', 'Sph-R', 'Sph-L', 'ECG', 'EMG1', 'EMG2',
                              '27', '28', '29', '30', '31', '32', 'DC', 'OSat', 'PR']

        # 标准EEG通道（保留的通道）
        self.standard_eeg_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                                      'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                                      'Fz', 'Cz', 'Pz']

        # 源定位相关参数
        self.source_space = None
        self.forward_model = None
        self.inverse_operator = None

    def load_hospital_edf(self, file_path):
        """加载医院EDF文件"""
        try:
            print(f"正在加载医院EDF文件: {file_path}")
            raw = mne.io.read_raw_edf(file_path, preload=True)

            print(f"原始通道: {raw.ch_names}")
            print(f"原始采样频率: {raw.info['sfreq']} Hz")
            print(f"数据时长: {raw.times[-1]:.2f} 秒")

            self.original_sfreq = raw.info['sfreq']

            # 检查通道是否匹配
            missing_channels = set(self.hospital_channels) - set(raw.ch_names)
            if missing_channels:
                print(f"警告: 缺少以下通道: {missing_channels}")

            self.raw_data = raw
            return True

        except Exception as e:
            print(f"加载EDF文件失败: {e}")
            return False

    def apply_channel_selection(self):
        """应用通道选择（只保留标准EEG通道）"""
        if self.raw_data is None:
            print("错误: 请先加载EDF文件")
            return False
        
        try:
            # 获取当前所有通道
            current_channels = self.raw_data.ch_names
            print(f"原始通道: {current_channels}")
            
            # 只保留标准EEG通道
            channels_to_keep = []
            for ch in current_channels:
                if ch in self.standard_eeg_channels:
                    channels_to_keep.append(ch)
            
            # 删除不在标准EEG通道列表中的通道
            channels_to_drop = [ch for ch in current_channels if ch not in self.standard_eeg_channels]
            
            if channels_to_drop:
                print(f"删除通道: {channels_to_drop}")
                self.raw_data.drop_channels(channels_to_drop)
            
            # 检查剩余通道
            remaining_channels = self.raw_data.ch_names
            print(f"保留的标准EEG通道: {remaining_channels}")
            print(f"保留通道数量: {len(remaining_channels)}")
            
            if len(remaining_channels) < 10:
                print("警告: EEG通道数量较少，可能影响分析精度")
            
            return True
            
        except Exception as e:
            print(f"通道选择失败: {e}")
            return False

    def apply_preprocessing(self, l_freq=0.1, h_freq=75.0):
        """应用预处理（滤波和重采样）"""
        if self.raw_data is None:
            print("错误: 请先加载EDF文件")
            return False

        try:
            print("开始预处理...")

            # 带通滤波
            print(f"应用带通滤波: {l_freq}-{h_freq} Hz")
            self.raw_data.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=4)

            # 陷波滤波去除工频干扰
            print("应用陷波滤波: 50 Hz")
            self.raw_data.notch_filter(50.0, n_jobs=4)

            # 重采样到200Hz
            if self.original_sfreq != self.sfreq:
                print(f"重采样: {self.original_sfreq} Hz -> {self.sfreq} Hz")
                self.raw_data.resample(self.sfreq, n_jobs=4)

            # 获取处理后的数据
            self.processed_data = self.raw_data.get_data(units='uV')
            self.channels = self.raw_data.ch_names

            print(f"预处理完成!")
            print(f"处理后数据形状: {self.processed_data.shape}")
            print(f"处理后通道: {self.channels}")

            return True

        except Exception as e:
            print(f"预处理失败: {e}")
            return False

    def detect_ictal_activity(self, window_size=2.0, overlap=0.5, threshold_std=3.0):
        """检测癫痫发作活动"""
        if self.processed_data is None:
            print("错误: 请先完成预处理")
            return None

        try:
            print("开始检测癫痫发作活动...")

            # 计算窗口参数
            window_samples = int(window_size * self.sfreq)
            overlap_samples = int(overlap * self.sfreq)
            step_samples = window_samples - overlap_samples

            # 存储检测结果
            ictal_segments = []
            channel_activities = []

            # 对每个通道进行分析
            for ch_idx, ch_name in enumerate(self.channels):
                if ch_name not in self.standard_eeg_channels:
                    continue

                ch_data = self.processed_data[ch_idx, :]

                # 滑动窗口分析
                window_activities = []
                window_times = []

                for start_idx in range(0, len(ch_data) - window_samples, step_samples):
                    end_idx = start_idx + window_samples
                    window_data = ch_data[start_idx:end_idx]

                    # 计算窗口内的特征
                    # 1. 功率谱密度
                    freqs, psd = signal.welch(window_data, fs=self.sfreq, nperseg=min(256, len(window_data)))

                    # 2. 高频活动检测（20-40Hz）
                    high_freq_mask = (freqs >= 20) & (freqs <= 40)
                    high_freq_power = np.sum(psd[high_freq_mask])

                    # 3. 低频活动检测（0.5-4Hz）
                    low_freq_mask = (freqs >= 0.5) & (freqs <= 4)
                    low_freq_power = np.sum(psd[low_freq_mask])

                    # 4. 总功率
                    total_power = np.sum(psd)

                    # 5. 信号变异性
                    signal_variance = np.var(window_data)

                    # 综合活动指标
                    activity_score = (high_freq_power + low_freq_power) / total_power * signal_variance

                    window_activities.append(activity_score)
                    window_times.append(start_idx / self.sfreq)

                # 检测异常活动
                if len(window_activities) > 0:
                    activities = np.array(window_activities)
                    mean_activity = np.mean(activities)
                    std_activity = np.std(activities)

                    # 异常阈值
                    threshold = mean_activity + threshold_std * std_activity

                    # 找到异常窗口
                    abnormal_windows = np.where(activities > threshold)[0]

                    if len(abnormal_windows) > 0:
                        print(f"通道 {ch_name} 检测到 {len(abnormal_windows)} 个异常窗口")

                        # 合并相邻的异常窗口
                        segments = self._merge_adjacent_segments(abnormal_windows, window_times)

                        for segment in segments:
                            ictal_segments.append({
                                'channel': ch_name,
                                'start_time': segment[0],
                                'end_time': segment[1],
                                'activity_score': np.max(activities[abnormal_windows])
                            })

            # 按时间排序
            ictal_segments.sort(key=lambda x: x['start_time'])

            print(f"总共检测到 {len(ictal_segments)} 个癫痫活动段")

            return ictal_segments

        except Exception as e:
            print(f"癫痫活动检测失败: {e}")
            return None

    def detect_preictal_activity(self, window_size=10.0, overlap=0.8, lookback_minutes=30):
        """检测发作前期活动"""
        if self.processed_data is None:
            print("错误: 请先完成预处理")
            return None
        
        try:
            print("开始检测发作前期活动...")
            
            # 计算窗口参数
            window_samples = int(window_size * self.sfreq)
            step_samples = int(window_size * self.sfreq * (1 - overlap))
            
            # 只分析最近30分钟的数据
            lookback_samples = int(lookback_minutes * 60 * self.sfreq)
            start_idx = max(0, self.processed_data.shape[1] - lookback_samples)
            analysis_data = self.processed_data[:, start_idx:]
            
            preictal_segments = []
            
            # 对每个通道进行分析
            for ch_idx, ch_name in enumerate(self.channels):
                if ch_name not in self.standard_eeg_channels:
                    continue
                
                ch_data = analysis_data[ch_idx, :]
                window_features = []
                window_times = []
                
                # 滑动窗口分析
                for start_sample in range(0, len(ch_data) - window_samples, step_samples):
                    end_sample = start_sample + window_samples
                    window_data = ch_data[start_sample:end_sample]
                    
                    # 计算多种特征
                    features = self._extract_preictal_features(window_data)
                    window_features.append(features)
                    window_times.append((start_idx + start_sample) / self.sfreq)
                
                # 检测异常模式
                if len(window_features) > 10:  # 需要足够的数据点
                    anomalies = self._detect_preictal_anomalies(window_features, window_times)
                    
                    for anomaly in anomalies:
                        preictal_segments.append({
                            'channel': ch_name,
                            'start_time': anomaly['start_time'],
                            'end_time': anomaly['end_time'],
                            'confidence': anomaly['confidence'],
                            'features': anomaly['features'],
                            'type': 'preictal'
                        })
            
            # 按时间排序
            preictal_segments.sort(key=lambda x: x['start_time'])
            
            print(f"检测到 {len(preictal_segments)} 个发作前期活动段")
            return preictal_segments
            
        except Exception as e:
            print(f"发作前期检测失败: {e}")
            return None

    def _merge_adjacent_segments(self, abnormal_windows, window_times, max_gap=1.0):
        """合并相邻的异常窗口"""
        if len(abnormal_windows) == 0:
            return []

        segments = []
        current_start = window_times[abnormal_windows[0]]
        current_end = window_times[abnormal_windows[0]] + 2.0  # 窗口大小

        for i in range(1, len(abnormal_windows)):
            window_time = window_times[abnormal_windows[i]]

            if window_time - current_end <= max_gap:
                # 合并到当前段
                current_end = window_time + 2.0
            else:
                # 开始新段
                segments.append((current_start, current_end))
                current_start = window_time
                current_end = window_time + 2.0

        # 添加最后一段
        segments.append((current_start, current_end))

        return segments

    def perform_source_localization(self, ictal_segments, method='dSPM'):
        """执行源定位分析"""
        if self.processed_data is None or len(ictal_segments) == 0:
            print("错误: 请先完成预处理和癫痫活动检测")
            return None

        try:
            print("开始源定位分析...")

            # 创建MNE info对象
            info = mne.create_info(
                ch_names=self.channels,
                sfreq=self.sfreq,
                ch_types=['eeg'] * len(self.channels)
            )

            # 设置标准10-20电极位置
            montage = mne.channels.make_standard_montage('standard_1020')
            info.set_montage(montage, on_missing='ignore')

            # 创建Raw对象
            raw = mne.io.RawArray(self.processed_data, info)

            # 设置EEG参考电极（这是源定位必需的）
            print("设置EEG参考电极...")
            raw.set_eeg_reference(projection=True)

            # 创建源空间
            print("创建源空间...")

            subjects_dir = mne.utils.get_config('SUBJECTS_DIR')
            print(f"使用SUBJECTS_DIR: {subjects_dir}")

            # 检查fsaverage是否存在
            fsaverage_path = os.path.join(subjects_dir, 'fsaverage')
            if not os.path.exists(fsaverage_path):
                print(f"❌ fsaverage不存在: {fsaverage_path}")
                return self._simple_source_localization(ictal_segments)

            src = mne.setup_source_space(subject='fsaverage', spacing='ico4',
                                         subjects_dir=subjects_dir, add_dist=False)

            # 创建正向模型
            print("创建正向模型...")
            trans = mne.transforms.Transform('head', 'mri')
            bem = mne.make_bem_model(subject='fsaverage', subjects_dir=subjects_dir)
            bem_sol = mne.make_bem_solution(bem)

            fwd = mne.make_forward_solution(
                info, trans=trans, src=src, bem=bem_sol,
                meg=False, eeg=True, mindist=5.0, n_jobs=4
            )

            # 创建逆算子
            print("创建逆算子...")
            noise_cov = mne.make_ad_hoc_cov(info)
            inv = mne.minimum_norm.make_inverse_operator(
                info, fwd, noise_cov, loose=0.2, depth=0.8
            )

            # 对每个癫痫活动段进行源定位
            source_results = []

            for segment in ictal_segments:
                print(f"分析段: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s")

                # 提取时间段数据
                start_sample = int(segment['start_time'] * self.sfreq)
                end_sample = int(segment['end_time'] * self.sfreq)

                epoch_data = self.processed_data[:, start_sample:end_sample]

                # 创建Epochs对象
                epochs = mne.EpochsArray(
                    epoch_data.reshape(1, epoch_data.shape[0], epoch_data.shape[1]),
                    info, tmin=0
                )

                # 应用参考电极投影
                epochs.set_eeg_reference(projection=True)

                # 计算源活动
                stc = mne.minimum_norm.apply_inverse(
                    epochs.average(), inv, method=method, pick_ori='normal'
                )

                # 找到最大激活源
                max_vertex = np.argmax(np.abs(stc.data))
                max_time = np.argmax(np.abs(stc.data[max_vertex, :]))
                max_amplitude = stc.data[max_vertex, max_time]

                source_results.append({
                    'segment': segment,
                    'source_vertex': max_vertex,
                    'max_amplitude': max_amplitude,
                    'source_time': max_time / self.sfreq,
                    'stc': stc
                })

            print(f"源定位分析完成，共分析 {len(source_results)} 个段")

            return source_results

        except Exception as e:
            print(f"源定位分析失败: {e}")
            print("使用简化源定位方法...")
            return self._simple_source_localization(ictal_segments)

    def _simple_source_localization(self, ictal_segments):
        """简化的源定位方法（不依赖FreeSurfer和nibabel）"""
        if self.processed_data is None or len(ictal_segments) == 0:
            print("错误: 请先完成预处理和癫痫活动检测")
            return None

        try:
            print("开始简化源定位分析...")

            source_results = []

            for segment in ictal_segments:
                print(f"分析段: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s")

                # 提取时间段数据
                start_sample = int(segment['start_time'] * self.sfreq)
                end_sample = int(segment['end_time'] * self.sfreq)

                # 计算每个通道在该时间段的活动强度
                channel_activities = []
                for ch_idx, ch_name in enumerate(self.channels):
                    if ch_name in self.standard_eeg_channels:
                        ch_data = self.processed_data[ch_idx, start_sample:end_sample]
                        
                        # 使用多种指标计算活动强度
                        variance = np.var(ch_data)
                        amplitude = np.max(np.abs(ch_data))
                        power = np.mean(ch_data**2)
                        
                        # 计算频域特征
                        freqs, psd = signal.welch(ch_data, fs=self.sfreq, nperseg=min(256, len(ch_data)))
                        high_freq_power = np.sum(psd[(freqs >= 20) & (freqs <= 40)])
                        low_freq_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4)])
                        
                        # 综合活动指标
                        activity_score = variance * amplitude * power * (high_freq_power + low_freq_power)
                        
                        channel_activities.append({
                            'channel': ch_name,
                            'activity_score': activity_score,
                            'variance': variance,
                            'amplitude': amplitude,
                            'power': power,
                            'high_freq_power': high_freq_power,
                            'low_freq_power': low_freq_power
                        })

                # 按活动强度排序
                channel_activities.sort(key=lambda x: x['activity_score'], reverse=True)

                # 找到活动最强的通道
                strongest_channel = channel_activities[0]['channel']
                max_activity = channel_activities[0]['activity_score']

                source_results.append({
                    'segment': segment,
                    'strongest_channel': strongest_channel,
                    'max_activity': max_activity,
                    'channel_activities': channel_activities,
                    'method': 'simplified'
                })

            print(f"简化源定位分析完成，共分析 {len(source_results)} 个段")
            return source_results

        except Exception as e:
            print(f"简化源定位分析失败: {e}")
            return None

    def _extract_preictal_features(self, window_data):
        """提取发作前期特征"""
        features = {}
        
        # 1. 频域特征
        freqs, psd = signal.welch(window_data, fs=self.sfreq, nperseg=min(512, len(window_data)))
        
        # 不同频段功率
        features['delta_power'] = np.sum(psd[(freqs >= 0.5) & (freqs <= 4)])
        features['theta_power'] = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
        features['alpha_power'] = np.sum(psd[(freqs >= 8) & (freqs <= 13)])
        features['beta_power'] = np.sum(psd[(freqs >= 13) & (freqs <= 30)])
        features['gamma_power'] = np.sum(psd[(freqs >= 30) & (freqs <= 50)])
        
        # 2. 时域特征
        features['variance'] = np.var(window_data)
        features['skewness'] = self._calculate_skewness(window_data)
        features['kurtosis'] = self._calculate_kurtosis(window_data)
        
        # 3. 棘波检测
        features['spike_count'] = self._detect_spikes(window_data)
        features['spike_amplitude'] = self._calculate_spike_amplitude(window_data)
        
        # 4. 复杂度特征
        features['sample_entropy'] = self._calculate_sample_entropy(window_data)
        features['hurst_exponent'] = self._calculate_hurst_exponent(window_data)
        
        return features

    def _detect_preictal_anomalies(self, window_features, window_times):
        """检测发作前期异常模式"""
        anomalies = []
        
        if len(window_features) < 20:
            return anomalies
        
        # 转换为numpy数组
        feature_matrix = np.array([[f[k] for k in sorted(f.keys())] for f in window_features])
        
        # 使用滑动窗口统计检测异常
        window_size = 10  # 10个窗口的统计窗口
        
        for i in range(window_size, len(window_features) - window_size):
            # 计算当前窗口与历史窗口的差异
            current_features = feature_matrix[i]
            historical_features = feature_matrix[i-window_size:i]
            
            # 计算Z-score
            mean_hist = np.mean(historical_features, axis=0)
            std_hist = np.std(historical_features, axis=0)
            z_scores = np.abs((current_features - mean_hist) / (std_hist + 1e-8))
            
            # 综合异常分数
            anomaly_score = np.mean(z_scores)
            
            # 棘波活动检测
            spike_score = window_features[i]['spike_count'] * window_features[i]['spike_amplitude']
            
            # 复杂度变化检测
            complexity_change = abs(window_features[i]['sample_entropy'] - 
                                   np.mean([wf['sample_entropy'] for wf in window_features[i-window_size:i]]))
            
            # 综合判断
            if anomaly_score > 2.0 or spike_score > 0.1 or complexity_change > 0.1:
                confidence = min(1.0, (anomaly_score + spike_score + complexity_change) / 3.0)
                
                anomalies.append({
                    'start_time': window_times[i],
                    'end_time': window_times[i] + 10.0,  # 10秒窗口
                    'confidence': confidence,
                    'features': window_features[i]
                })
        
        return anomalies

    def _calculate_skewness(self, data):
        """计算偏度"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data):
        """计算峰度"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3

    def _detect_spikes(self, data, threshold=3.0):
        """检测棘波"""
        std = np.std(data)
        spikes = np.abs(data) > threshold * std
        return np.sum(spikes)

    def _calculate_spike_amplitude(self, data, threshold=3.0):
        """计算棘波幅度"""
        std = np.std(data)
        spikes = data[np.abs(data) > threshold * std]
        return np.mean(np.abs(spikes)) if len(spikes) > 0 else 0

    def _calculate_sample_entropy(self, data, m=2, r=0.2):
        """计算样本熵 - 最优化版本"""
        N = len(data)
        if N < m + 1:
            return 0
        
        r = r * np.std(data)
        
        def _sample_entropy_fast(U, m):
            """快速样本熵计算"""
            N = len(U)
            if N < m + 1:
                return 0
            
            # 创建模板矩阵
            templates = np.array([U[i:i+m] for i in range(N - m + 1)])
            n_templates = len(templates)
            
            if n_templates < 2:
                return 0
            
            # 使用广播计算所有模板对的距离
            # 重塑模板矩阵以便广播
            t1 = templates[:, np.newaxis, :]  # (n, 1, m)
            t2 = templates[np.newaxis, :, :]  # (1, n, m)
            
            # 计算最大距离矩阵
            distances = np.max(np.abs(t1 - t2), axis=2)  # (n, n)
            
            # 计算匹配数（排除自匹配）
            np.fill_diagonal(distances, np.inf)  # 排除自匹配
            matches = np.sum(distances <= r, axis=1)
            
            # 计算phi值
            if np.any(matches == 0):
                return 0
            
            phi = np.mean(np.log(matches / (n_templates - 1)))
            return phi
        
        phi_m = _sample_entropy_fast(data, m)
        phi_m1 = _sample_entropy_fast(data, m + 1)
        
        if phi_m == 0 or phi_m1 == 0:
            return 0
        
        return phi_m - phi_m1

    def _calculate_hurst_exponent(self, data):
        """计算Hurst指数"""
        N = len(data)
        if N < 10:
            return 0.5
        
        # 简化的Hurst指数计算
        lags = range(2, min(20, N//4))
        tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag]))) for lag in lags]
        
        if len(tau) < 2:
            return 0.5
        
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    def visualize_results(self, ictal_segments, source_results=None, preictal_segments=None, save_path=None):
        """可视化分析结果"""
        if self.processed_data is None:
            print("错误: 请先完成预处理")
            return
        
        try:
            # 创建时间轴
            time_axis = np.arange(self.processed_data.shape[1]) / self.sfreq
            
            # 创建图形 - 增加一个子图用于发作前期
            fig, axes = plt.subplots(4, 1, figsize=(15, 16))
            
            # 1. 原始EEG信号
            axes[0].set_title("原始EEG信号")
            for i, ch_name in enumerate(self.channels[:10]):
                if ch_name in self.standard_eeg_channels:
                    axes[0].plot(time_axis, self.processed_data[i, :] + i * 100,
                                 label=ch_name, linewidth=0.5)
            axes[0].set_ylabel("通道")
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            
            # 2. 发作前期检测结果
            axes[1].set_title("发作前期活动检测结果")
            axes[1].plot(time_axis, np.mean(self.processed_data, axis=0),
                         label="平均信号", linewidth=1, color="blue")
            
            if preictal_segments:
                colors = plt.cm.viridis(np.linspace(0, 1, len(preictal_segments)))
                for i, segment in enumerate(preictal_segments):
                    color = colors[i]
                    label = f"发作前期 ({segment['channel']}, 置信度: {segment['confidence']:.2f})"
                    axes[1].axvspan(segment['start_time'], segment['end_time'],
                                    alpha=0.4, color=color, label=label)
            
            axes[1].set_ylabel("幅度 (μV)")
            axes[1].legend()
            
            # 3. 癫痫活动检测结果
            axes[2].set_title("癫痫发作期活动检测结果")
            axes[2].plot(time_axis, np.mean(self.processed_data, axis=0),
                         label="平均信号", linewidth=1, color="blue")
            
            if len(ictal_segments) > 0:
                colors = plt.cm.Set3(np.linspace(0, 1, len(ictal_segments)))
                legend_added = set()
                
                for i, segment in enumerate(ictal_segments):
                    color = colors[i]
                    label = f"癫痫活动 ({segment['channel']})"
                    
                    if segment['channel'] not in legend_added:
                        axes[2].axvspan(segment['start_time'], segment['end_time'],
                                        alpha=0.3, color=color, label=label)
                        legend_added.add(segment['channel'])
                    else:
                        axes[2].axvspan(segment['start_time'], segment['end_time'],
                                        alpha=0.3, color=color)
            
            axes[2].set_ylabel("幅度 (μV)")
            axes[2].legend()
            
            # 4. 源定位结果
            if source_results:
                axes[3].set_title("源定位结果")
                
                if source_results[0].get("method") == "simplified":
                    # 简化版本的结果显示
                    source_times = [result['segment']['start_time'] for result in source_results]
                    source_channels = [result['strongest_channel'] for result in source_results]
                    source_activities = [result['max_activity'] for result in source_results]
                    
                    # 创建散点图
                    scatter = axes[3].scatter(source_times, source_activities, 
                                    c=range(len(source_times)), s=100, alpha=0.7, cmap="viridis")
                    
                    # 添加通道标签
                    for i, (time, channel) in enumerate(zip(source_times, source_channels)):
                        axes[3].annotate(channel, (time, source_activities[i]), 
                                       xytext=(5, 5), textcoords="offset points", fontsize=8)
                    
                    axes[3].set_ylabel("活动强度")
                    axes[3].set_xlabel("时间 (s)")
                    plt.colorbar(scatter, ax=axes[3], label="段编号")
                else:
                    # 完整版本的结果显示
                    source_times = [result['source_time'] for result in source_results]
                    source_amplitudes = [result['max_amplitude'] for result in source_results]
                    
                    axes[3].scatter(source_times, source_amplitudes, 
                                  c="red", s=100, alpha=0.7)
                    axes[3].set_ylabel("源幅度")
                    axes[3].set_xlabel("时间 (s)")
            else:
                axes[3].set_title("源定位结果 (未可用)")
                axes[3].text(0.5, 0.5, "源定位需要FreeSurfer环境", 
                           ha="center", va="center", transform=axes[3].transAxes)
                
                plt.tight_layout()
                
            # 保存图片（如果提供了保存路径）
            if save_path:
                try:
                    # 确保保存目录存在
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    # 保存图片
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"可视化结果已保存到: {save_path}")
                except Exception as e:
                    print(f"保存图片失败: {e}")
            else:
                plt.show()

        except Exception as e:
            print(f"可视化失败: {e}")

    def save_results(self, ictal_segments, source_results=None, output_dir='./results'):
        """保存分析结果"""
        try:
            os.makedirs(output_dir, exist_ok=True)

            # 保存癫痫活动检测结果
            if ictal_segments:
                np.save(os.path.join(output_dir, 'ictal_segments.npy'), ictal_segments)
                print(f"癫痫活动检测结果已保存到: {output_dir}/ictal_segments.npy")

            # 保存源定位结果
            if source_results:
                np.save(os.path.join(output_dir, 'source_results.npy'), source_results)
                print(f"源定位结果已保存到: {output_dir}/source_results.npy")

            # 保存处理后的数据
            if self.processed_data is not None:
                np.save(os.path.join(output_dir, 'processed_eeg_data.npy'), self.processed_data)
                np.save(os.path.join(output_dir, 'channel_names.npy'), self.channels)
                print(f"处理后数据已保存到: {output_dir}/processed_eeg_data.npy")

        except Exception as e:
            print(f"保存结果失败: {e}")


def main():
    """主函数示例"""
    import mne
    from pathlib import Path
    import os
    
    # 自定义下载路径
    download_dir = "D:/mne_data"
    
    # 创建目录（如果不存在）
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    
    subjects_dir = mne.datasets.fetch_fsaverage(download_dir, verbose=True)
    print(f"✅ fsaverage下载到 {subjects_dir}")
    
    # 设置SUBJECTS_DIR环境变量（注意：这里设置的是包含fsaverage的父目录）
    mne.utils.set_config("SUBJECTS_DIR", download_dir, set_env=True)
    print(f"✅ SUBJECTS_DIR 已设置: {download_dir}")
    
    # 验证设置
    config_subjects_dir = mne.utils.get_config("SUBJECTS_DIR")
    print(f"✅ 验证SUBJECTS_DIR: {config_subjects_dir}")
    
    # 设置根目录和输出目录
    root_dir = Path("E:/DataSet/EEG/EEG dataset_SUAT")
    output_dir = Path("./results")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有EDF文件
    edf_files = list(root_dir.rglob("*.edf"))
    print(f"找到 {len(edf_files)} 个EDF文件")
    
    # 遍历处理每个EDF文件
    for i, edf_file in enumerate(edf_files):
        print(f"\n=== 处理文件 {i+1}/{len(edf_files)}: {edf_file.name} ===")
        
        # 计算相对路径
        relative_path = edf_file.relative_to(root_dir)
        file_output_dir = output_dir / relative_path.parent / edf_file.stem
        
        # 创建该文件的输出目录
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 创建处理器
            processor = HospitalEEGProcessor()
            
            # 1. 加载EDF文件
            if not processor.load_hospital_edf(str(edf_file)):
                print(f"跳过文件: {edf_file.name}")
                continue
            
            # 2. 应用通道选择
            if not processor.apply_channel_selection():
                print(f"通道选择失败，跳过文件: {edf_file.name}")
                continue
            
            # 3. 应用预处理
            if not processor.apply_preprocessing():
                print(f"预处理失败，跳过文件: {edf_file.name}")
                continue
            
            # 4. 检测癫痫活动
            ictal_segments = processor.detect_ictal_activity()

            # 4.5. 检测发作前期活动
            preictal_segments = processor.detect_preictal_activity()
            
            # 5. 源定位分析（可选，需要FreeSurfer环境）
            source_results = None
            try:
                source_results = processor.perform_source_localization(ictal_segments)
            except Exception as e:
                print(f"源定位跳过: {e}")
            
            # 6. 可视化结果
            processor.visualize_results(ictal_segments, source_results, preictal_segments,
                          str(file_output_dir / "eeg_analysis.png"))
            
            # 7. 保存结果
            processor.save_results(ictal_segments, source_results, str(file_output_dir))
            
            print(f"✅ 文件 {edf_file.name} 处理完成")
            break

            
        except Exception as e:
            print(f"❌ 处理文件 {edf_file.name} 时出错: {e}")
            continue
    
    print(f"\n=== 所有文件处理完成 ===")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()