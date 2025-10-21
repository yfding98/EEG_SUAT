import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, cheby1, cheby2, ellip
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
from pathlib import Path
from sklearn.decomposition import FastICA
import warnings
warnings.filterwarnings('ignore')


class AdvancedEEGProcessor:
    """对标EEGLAB的高级EEG数据处理工具"""
    
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.channels = None
        self.channel_types = None
        self.sfreq = 200
        self.original_sfreq = 200
        self.current_window_start = 0
        self.window_duration = 10
        self.selected_channels = []
        
        # 电极信息
        self.montage = None
        self.channel_positions = None
        self.channel_connectivity = None
        
        # 分段信息
        self.epochs = None
        self.epochs_info = None
        self.baseline_corrected = False
        
        # ICA信息
        self.ica = None
        self.ica_components = None
        self.excluded_components = []
        
        # 事件信息
        self.events = None
        self.event_id = None
        
    def load_edf_file(self, file_path, montage_type='standard_1020'):
        """加载EDF文件并设置电极位置"""
        try:
            print(f"正在加载文件: {file_path}")
            raw = mne.io.read_raw_edf(file_path, preload=True)
            
            # 获取数据
            self.raw_data = raw.get_data(units='uV')
            self.channels = raw.ch_names
            self.channel_types = raw.get_channel_types()
            self.original_sfreq = raw.info['sfreq']
            self.sfreq = self.original_sfreq
            self.processed_data = self.raw_data.copy()
            
            # 设置电极位置
            self.set_montage(montage_type)
            
            print(f"文件加载成功!")
            print(f"数据形状: {self.raw_data.shape}")
            print(f"通道数量: {len(self.channels)}")
            print(f"采样频率: {self.sfreq} Hz")
            print(f"时长: {self.raw_data.shape[1] / self.sfreq:.2f} 秒")
            print(f"通道名称: {self.channels}")
            
            return True
            
        except Exception as e:
            print(f"加载文件失败: {e}")
            return False
    
    def set_montage(self, montage_type='standard_1020'):
        """设置电极位置信息"""
        try:
            # 创建MNE info对象
            info = mne.create_info(
                ch_names=self.channels,
                sfreq=self.sfreq,
                ch_types=['eeg'] * len(self.channels)
            )
            
            # 设置montage
            if montage_type == 'standard_1020':
                montage = mne.channels.make_standard_montage('standard_1020')
            elif montage_type == 'standard_1005':
                montage = mne.channels.make_standard_montage('standard_1005')
            elif montage_type == 'biosemi64':
                montage = mne.channels.make_standard_montage('biosemi64')
            else:
                montage = mne.channels.make_standard_montage('standard_1020')
            
            # 应用montage到info
            info.set_montage(montage, on_missing='ignore')
            self.montage = info
            
            # 获取电极位置
            if hasattr(info, 'ch_names'):
                self.channel_positions = {}
                for i, ch_name in enumerate(info.ch_names):
                    if ch_name in info.chs:
                        pos = info.chs[i]['loc'][:3]  # x, y, z坐标
                        self.channel_positions[ch_name] = pos
            
            print(f"电极位置设置完成: {montage_type}")
            return True
            
        except Exception as e:
            print(f"设置电极位置失败: {e}")
            return False
    
    def apply_filter(self, filter_type='bandpass', low_freq=None, high_freq=None, 
                    filter_method='butter', order=4, ripple=0.1):
        """应用各种类型的滤波器"""
        if self.processed_data is None:
            return False
            
        try:
            nyquist = self.sfreq / 2
            
            if filter_type == 'lowpass':
                if high_freq is None:
                    high_freq = 40.0
                cutoff = high_freq / nyquist
                btype = 'low'
                
            elif filter_type == 'highpass':
                if low_freq is None:
                    low_freq = 0.5
                cutoff = low_freq / nyquist
                btype = 'high'
                
            elif filter_type == 'bandpass':
                if low_freq is None:
                    low_freq = 0.5
                if high_freq is None:
                    high_freq = 40.0
                cutoff = [low_freq / nyquist, high_freq / nyquist]
                btype = 'band'
                
            elif filter_type == 'bandstop':
                if low_freq is None:
                    low_freq = 48.0
                if high_freq is None:
                    high_freq = 52.0
                cutoff = [low_freq / nyquist, high_freq / nyquist]
                btype = 'bandstop'
            else:
                print(f"不支持的滤波器类型: {filter_type}")
                return False
            
            # 设计滤波器
            if filter_method == 'butter':
                b, a = butter(order, cutoff, btype=btype)
            elif filter_method == 'cheby1':
                b, a = cheby1(order, ripple, cutoff, btype=btype)
            elif filter_method == 'cheby2':
                b, a = cheby2(order, ripple, cutoff, btype=btype)
            elif filter_method == 'ellip':
                b, a = ellip(order, ripple, ripple, cutoff, btype=btype)
            else:
                b, a = butter(order, cutoff, btype=btype)
            
            # 应用滤波器
            for i in range(self.processed_data.shape[0]):
                self.processed_data[i, :] = filtfilt(b, a, self.processed_data[i, :])
            
            print(f"{filter_type}滤波完成: {low_freq}-{high_freq} Hz, 方法: {filter_method}")
            return True
            
        except Exception as e:
            print(f"滤波失败: {e}")
            return False
    
    def resample_data(self, new_sfreq):
        """降采样数据"""
        if self.processed_data is None:
            return False
            
        try:
            if new_sfreq >= self.sfreq:
                print("新采样频率必须小于当前采样频率")
                return False
            
            # 计算降采样因子
            decimation_factor = int(self.sfreq / new_sfreq)
            
            # 应用抗混叠滤波
            nyquist = self.sfreq / 2
            cutoff = new_sfreq / 2
            b, a = butter(4, cutoff / nyquist, btype='low')
            
            # 滤波后降采样
            filtered_data = np.zeros_like(self.processed_data)
            for i in range(self.processed_data.shape[0]):
                filtered_data[i, :] = filtfilt(b, a, self.processed_data[i, :])
            
            # 降采样
            self.processed_data = filtered_data[:, ::decimation_factor]
            self.sfreq = new_sfreq
            
            print(f"降采样完成: {self.original_sfreq} Hz -> {self.sfreq} Hz")
            return True
            
        except Exception as e:
            print(f"降采样失败: {e}")
            return False
    
    def apply_rereference(self, ref_type='average', ref_channels=None):
        """重参考化"""
        if self.processed_data is None:
            return False
            
        try:
            if ref_type == 'average':
                # 平均参考
                avg_ref = np.mean(self.processed_data, axis=0, keepdims=True)
                self.processed_data = self.processed_data - avg_ref
                print("平均参考完成")
                
            elif ref_type == 'mastoid':
                # 双耳参考（乳突参考）
                if ref_channels is None:
                    # 寻找乳突电极
                    mastoid_channels = []
                    for ch in self.channels:
                        if 'M1' in ch.upper() or 'A1' in ch.upper():
                            mastoid_channels.append(ch)
                        elif 'M2' in ch.upper() or 'A2' in ch.upper():
                            mastoid_channels.append(ch)
                    
                    if len(mastoid_channels) >= 2:
                        ref_channels = mastoid_channels[:2]
                    else:
                        print("未找到乳突电极，使用平均参考")
                        return self.apply_rereference('average')
                
                # 计算参考信号
                ref_indices = [self.channels.index(ch) for ch in ref_channels if ch in self.channels]
                if len(ref_indices) >= 2:
                    ref_signal = np.mean(self.processed_data[ref_indices, :], axis=0, keepdims=True)
                    self.processed_data = self.processed_data - ref_signal
                    print(f"双耳参考完成: {ref_channels}")
                
            elif ref_type == 'nose':
                # 鼻尖参考
                nose_channels = []
                for ch in self.channels:
                    if 'NZ' in ch.upper() or 'NOSE' in ch.upper():
                        nose_channels.append(ch)
                
                if nose_channels:
                    ref_index = self.channels.index(nose_channels[0])
                    ref_signal = self.processed_data[ref_index:ref_index+1, :]
                    self.processed_data = self.processed_data - ref_signal
                    print(f"鼻尖参考完成: {nose_channels[0]}")
                else:
                    print("未找到鼻尖电极，使用平均参考")
                    return self.apply_rereference('average')
                
            elif ref_type == 'single':
                # 单电极参考
                if ref_channels is None or len(ref_channels) == 0:
                    print("请指定参考电极")
                    return False
                
                ref_channel = ref_channels[0]
                if ref_channel in self.channels:
                    ref_index = self.channels.index(ref_channel)
                    ref_signal = self.processed_data[ref_index:ref_index+1, :]
                    self.processed_data = self.processed_data - ref_signal
                    print(f"单电极参考完成: {ref_channel}")
                else:
                    print(f"参考电极 {ref_channel} 不存在")
                    return False
            
            return True
            
        except Exception as e:
            print(f"重参考化失败: {e}")
            return False
    
    def create_epochs(self, events, event_id, tmin=-0.2, tmax=0.8, baseline=(None, 0)):
        """创建分段数据"""
        if self.processed_data is None:
            return False
            
        try:
            # 创建MNE info对象
            info = mne.create_info(
                ch_names=self.channels,
                sfreq=self.sfreq,
                ch_types=['eeg'] * len(self.channels)
            )
            
            # 创建Raw对象
            raw = mne.io.RawArray(self.processed_data, info)
            
            # 创建Epochs对象
            epochs = mne.Epochs(raw, events, event_id, tmin, tmax, 
                              baseline=baseline, preload=True)
            
            self.epochs = epochs
            self.epochs_info = {
                'tmin': tmin,
                'tmax': tmax,
                'baseline': baseline,
                'n_epochs': len(epochs),
                'event_id': event_id
            }
            
            print(f"分段创建完成: {len(epochs)} 个分段")
            return True
            
        except Exception as e:
            print(f"分段创建失败: {e}")
            return False
    
    def apply_baseline_correction(self, baseline=(None, 0)):
        """基线矫正"""
        if self.epochs is None:
            print("请先创建分段数据")
            return False
            
        try:
            self.epochs.apply_baseline(baseline)
            self.baseline_corrected = True
            print(f"基线矫正完成: {baseline}")
            return True
            
        except Exception as e:
            print(f"基线矫正失败: {e}")
            return False
    
    def reject_bad_epochs(self, reject_criteria=None):
        """剔除坏段"""
        if self.epochs is None:
            print("请先创建分段数据")
            return False
            
        try:
            if reject_criteria is None:
                reject_criteria = {
                    'eeg': 100e-6,  # 100 μV
                    'eog': 200e-6   # 200 μV
                }
            
            # 剔除坏段
            self.epochs.drop_bad(reject=reject_criteria)
            
            print(f"坏段剔除完成，剩余分段: {len(self.epochs)}")
            return True
            
        except Exception as e:
            print(f"坏段剔除失败: {e}")
            return False
    
    def run_ica(self, n_components=None, method='fastica'):
        """运行ICA分析"""
        if self.processed_data is None:
            return False
            
        try:
            if n_components is None:
                n_components = min(15, len(self.channels) - 1)
            
            # 创建MNE info对象
            info = mne.create_info(
                ch_names=self.channels,
                sfreq=self.sfreq,
                ch_types=['eeg'] * len(self.channels)
            )
            
            # 创建Raw对象
            raw = mne.io.RawArray(self.processed_data, info)
            
            # 运行ICA
            self.ica = mne.preprocessing.ICA(
                n_components=n_components,
                method=method,
                random_state=97
            )
            self.ica.fit(raw)
            
            self.ica_components = self.ica.get_components()
            
            print(f"ICA分析完成: {n_components} 个成分")
            return True
            
        except Exception as e:
            print(f"ICA分析失败: {e}")
            return False
    
    def detect_artifacts(self, artifact_types=['eog', 'ecg', 'muscle', 'line_noise']):
        """检测各种伪迹"""
        if self.ica is None:
            print("请先运行ICA分析")
            return False
            
        try:
            # 创建MNE info对象
            info = mne.create_info(
                ch_names=self.channels,
                sfreq=self.sfreq,
                ch_types=['eeg'] * len(self.channels)
            )
            
            raw = mne.io.RawArray(self.processed_data, info)
            
            detected_artifacts = {}
            
            # 检测眼动伪迹
            if 'eog' in artifact_types:
                eog_indices, eog_scores = self.ica.find_bads_eog(raw)
                if len(eog_indices) > 0:
                    detected_artifacts['eog'] = eog_indices
                    print(f"检测到眼动伪迹成分: {eog_indices}")
            
            # 检测心电伪迹
            if 'ecg' in artifact_types:
                ecg_indices, ecg_scores = self.ica.find_bads_ecg(raw)
                if len(ecg_indices) > 0:
                    detected_artifacts['ecg'] = ecg_indices
                    print(f"检测到心电伪迹成分: {ecg_indices}")
            
            # 检测肌电伪迹
            if 'muscle' in artifact_types:
                muscle_indices, muscle_scores = self.ica.find_bads_muscle(raw)
                if len(muscle_indices) > 0:
                    detected_artifacts['muscle'] = muscle_indices
                    print(f"检测到肌电伪迹成分: {muscle_indices}")
            
            # 检测工频干扰
            if 'line_noise' in artifact_types:
                line_noise_indices, line_noise_scores = self.ica.find_bads_line_noise(raw)
                if len(line_noise_indices) > 0:
                    detected_artifacts['line_noise'] = line_noise_indices
                    print(f"检测到工频干扰成分: {line_noise_indices}")
            
            return detected_artifacts
            
        except Exception as e:
            print(f"伪迹检测失败: {e}")
            return {}
    
    def remove_artifacts(self, artifact_components):
        """移除伪迹成分"""
        if self.ica is None:
            print("请先运行ICA分析")
            return False
            
        try:
            # 设置要排除的成分
            self.ica.exclude = artifact_components
            self.excluded_components = artifact_components
            
            # 应用ICA
            info = mne.create_info(
                ch_names=self.channels,
                sfreq=self.sfreq,
                ch_types=['eeg'] * len(self.channels)
            )
            
            raw = mne.io.RawArray(self.processed_data, info)
            raw_clean = self.ica.apply(raw)
            
            self.processed_data = raw_clean.get_data()
            
            print(f"伪迹移除完成，排除成分: {artifact_components}")
            return True
            
        except Exception as e:
            print(f"伪迹移除失败: {e}")
            return False
    
    def plot_ica_components(self, n_components=8):
        """绘制ICA成分"""
        if self.ica is None:
            print("请先运行ICA分析")
            return False
            
        try:
            # 创建MNE info对象
            info = mne.create_info(
                ch_names=self.channels,
                sfreq=self.sfreq,
                ch_types=['eeg'] * len(self.channels)
            )
            
            raw = mne.io.RawArray(self.processed_data, info)
            
            # 绘制ICA成分
            fig = self.ica.plot_components(inst=raw, picks=range(min(n_components, self.ica.n_components_)))
            return fig
            
        except Exception as e:
            print(f"绘制ICA成分失败: {e}")
            return None
    
    def plot_ica_sources(self, picks=None):
        """绘制ICA源信号"""
        if self.ica is None:
            print("请先运行ICA分析")
            return False
            
        try:
            # 创建MNE info对象
            info = mne.create_info(
                ch_names=self.channels,
                sfreq=self.sfreq,
                ch_types=['eeg'] * len(self.channels)
            )
            
            raw = mne.io.RawArray(self.processed_data, info)
            
            # 绘制ICA源信号
            if picks is None:
                picks = range(min(8, self.ica.n_components_))
            
            fig = self.ica.plot_sources(inst=raw, picks=picks)
            return fig
            
        except Exception as e:
            print(f"绘制ICA源信号失败: {e}")
            return None
    
    def get_window_data(self, start_time=None, duration=None):
        """获取滑动窗口数据"""
        if self.processed_data is None:
            return None, None
            
        if start_time is not None:
            self.current_window_start = start_time
        if duration is not None:
            self.window_duration = duration
            
        start_sample = int(self.current_window_start * self.sfreq)
        end_sample = int((self.current_window_start + self.window_duration) * self.sfreq)
        
        start_sample = max(0, start_sample)
        end_sample = min(self.processed_data.shape[1], end_sample)
        
        window_data = self.processed_data[:, start_sample:end_sample]
        time_axis = np.arange(start_sample, end_sample) / self.sfreq
        
        return window_data, time_axis
    
    def select_channels(self, channel_indices):
        """选择要查看的通道"""
        if self.processed_data is None:
            return False
            
        if isinstance(channel_indices, list):
            self.selected_channels = channel_indices
        else:
            self.selected_channels = [channel_indices]
        
        print(f"已选择通道: {[self.channels[i] for i in self.selected_channels]}")
        return True


class AdvancedEEGLABVisualizer:
    """对标EEGLAB的高级可视化界面"""
    
    def __init__(self):
        self.processor = AdvancedEEGProcessor()
        self.root = tk.Tk()
        self.root.title("高级EEG数据处理工具 - 对标EEGLAB")
        self.root.geometry("1600x1000")
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建左侧控制面板
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 文件操作
        self.create_file_section(control_frame)
        
        # 电极设置
        self.create_montage_section(control_frame)
        
        # 滤波设置
        self.create_filter_section(control_frame)
        
        # 重参考设置
        self.create_rereference_section(control_frame)
        
        # 分段设置
        self.create_epoching_section(control_frame)
        
        # ICA设置
        self.create_ica_section(control_frame)
        
        # 通道选择
        self.create_channel_section(control_frame)
        
        # 窗口控制
        self.create_window_section(control_frame)
        
        # 右侧绘图区域
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建matplotlib图形
        self.fig = Figure(figsize=(14, 10), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 状态栏
        self.status_label = ttk.Label(self.root, text="就绪", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_file_section(self, parent):
        """创建文件操作区域"""
        file_frame = ttk.LabelFrame(parent, text="文件操作")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="加载EDF文件", command=self.load_file).pack(fill=tk.X, pady=5)
        
        # 降采样设置
        resample_frame = ttk.Frame(file_frame)
        resample_frame.pack(fill=tk.X, pady=2)
        ttk.Label(resample_frame, text="降采样:").pack(side=tk.LEFT)
        self.new_sfreq = tk.StringVar(value="200")
        ttk.Entry(resample_frame, textvariable=self.new_sfreq, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(resample_frame, text="Hz").pack(side=tk.LEFT)
        ttk.Button(resample_frame, text="应用", command=self.apply_resample).pack(side=tk.RIGHT)
    
    def create_montage_section(self, parent):
        """创建电极设置区域"""
        montage_frame = ttk.LabelFrame(parent, text="电极设置")
        montage_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(montage_frame, text="电极标准:").pack(anchor=tk.W)
        self.montage_type = tk.StringVar(value="standard_1020")
        montage_combo = ttk.Combobox(montage_frame, textvariable=self.montage_type, 
                                   values=["standard_1020", "standard_1005", "biosemi64"])
        montage_combo.pack(fill=tk.X, pady=2)
        
        ttk.Button(montage_frame, text="设置电极位置", command=self.set_montage).pack(fill=tk.X, pady=2)
    
    def create_filter_section(self, parent):
        """创建滤波设置区域"""
        filter_frame = ttk.LabelFrame(parent, text="滤波设置")
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 滤波器类型
        ttk.Label(filter_frame, text="滤波器类型:").pack(anchor=tk.W)
        self.filter_type = tk.StringVar(value="bandpass")
        filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_type,
                                  values=["lowpass", "highpass", "bandpass", "bandstop"])
        filter_combo.pack(fill=tk.X, pady=2)
        
        # 频率设置
        freq_frame = ttk.Frame(filter_frame)
        freq_frame.pack(fill=tk.X, pady=2)
        ttk.Label(freq_frame, text="低频:").pack(side=tk.LEFT)
        self.low_freq = tk.StringVar(value="0.5")
        ttk.Entry(freq_frame, textvariable=self.low_freq, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(freq_frame, text="高频:").pack(side=tk.LEFT)
        self.high_freq = tk.StringVar(value="40.0")
        ttk.Entry(freq_frame, textvariable=self.high_freq, width=8).pack(side=tk.LEFT, padx=2)
        
        # 滤波方法
        ttk.Label(filter_frame, text="滤波方法:").pack(anchor=tk.W)
        self.filter_method = tk.StringVar(value="butter")
        method_combo = ttk.Combobox(filter_frame, textvariable=self.filter_method,
                                  values=["butter", "cheby1", "cheby2", "ellip"])
        method_combo.pack(fill=tk.X, pady=2)
        
        ttk.Button(filter_frame, text="应用滤波", command=self.apply_filter).pack(fill=tk.X, pady=2)
    
    def create_rereference_section(self, parent):
        """创建重参考设置区域"""
        ref_frame = ttk.LabelFrame(parent, text="重参考设置")
        ref_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(ref_frame, text="参考类型:").pack(anchor=tk.W)
        self.ref_type = tk.StringVar(value="average")
        ref_combo = ttk.Combobox(ref_frame, textvariable=self.ref_type,
                               values=["average", "mastoid", "nose", "single"])
        ref_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(ref_frame, text="参考电极:").pack(anchor=tk.W)
        self.ref_channels = tk.StringVar()
        ttk.Entry(ref_frame, textvariable=self.ref_channels, width=20).pack(fill=tk.X, pady=2)
        
        ttk.Button(ref_frame, text="应用重参考", command=self.apply_rereference).pack(fill=tk.X, pady=2)
    
    def create_epoching_section(self, parent):
        """创建分段设置区域"""
        epoch_frame = ttk.LabelFrame(parent, text="分段设置")
        epoch_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 时间窗口
        time_frame = ttk.Frame(epoch_frame)
        time_frame.pack(fill=tk.X, pady=2)
        ttk.Label(time_frame, text="开始:").pack(side=tk.LEFT)
        self.tmin = tk.StringVar(value="-0.2")
        ttk.Entry(time_frame, textvariable=self.tmin, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(time_frame, text="结束:").pack(side=tk.LEFT)
        self.tmax = tk.StringVar(value="0.8")
        ttk.Entry(time_frame, textvariable=self.tmax, width=8).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(epoch_frame, text="创建分段", command=self.create_epochs).pack(fill=tk.X, pady=2)
        ttk.Button(epoch_frame, text="基线矫正", command=self.apply_baseline).pack(fill=tk.X, pady=2)
        ttk.Button(epoch_frame, text="剔除坏段", command=self.reject_bad_epochs).pack(fill=tk.X, pady=2)
    
    def create_ica_section(self, parent):
        """创建ICA设置区域"""
        ica_frame = ttk.LabelFrame(parent, text="ICA分析")
        ica_frame.pack(fill=tk.X, pady=(0, 10))
        
        # ICA参数
        ica_param_frame = ttk.Frame(ica_frame)
        ica_param_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ica_param_frame, text="成分数:").pack(side=tk.LEFT)
        self.n_components = tk.StringVar(value="15")
        ttk.Entry(ica_param_frame, textvariable=self.n_components, width=8).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(ica_frame, text="运行ICA", command=self.run_ica).pack(fill=tk.X, pady=2)
        ttk.Button(ica_frame, text="检测伪迹", command=self.detect_artifacts).pack(fill=tk.X, pady=2)
        ttk.Button(ica_frame, text="移除伪迹", command=self.remove_artifacts).pack(fill=tk.X, pady=2)
        ttk.Button(ica_frame, text="绘制成分", command=self.plot_ica_components).pack(fill=tk.X, pady=2)
    
    def create_channel_section(self, parent):
        """创建通道选择区域"""
        channel_frame = ttk.LabelFrame(parent, text="通道选择")
        channel_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.channel_listbox = tk.Listbox(channel_frame, height=8, selectmode=tk.MULTIPLE)
        self.channel_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Button(channel_frame, text="选择通道", command=self.select_channels).pack(fill=tk.X, pady=2)
        ttk.Button(channel_frame, text="全选", command=self.select_all_channels).pack(fill=tk.X, pady=2)
    
    def create_window_section(self, parent):
        """创建窗口控制区域"""
        window_frame = ttk.LabelFrame(parent, text="滑动窗口")
        window_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 窗口时长
        duration_frame = ttk.Frame(window_frame)
        duration_frame.pack(fill=tk.X, pady=2)
        ttk.Label(duration_frame, text="窗口时长:").pack(side=tk.LEFT)
        self.window_duration = tk.StringVar(value="10")
        ttk.Entry(duration_frame, textvariable=self.window_duration, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(duration_frame, text="秒").pack(side=tk.LEFT)
        
        # 时间控制
        time_frame = ttk.Frame(window_frame)
        time_frame.pack(fill=tk.X, pady=2)
        ttk.Button(time_frame, text="<<", command=self.prev_window).pack(side=tk.LEFT, padx=2)
        ttk.Button(time_frame, text="<", command=self.prev_second).pack(side=tk.LEFT, padx=2)
        ttk.Button(time_frame, text=">", command=self.next_second).pack(side=tk.LEFT, padx=2)
        ttk.Button(time_frame, text=">>", command=self.next_window).pack(side=tk.LEFT, padx=2)
        
        # 时间显示
        self.time_label = ttk.Label(window_frame, text="时间: 0.0s")
        self.time_label.pack(pady=2)
    
    def load_file(self):
        """加载EDF文件"""
        file_path = filedialog.askopenfilename(
            title="选择EDF文件",
            filetypes=[("EDF files", "*.edf"), ("All files", "*.*")]
        )
        
        if file_path:
            montage_type = self.montage_type.get()
            if self.processor.load_edf_file(file_path, montage_type):
                self.update_channel_list()
                self.update_status(f"已加载: {os.path.basename(file_path)}")
                self.plot_data()
            else:
                messagebox.showerror("错误", "文件加载失败!")
    
    def set_montage(self):
        """设置电极位置"""
        montage_type = self.montage_type.get()
        if self.processor.set_montage(montage_type):
            self.update_status(f"电极位置设置完成: {montage_type}")
    
    def apply_resample(self):
        """应用降采样"""
        try:
            new_sfreq = float(self.new_sfreq.get())
            if self.processor.resample_data(new_sfreq):
                self.plot_data()
                self.update_status(f"降采样完成: {new_sfreq} Hz")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的采样频率!")
    
    def apply_filter(self):
        """应用滤波"""
        try:
            filter_type = self.filter_type.get()
            low_freq = float(self.low_freq.get()) if self.low_freq.get() else None
            high_freq = float(self.high_freq.get()) if self.high_freq.get() else None
            filter_method = self.filter_method.get()
            
            if self.processor.apply_filter(filter_type, low_freq, high_freq, filter_method):
                self.plot_data()
                self.update_status(f"滤波完成: {filter_type}")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的频率值!")
    
    def apply_rereference(self):
        """应用重参考"""
        ref_type = self.ref_type.get()
        ref_channels = self.ref_channels.get().split(',') if self.ref_channels.get() else None
        
        if self.processor.apply_rereference(ref_type, ref_channels):
            self.plot_data()
            self.update_status(f"重参考完成: {ref_type}")
    
    def create_epochs(self):
        """创建分段"""
        try:
            tmin = float(self.tmin.get())
            tmax = float(self.tmax.get())
            
            # 这里需要用户提供事件信息，暂时使用模拟事件
            events = np.array([[1000, 0, 1], [2000, 0, 1], [3000, 0, 1]])
            event_id = {'event': 1}
            
            if self.processor.create_epochs(events, event_id, tmin, tmax):
                self.update_status("分段创建完成")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的时间值!")
    
    def apply_baseline(self):
        """应用基线矫正"""
        if self.processor.apply_baseline_correction():
            self.update_status("基线矫正完成")
    
    def reject_bad_epochs(self):
        """剔除坏段"""
        if self.processor.reject_bad_epochs():
            self.update_status("坏段剔除完成")
    
    def run_ica(self):
        """运行ICA"""
        try:
            n_components = int(self.n_components.get())
            if self.processor.run_ica(n_components):
                self.update_status("ICA分析完成")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的成分数!")
    
    def detect_artifacts(self):
        """检测伪迹"""
        artifacts = self.processor.detect_artifacts()
        if artifacts:
            self.update_status(f"检测到伪迹: {list(artifacts.keys())}")
        else:
            self.update_status("未检测到明显伪迹")
    
    def remove_artifacts(self):
        """移除伪迹"""
        # 这里需要用户选择要移除的成分
        messagebox.showinfo("提示", "请先检测伪迹，然后手动选择要移除的成分")
    
    def plot_ica_components(self):
        """绘制ICA成分"""
        fig = self.processor.plot_ica_components()
        if fig:
            self.update_status("ICA成分图已显示")
    
    def update_channel_list(self):
        """更新通道列表"""
        self.channel_listbox.delete(0, tk.END)
        if self.processor.channels:
            for i, ch in enumerate(self.processor.channels):
                self.channel_listbox.insert(tk.END, f"{i}: {ch}")
    
    def select_channels(self):
        """选择通道"""
        selected_indices = self.channel_listbox.curselection()
        if selected_indices:
            self.processor.select_channels(list(selected_indices))
            self.plot_data()
    
    def select_all_channels(self):
        """全选通道"""
        self.channel_listbox.select_set(0, tk.END)
        self.processor.select_channels(list(range(len(self.processor.channels))))
        self.plot_data()
    
    def prev_window(self):
        """上一个窗口"""
        if self.processor.processed_data is not None:
            duration = float(self.window_duration.get())
            self.processor.current_window_start = max(0, self.processor.current_window_start - duration)
            self.plot_data()
            self.update_time_label()
    
    def next_window(self):
        """下一个窗口"""
        if self.processor.processed_data is not None:
            duration = float(self.window_duration.get())
            max_time = self.processor.processed_data.shape[1] / self.processor.sfreq - duration
            self.processor.current_window_start = min(max_time, self.processor.current_window_start + duration)
            self.plot_data()
            self.update_time_label()
    
    def prev_second(self):
        """上一秒"""
        if self.processor.processed_data is not None:
            self.processor.current_window_start = max(0, self.processor.current_window_start - 1)
            self.plot_data()
            self.update_time_label()
    
    def next_second(self):
        """下一秒"""
        if self.processor.processed_data is not None:
            duration = float(self.window_duration.get())
            max_time = self.processor.processed_data.shape[1] / self.processor.sfreq - duration
            self.processor.current_window_start = min(max_time, self.processor.current_window_start + 1)
            self.plot_data()
            self.update_time_label()
    
    def update_time_label(self):
        """更新时间标签"""
        if self.processor.processed_data is not None:
            duration = float(self.window_duration.get())
            self.time_label.config(text=f"时间: {self.processor.current_window_start:.1f}s - {self.processor.current_window_start + duration:.1f}s")
    
    def plot_data(self):
        """绘制数据"""
        if self.processor.processed_data is None:
            return
        
        self.fig.clear()
        
        # 获取窗口数据
        duration = float(self.window_duration.get())
        window_data, time_axis = self.processor.get_window_data(duration=duration)
        
        if window_data is None:
            return
        
        # 选择要显示的通道
        if self.processor.selected_channels:
            display_channels = self.processor.selected_channels
        else:
            display_channels = list(range(min(8, len(self.processor.channels))))
        
        # 创建子图
        n_channels = len(display_channels)
        if n_channels == 0:
            return
        
        axes = []
        for i, ch_idx in enumerate(display_channels):
            ax = self.fig.add_subplot(n_channels, 1, i + 1)
            ax.plot(time_axis, window_data[ch_idx, :], linewidth=0.8)
            ax.set_ylabel(f'{self.processor.channels[ch_idx]}\n(μV)')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-200, 200)
            
            if i == n_channels - 1:
                ax.set_xlabel('时间 (秒)')
            else:
                ax.set_xticks([])
            
            axes.append(ax)
        
        self.fig.suptitle(f'EEG数据 - 窗口: {self.processor.current_window_start:.1f}s - {self.processor.current_window_start + duration:.1f}s')
        self.fig.tight_layout()
        self.canvas.draw()
        
        self.update_time_label()
    
    def update_status(self, message):
        """更新状态栏"""
        self.status_label.config(text=message)
    
    def run(self):
        """运行应用"""
        self.root.mainloop()


def main():
    """主函数"""
    print("启动高级EEG数据处理工具 - 对标EEGLAB...")
    app = AdvancedEEGLABVisualizer()
    app.run()


if __name__ == "__main__":
    main() 