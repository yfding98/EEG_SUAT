import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
from pathlib import Path
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


class EpilepsyEEGProcessor:
    """专门针对癫痫EEG数据的处理工具"""
    
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.channels = None
        self.channel_types = None
        self.sfreq = None
        self.original_sfreq = None
        self.current_window_start = 0
        self.window_duration = 10
        self.selected_channels = []
        
        # 数据信息
        self.data_info = {}
        self.processing_history = []
        
        # 癫痫相关参数
        self.seizure_events = []
        self.ictal_periods = []
        self.interictal_periods = []
        
    def load_edf_file(self, file_path):
        """加载EDF文件并分析数据信息"""
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
            
            # 分析数据信息
            self.analyze_data_info(raw)
            
            # 生成处理建议
            self.generate_processing_recommendations()
            
            print(f"文件加载成功!")
            print(f"数据形状: {self.raw_data.shape}")
            print(f"通道数量: {len(self.channels)}")
            print(f"采样频率: {self.sfreq} Hz")
            print(f"时长: {self.raw_data.shape[1] / self.sfreq:.2f} 秒")
            
            return True
            
        except Exception as e:
            print(f"加载文件失败: {e}")
            return False
    
    def analyze_data_info(self, raw):
        """分析EEG数据信息"""
        self.data_info = {
            'duration': raw.times[-1],
            'n_channels': len(raw.ch_names),
            'sfreq': raw.info['sfreq'],
            'ch_names': raw.ch_names,
            'ch_types': raw.get_channel_types(),
            'data_range': (self.raw_data.min(), self.raw_data.max()),
            'data_std': np.std(self.raw_data),
            'has_events': len(raw.annotations) > 0 if hasattr(raw, 'annotations') else False
        }
        
        # 检测通道类型
        eeg_channels = [ch for ch, ch_type in zip(self.channels, self.channel_types) if ch_type == 'eeg']
        eog_channels = [ch for ch, ch_type in zip(self.channels, self.channel_types) if ch_type == 'eog']
        ecg_channels = [ch for ch, ch_type in zip(self.channels, self.channel_types) if ch_type == 'ecg']
        
        self.data_info.update({
            'eeg_channels': eeg_channels,
            'eog_channels': eog_channels,
            'ecg_channels': ecg_channels,
            'n_eeg': len(eeg_channels),
            'n_eog': len(eog_channels),
            'n_ecg': len(ecg_channels)
        })
        
        # 分析数据质量
        self.analyze_data_quality()
        
        print("数据信息分析完成:")
        print(f"  - EEG通道: {len(eeg_channels)}")
        print(f"  - EOG通道: {len(eog_channels)}")
        print(f"  - ECG通道: {len(ecg_channels)}")
        print(f"  - 数据范围: {self.data_info['data_range'][0]:.1f} ~ {self.data_info['data_range'][1]:.1f} μV")
        print(f"  - 数据标准差: {self.data_info['data_std']:.1f} μV")
    
    def analyze_data_quality(self):
        """分析数据质量"""
        # 计算每个通道的统计信息
        channel_stats = {}
        for i, ch in enumerate(self.channels):
            ch_data = self.raw_data[i, :]
            channel_stats[ch] = {
                'mean': np.mean(ch_data),
                'std': np.std(ch_data),
                'min': np.min(ch_data),
                'max': np.max(ch_data),
                'range': np.max(ch_data) - np.min(ch_data),
                'zero_crossings': np.sum(np.diff(np.sign(ch_data)) != 0)
            }
        
        self.data_info['channel_stats'] = channel_stats
        
        # 检测异常通道
        bad_channels = []
        for ch, stats in channel_stats.items():
            # 检测平线通道
            if stats['std'] < 1.0:  # 标准差过小
                bad_channels.append(f"{ch} (平线)")
            # 检测饱和通道
            elif stats['range'] > 1000:  # 范围过大
                bad_channels.append(f"{ch} (饱和)")
            # 检测噪声通道
            elif stats['zero_crossings'] < 10:  # 过零点过少
                bad_channels.append(f"{ch} (噪声)")
        
        self.data_info['bad_channels'] = bad_channels
        
        if bad_channels:
            print(f"检测到异常通道: {bad_channels}")
        else:
            print("未检测到明显异常通道")
    
    def generate_processing_recommendations(self):
        """生成处理建议"""
        recommendations = []
        
        # 基于采样频率的建议
        if self.sfreq > 500:
            recommendations.append("建议降采样到200-250Hz以减少计算负担")
        
        # 基于通道数的建议
        if self.data_info['n_eeg'] > 64:
            recommendations.append("高密度EEG数据，建议使用空间滤波")
        
        # 基于数据质量的建议
        if len(self.data_info['bad_channels']) > 0:
            recommendations.append("检测到异常通道，建议先进行坏通道插值")
        
        # 基于数据范围的建议
        if self.data_info['data_std'] > 100:
            recommendations.append("数据噪声较大，建议加强滤波")
        
        # 癫痫数据特殊建议
        recommendations.extend([
            "癫痫数据建议使用0.5-40Hz带通滤波",
            "建议检测并标记发作期和发作间期",
            "建议使用ICA去除眼动和肌电伪迹",
            "建议进行重参考化处理"
        ])
        
        self.data_info['recommendations'] = recommendations
        
        print("\n处理建议:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    def apply_epilepsy_filtering(self):
        """应用癫痫数据专用滤波"""
        if self.processed_data is None:
            return False
        
        try:
            # 1. 带通滤波 0.5-40Hz (适合癫痫分析)
            self.apply_bandpass_filter(0.5, 40.0)
            self.processing_history.append("带通滤波: 0.5-40Hz")
            
            # 2. 陷波滤波去除工频干扰
            self.apply_notch_filter(50.0)
            self.processing_history.append("陷波滤波: 50Hz")
            
            # 3. 如果采样频率过高，进行降采样
            if self.sfreq > 250:
                target_sfreq = 200
                self.resample_data(target_sfreq)
                self.processing_history.append(f"降采样: {self.sfreq}Hz -> {target_sfreq}Hz")
            
            print("癫痫数据专用滤波完成")
            return True
            
        except Exception as e:
            print(f"滤波失败: {e}")
            return False
    
    def apply_bandpass_filter(self, low_freq=0.5, high_freq=40.0):
        """带通滤波"""
        try:
            nyquist = self.sfreq / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            b, a = butter(4, [low, high], btype='band')
            
            for i in range(self.processed_data.shape[0]):
                self.processed_data[i, :] = filtfilt(b, a, self.processed_data[i, :])
            
            print(f"带通滤波完成: {low_freq}-{high_freq} Hz")
            return True
            
        except Exception as e:
            print(f"带通滤波失败: {e}")
            return False
    
    def apply_notch_filter(self, notch_freq=50.0):
        """陷波滤波"""
        try:
            b, a = iirnotch(notch_freq, 30, self.sfreq)
            
            for i in range(self.processed_data.shape[0]):
                self.processed_data[i, :] = filtfilt(b, a, self.processed_data[i, :])
            
            print(f"陷波滤波完成: {notch_freq} Hz")
            return True
            
        except Exception as e:
            print(f"陷波滤波失败: {e}")
            return False
    
    def resample_data(self, new_sfreq):
        """降采样"""
        try:
            if new_sfreq >= self.sfreq:
                print("新采样频率必须小于当前采样频率")
                return False
            
            decimation_factor = int(self.sfreq / new_sfreq)
            nyquist = self.sfreq / 2
            cutoff = new_sfreq / 2
            b, a = butter(4, cutoff / nyquist, btype='low')
            
            filtered_data = np.zeros_like(self.processed_data)
            for i in range(self.processed_data.shape[0]):
                filtered_data[i, :] = filtfilt(b, a, self.processed_data[i, :])
            
            self.processed_data = filtered_data[:, ::decimation_factor]
            self.sfreq = new_sfreq
            
            print(f"降采样完成: {self.original_sfreq} Hz -> {self.sfreq} Hz")
            return True
            
        except Exception as e:
            print(f"降采样失败: {e}")
            return False
    
    def apply_rereference(self, ref_type='average'):
        """重参考化"""
        try:
            if ref_type == 'average':
                avg_ref = np.mean(self.processed_data, axis=0, keepdims=True)
                self.processed_data = self.processed_data - avg_ref
                print("平均参考完成")
                self.processing_history.append("重参考: 平均参考")
            
            return True
            
        except Exception as e:
            print(f"重参考化失败: {e}")
            return False
    
    def get_channel_types_for_mne(self):
        """根据通道名称确定通道类型"""
        ch_types = []
        for ch in self.channels:
            ch_upper = ch.upper()
            if ch_upper in ['ECG']:
                ch_types.append('ecg')
            elif ch_upper in ['EMG1', 'EMG2', 'EMG']:
                ch_types.append('emg')
            elif ch_upper in ['EOG', 'VEOG', 'HEOG', 'SPH-R', 'SPH-L']:  # 添加眼动电极
                ch_types.append('eog')
            elif ch_upper in ['DC', 'OSAT', 'PR']:
                ch_types.append('misc')
            elif ch.isdigit():  # 数字通道
                ch_types.append('misc')
            else:
                ch_types.append('eeg')  # 默认EEG通道
        
        return ch_types
    
    def set_montage_10_20(self):
        """设置10-20系统电极位置"""
        try:
            # 创建MNE info对象
            ch_types = self.get_channel_types_for_mne()
            info = mne.create_info(
                ch_names=self.channels,
                sfreq=self.sfreq,
                ch_types=ch_types
            )
            
            # 设置10-20系统montage
            montage = mne.channels.make_standard_montage('standard_1020')
            
            # 为特殊眼动电极添加位置信息
            # Sph-R 和 Sph-L 是眼动监测电极，通常位于眼外眦
            if 'Sph-R' in self.channels or 'Sph-L' in self.channels:
                # 创建自定义montage
                from mne.channels import make_dig_montage
                
                # 获取标准10-20位置
                standard_pos = montage.get_positions()
                ch_pos = standard_pos['ch_pos'].copy()
                
                # 添加眼动电极位置（眼外眦位置）
                if 'Sph-R' in self.channels:
                    ch_pos['Sph-R'] = np.array([0.08, -0.05, 0.02])  # 右眼外眦
                if 'Sph-L' in self.channels:
                    ch_pos['Sph-L'] = np.array([-0.08, -0.05, 0.02])  # 左眼外眦
                
                # 创建自定义montage
                custom_montage = make_dig_montage(ch_pos=ch_pos)
                info.set_montage(custom_montage)
            else:
                # 使用标准10-20 montage
                info.set_montage(montage, on_missing='ignore')
            
            # 保存montage信息
            self.montage_info = info
            self.channel_positions = {}
            
            # 获取电极位置信息 - 修复访问方式
            for i, ch_info in enumerate(info['chs']):
                ch_name = ch_info['ch_name']
                if  'loc' in ch_info and ch_info['loc'] is not None:
                    # loc数组前3个元素是x, y, z坐标
                    loc = ch_info['loc']
                    if not np.isnan(loc[0]) and not np.isnan(loc[1]) and not np.isnan(loc[2]):
                        pos = loc[:3]  # x, y, z坐标
                        self.channel_positions[ch_name] = pos
            
            print("10-20系统电极位置设置完成")
            print(f"已设置 {len(self.channel_positions)} 个电极位置")
            
            # 显示电极位置信息
            eeg_channels = [ch for ch in self.channels if ch in self.channel_positions]
            print(f"EEG电极: {eeg_channels}")
            
            return True
            
        except Exception as e:
            print(f"电极位置设置失败: {e}")
            return False
    
    def plot_electrode_positions(self):
        """绘制电极位置图"""
        try:
            if not hasattr(self, 'montage_info') or self.montage_info is None:
                print("请先设置电极位置")
                return None
            
            # 创建电极位置图
            fig = self.montage_info.plot_sensors(show_names=True, show=False)
            fig.suptitle('10-20系统电极位置图', fontsize=14)
            
            return fig
            
        except Exception as e:
            print(f"绘制电极位置图失败: {e}")
            return None
    
    def get_electrode_info(self):
        """获取电极信息摘要"""
        if not hasattr(self, 'channel_positions'):
            return "未设置电极位置"
        
        info_text = "电极信息:\n"
        info_text += f"总电极数: {len(self.channels)}\n"
        info_text += f"已定位电极: {len(self.channel_positions)}\n\n"
        
        # 按类型分组显示
        eeg_channels = []
        eog_channels = []
        other_channels = []
        
        for ch in self.channels:
            if ch in self.channel_positions:
                if ch.upper() in ['SPH-R', 'SPH-L', 'EOG', 'VEOG', 'HEOG']:
                    eog_channels.append(ch)
                elif ch.upper() in ['ECG', 'EMG1', 'EMG2', 'EMG']:
                    other_channels.append(ch)
                else:
                    eeg_channels.append(ch)
        
        info_text += f"EEG电极 ({len(eeg_channels)}):\n"
        for ch in eeg_channels:
            info_text += f"  {ch}\n"
        
        if eog_channels:
            info_text += f"\n眼动电极 ({len(eog_channels)}):\n"
            for ch in eog_channels:
                info_text += f"  {ch}\n"
        
        if other_channels:
            info_text += f"\n其他电极 ({len(other_channels)}):\n"
            for ch in other_channels:
                info_text += f"  {ch}\n"
        
        return info_text
    
    def run_ica_for_epilepsy(self):
        """运行ICA分析去除伪迹"""
        try:
            # 根据通道名称确定通道类型
            ch_types = self.get_channel_types_for_mne()
            
            # 创建MNE info对象
            info = mne.create_info(
                ch_names=self.channels,
                sfreq=self.sfreq,
                ch_types=ch_types
            )
            
            raw = mne.io.RawArray(self.processed_data, info)
            
            # 运行ICA
            n_components = min(15, len(self.channels) - 1)
            ica = mne.preprocessing.ICA(
                n_components=n_components,
                method='fastica',
                random_state=97
            )
            ica.fit(raw)
            
            # 检测伪迹
            exclude_components = []
            
            # 检测心电伪迹
            try:
                ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
                if len(ecg_indices) > 0:
                    exclude_components.extend(ecg_indices[:1])
                    print(f"检测到心电伪迹成分: {ecg_indices[:1]}")
            except Exception as e:
                print(f"心电伪迹检测失败: {e}")
            
            # 检测肌电伪迹
            try:
                muscle_indices, muscle_scores = ica.find_bads_muscle(raw)
                if len(muscle_indices) > 0:
                    exclude_components.extend(muscle_indices[:2])
                    print(f"检测到肌电伪迹成分: {muscle_indices[:2]}")
            except Exception as e:
                print(f"肌电伪迹检测失败: {e}")
            
            # 检测工频干扰
            try:
                line_noise_indices, line_noise_scores = ica.find_bads_line_noise(raw)
                if len(line_noise_indices) > 0:
                    exclude_components.extend(line_noise_indices[:1])
                    print(f"检测到工频干扰成分: {line_noise_indices[:1]}")
            except Exception as e:
                print(f"工频干扰检测失败: {e}")
            
            # 如果没有检测到伪迹，使用基于统计的方法
            if len(exclude_components) == 0:
                print("使用统计方法检测异常成分...")
                # 计算每个成分的峰度和偏度
                components = ica.get_components()
                scores = []
                for i in range(components.shape[1]):
                    comp_data = components[:, i]
                    # 计算峰度（衡量分布的尖锐程度）
                    kurtosis = np.mean((comp_data - np.mean(comp_data))**4) / (np.std(comp_data)**4)
                    # 计算偏度（衡量分布的对称性）
                    skewness = np.mean((comp_data - np.mean(comp_data))**3) / (np.std(comp_data)**3)
                    # 综合评分
                    score = abs(kurtosis - 3) + abs(skewness)  # 正态分布的峰度为3，偏度为0
                    scores.append(score)
                
                # 选择评分最高的前2个成分作为可能的伪迹
                if len(scores) > 0:
                    sorted_indices = np.argsort(scores)[::-1]  # 降序排列
                    exclude_components = sorted_indices[:2].tolist()
                    print(f"基于统计方法检测到异常成分: {exclude_components}")
            
            # 应用ICA
            if exclude_components:
                ica.exclude = exclude_components
                raw_clean = ica.apply(raw)
                self.processed_data = raw_clean.get_data()
                print(f"ICA伪迹去除完成，排除成分: {exclude_components}")
                self.processing_history.append(f"ICA伪迹去除: 排除{len(exclude_components)}个成分")
            else:
                print("未检测到明显伪迹，跳过ICA处理")
            
            return True
            
        except Exception as e:
            print(f"ICA分析失败: {e}")
            return False
    
    def detect_seizure_events(self, threshold=3.0):
        """检测癫痫发作事件"""
        try:
            # 计算每个通道的功率
            window_size = int(2 * self.sfreq)  # 2秒窗口
            step_size = int(0.5 * self.sfreq)  # 0.5秒步长
            
            seizure_scores = []
            time_points = []
            
            for start in range(0, self.processed_data.shape[1] - window_size, step_size):
                end = start + window_size
                window_data = self.processed_data[:, start:end]
                
                # 计算功率
                power = np.mean(window_data ** 2, axis=1)
                total_power = np.sum(power)
                
                seizure_scores.append(total_power)
                time_points.append(start / self.sfreq)
            
            # 检测异常功率
            mean_power = np.mean(seizure_scores)
            std_power = np.std(seizure_scores)
            threshold_value = mean_power + threshold * std_power
            
            seizure_indices = np.where(np.array(seizure_scores) > threshold_value)[0]
            
            if len(seizure_indices) > 0:
                self.seizure_events = [(time_points[i], seizure_scores[i]) for i in seizure_indices]
                print(f"检测到 {len(seizure_indices)} 个可能的癫痫发作事件")
                for i, (time, score) in enumerate(self.seizure_events):
                    print(f"  事件 {i+1}: 时间 {time:.1f}s, 功率 {score:.1f}")
            else:
                print("未检测到明显的癫痫发作事件")
            
            return True
            
        except Exception as e:
            print(f"癫痫事件检测失败: {e}")
            return False
    
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
    
    def get_processing_summary(self):
        """获取处理摘要"""
        summary = {
            'data_info': self.data_info,
            'processing_history': self.processing_history,
            'seizure_events': self.seizure_events,
            'current_sfreq': self.sfreq,
            'data_shape': self.processed_data.shape if self.processed_data is not None else None
        }
        return summary

    def get_electrode_positions_info(self):
        """获取详细的电极位置信息"""
        if not hasattr(self, 'montage_info') or self.montage_info is None:
            return "未设置电极位置"
        
        info = self.montage_info
        positions_info = {}
        
        # 从info['chs']获取电极位置信息
        for ch_info in info['chs']:
            ch_name = ch_info['ch_name']
            ch_kind = ch_info['kind']
            
            # 检查是否有有效的位置信息 - 修复访问方式
            if 'loc' in ch_info and ch_info['loc'] is not None:
                loc = ch_info['loc']
                if not np.isnan(loc[0]) and not np.isnan(loc[1]) and not np.isnan(loc[2]):
                    # 根据通道类型分类
                    if ch_kind == 2:  # FIFFV_EEG_CH
                        positions_info[ch_name] = {
                            'type': 'EEG',
                            'position': loc[:3],  # x, y, z坐标
                            'coordinate_system': 'head'
                        }
                    elif ch_kind == 202:  # FIFFV_EOG_CH
                        positions_info[ch_name] = {
                            'type': 'EOG',
                            'position': loc[:3],
                            'coordinate_system': 'head'
                        }
                    elif ch_kind == 402:  # FIFFV_ECG_CH
                        positions_info[ch_name] = {
                            'type': 'ECG',
                            'position': loc[:3],
                            'coordinate_system': 'head'
                        }
                    elif ch_kind == 302:  # FIFFV_EMG_CH
                        positions_info[ch_name] = {
                            'type': 'EMG',
                            'position': loc[:3],
                            'coordinate_system': 'head'
                        }
        
        # 获取数字化点信息（解剖标志点等）
        if hasattr(info, 'dig') and info.dig is not None:
            for dig_point in info.dig:
                if dig_point['kind'] == mne.channels.constants.FIFF.FIFFV_POINT_CARDINAL:
                    # 解剖标志点
                    if dig_point['ident'] == mne.channels.constants.FIFF.FIFFV_POINT_LPA:
                        positions_info['LPA'] = {
                            'type': 'Cardinal',
                            'position': dig_point['r'],
                            'coordinate_system': 'head'
                        }
                    elif dig_point['ident'] == mne.channels.constants.FIFF.FIFFV_POINT_RPA:
                        positions_info['RPA'] = {
                            'type': 'Cardinal', 
                            'position': dig_point['r'],
                            'coordinate_system': 'head'
                        }
                    elif dig_point['ident'] == mne.channels.constants.FIFF.FIFFV_POINT_NASION:
                        positions_info['Nasion'] = {
                            'type': 'Cardinal',
                            'position': dig_point['r'],
                            'coordinate_system': 'head'
                        }
        
        return positions_info
    
    def analyze_electrode_coverage(self):
        """分析电极覆盖范围"""
        positions_info = self.get_electrode_positions_info()
        
        if not positions_info:
            return "无电极位置信息"
        
        # 提取EEG电极位置
        eeg_positions = []
        eog_positions = []
        
        for ch_name, info in positions_info.items():
            if info['type'] == 'EEG':
                eeg_positions.append(info['position'])
            elif info['type'] == 'EOG':
                eog_positions.append(info['position'])
        
        if not eeg_positions:
            return "无EEG电极位置信息"
        
        eeg_positions = np.array(eeg_positions)
        
        # 计算覆盖范围
        coverage_info = {
            'total_eeg_channels': len(eeg_positions),
            'frontal_channels': 0,
            'central_channels': 0,
            'parietal_channels': 0,
            'occipital_channels': 0,
            'temporal_channels': 0,
            'coverage_quality': 'Unknown'
        }
        
        # 根据Y坐标（前后方向）和通道名称分类电极
        for ch_name, info in positions_info.items():
            if info['type'] == 'EEG':
                pos = info['position']
                y_coord = pos[1]  # Y坐标，正值表示前方
                
                # 根据通道名称和位置分类
                if 'F' in ch_name:
                    coverage_info['frontal_channels'] += 1
                elif 'C' in ch_name:
                    coverage_info['central_channels'] += 1
                elif 'P' in ch_name:
                    coverage_info['parietal_channels'] += 1
                elif 'O' in ch_name:
                    coverage_info['occipital_channels'] += 1
                elif 'T' in ch_name:
                    coverage_info['temporal_channels'] += 1
        
        # 评估覆盖质量
        total_regions = sum([1 for v in coverage_info.values() 
                           if isinstance(v, int) and v > 0])
        if total_regions >= 4:
            coverage_info['coverage_quality'] = 'Good'
        elif total_regions >= 3:
            coverage_info['coverage_quality'] = 'Fair'
        else:
            coverage_info['coverage_quality'] = 'Poor'
        
        return coverage_info
    
    def plot_topographic_map(self, data_timepoint=0, time_window=1.0):
        """绘制地形图"""
        try:
            if not hasattr(self, 'montage_info') or self.montage_info is None:
                print("请先设置电极位置")
                return None
            
            if self.processed_data is None:
                print("无数据可显示")
                return None
            
            # 选择时间点
            start_sample = int(data_timepoint * self.sfreq)
            end_sample = int((data_timepoint + time_window) * self.sfreq)
            end_sample = min(end_sample, self.processed_data.shape[1])
            
            # 获取有效EEG通道的信息
            valid_eeg_channels = []
            valid_eeg_data = []
            valid_eeg_positions = []
            
            # 遍历所有通道，找到有有效位置的EEG通道
            for i, ch in enumerate(self.channels):
                ch_info = self.montage_info['chs'][i]
                if ch_info['kind'] == 2:  # FIFFV_EEG_CH
                    # 检查是否有有效位置
                    if 'loc' in ch_info and ch_info['loc'] is not None:
                        loc = ch_info['loc']
                        if not np.isnan(loc[0]) and not np.isnan(loc[1]) and not np.isnan(loc[2]):
                            valid_eeg_channels.append(ch)
                            valid_eeg_data.append(np.mean(self.processed_data[i, start_sample:end_sample]))
                            valid_eeg_positions.append(loc[:3])
            
            if len(valid_eeg_data) < 4:
                print(f"EEG通道数量不足，无法绘制地形图 (当前: {len(valid_eeg_data)}个)")
                return None
            
            # 创建新的info对象，只包含有效的EEG通道
            info_eeg = mne.create_info(
                ch_names=valid_eeg_channels,
                sfreq=self.sfreq,
                ch_types=['eeg'] * len(valid_eeg_channels)
            )
            
            # 设置电极位置
            from mne.channels import make_dig_montage
            ch_pos = {ch: pos for ch, pos in zip(valid_eeg_channels, valid_eeg_positions)}
            montage = make_dig_montage(ch_pos=ch_pos)
            info_eeg.set_montage(montage)
            
            # 创建平均数据 - 修复数据形状
            evoked_data = np.array(valid_eeg_data).reshape(-1, 1)  # 改为 (n_channels, 1)
            evoked = mne.EvokedArray(evoked_data, info_eeg, tmin=0)
            
            # 绘制地形图 - 修复布局问题
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 2D地形图 - 不指定axes参数，让MNE自动创建
            evoked.plot_topomap(times=0, show=False)
            axes[0].set_title(f'EEG地形图 (时间: {data_timepoint:.1f}s)')
            
            # 3D地形图 - 不指定axes参数，让MNE自动创建
            evoked.plot_topomap(times=0, show=False)
            axes[1].set_title(f'EEG地形图 3D (时间: {data_timepoint:.1f}s)')
            
            # 修复布局问题 - 使用subplots_adjust代替tight_layout
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3)
            return fig
            
        except Exception as e:
            print(f"绘制地形图失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_electrode_distances(self):
        """计算电极间距离"""
        positions_info = self.get_electrode_positions_info()
        
        if not positions_info:
            return {}
        
        # 提取EEG电极位置
        eeg_positions = {}
        for ch_name, info in positions_info.items():
            if info['type'] == 'EEG':
                eeg_positions[ch_name] = info['position']
        
        if len(eeg_positions) < 2:
            return {}
        
        # 计算距离矩阵
        distances = {}
        ch_names = list(eeg_positions.keys())
        
        for i, ch1 in enumerate(ch_names):
            for j, ch2 in enumerate(ch_names[i+1:], i+1):
                pos1 = eeg_positions[ch1]
                pos2 = eeg_positions[ch2]
                distance = np.linalg.norm(pos1 - pos2)
                distances[f"{ch1}-{ch2}"] = distance
        
        return distances


class EpilepsyEEGVisualizer:
    """癫痫EEG数据可视化界面"""
    
    def __init__(self):
        self.processor = EpilepsyEEGProcessor()
        self.root = tk.Tk()
        self.root.title("癫痫EEG数据处理工具")
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
        
        # 数据信息显示
        self.create_info_section(control_frame)
        
        # 处理建议
        self.create_recommendations_section(control_frame)
        
        # 处理流程
        self.create_processing_section(control_frame)
        
        # 癫痫检测
        self.create_seizure_detection_section(control_frame)
        
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
        ttk.Button(file_frame, text="显示处理摘要", command=self.show_summary).pack(fill=tk.X, pady=2)
    
    def create_info_section(self, parent):
        """创建数据信息显示区域"""
        info_frame = ttk.LabelFrame(parent, text="数据信息")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_text = tk.Text(info_frame, height=8, width=30)
        self.info_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 添加电极信息按钮
        button_frame = ttk.Frame(info_frame)
        button_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(button_frame, text="电极信息", command=self.show_electrode_info).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="电极位置", command=self.plot_electrode_positions).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="地形图", command=self.plot_topographic_map).pack(side=tk.LEFT, padx=2)
    
    def create_recommendations_section(self, parent):
        """创建处理建议区域"""
        rec_frame = ttk.LabelFrame(parent, text="处理建议")
        rec_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.rec_text = tk.Text(rec_frame, height=6, width=30)
        self.rec_text.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_processing_section(self, parent):
        """创建处理流程区域"""
        proc_frame = ttk.LabelFrame(parent, text="处理流程")
        proc_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(proc_frame, text="0. 设置电极位置", command=self.set_montage).pack(fill=tk.X, pady=2)
        ttk.Button(proc_frame, text="1. 癫痫专用滤波", command=self.apply_epilepsy_filtering).pack(fill=tk.X, pady=2)
        ttk.Button(proc_frame, text="2. 重参考化", command=self.apply_rereference).pack(fill=tk.X, pady=2)
        ttk.Button(proc_frame, text="3. ICA伪迹去除", command=self.run_ica).pack(fill=tk.X, pady=2)
        ttk.Button(proc_frame, text="4. 检测癫痫事件", command=self.detect_seizures).pack(fill=tk.X, pady=2)
    
    def create_seizure_detection_section(self, parent):
        """创建癫痫检测区域"""
        seizure_frame = ttk.LabelFrame(parent, text="癫痫检测")
        seizure_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(seizure_frame, text="检测阈值:").pack(anchor=tk.W)
        self.threshold = tk.StringVar(value="3.0")
        ttk.Entry(seizure_frame, textvariable=self.threshold, width=10).pack(fill=tk.X, pady=2)
        
        ttk.Button(seizure_frame, text="检测癫痫事件", command=self.detect_seizures).pack(fill=tk.X, pady=2)
        ttk.Button(seizure_frame, text="显示检测结果", command=self.show_seizure_results).pack(fill=tk.X, pady=2)
    
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
            if self.processor.load_edf_file(file_path):
                self.update_info_display()
                self.update_recommendations_display()
                self.update_channel_list()
                self.update_status(f"已加载: {os.path.basename(file_path)}")
                self.plot_data()
            else:
                messagebox.showerror("错误", "文件加载失败!")
    
    def update_info_display(self):
        """更新数据信息显示"""
        self.info_text.delete(1.0, tk.END)
        if self.processor.data_info:
            info = self.processor.data_info
            self.info_text.insert(tk.END, f"数据时长: {info['duration']:.1f}秒\n")
            self.info_text.insert(tk.END, f"采样频率: {info['sfreq']}Hz\n")
            self.info_text.insert(tk.END, f"EEG通道: {info['n_eeg']}\n")
            self.info_text.insert(tk.END, f"EOG通道: {info['n_eog']}\n")
            self.info_text.insert(tk.END, f"ECG通道: {info['n_ecg']}\n")
            self.info_text.insert(tk.END, f"数据范围: {info['data_range'][0]:.1f}~{info['data_range'][1]:.1f}μV\n")
            self.info_text.insert(tk.END, f"数据标准差: {info['data_std']:.1f}μV\n")
            if info['bad_channels']:
                self.info_text.insert(tk.END, f"异常通道: {len(info['bad_channels'])}\n")
    
    def update_recommendations_display(self):
        """更新处理建议显示"""
        self.rec_text.delete(1.0, tk.END)
        if self.processor.data_info and 'recommendations' in self.processor.data_info:
            for i, rec in enumerate(self.processor.data_info['recommendations'], 1):
                self.rec_text.insert(tk.END, f"{i}. {rec}\n")
    
    def apply_epilepsy_filtering(self):
        """应用癫痫专用滤波"""
        if self.processor.apply_epilepsy_filtering():
            self.plot_data()
            self.update_status("癫痫专用滤波完成")
    
    def apply_rereference(self):
        """应用重参考"""
        if self.processor.apply_rereference():
            self.plot_data()
            self.update_status("重参考化完成")
    
    def run_ica(self):
        """运行ICA"""
        if self.processor.run_ica_for_epilepsy():
            self.plot_data()
            self.update_status("ICA伪迹去除完成")
    
    def detect_seizures(self):
        """检测癫痫事件"""
        try:
            threshold = float(self.threshold.get())
            if self.processor.detect_seizure_events(threshold):
                self.update_status("癫痫事件检测完成")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的阈值!")
    
    def show_seizure_results(self):
        """显示癫痫检测结果"""
        if self.processor.seizure_events:
            result_text = "检测到的癫痫事件:\n"
            for i, (time, score) in enumerate(self.processor.seizure_events):
                result_text += f"事件 {i+1}: 时间 {time:.1f}s, 功率 {score:.1f}\n"
            messagebox.showinfo("癫痫检测结果", result_text)
        else:
            messagebox.showinfo("癫痫检测结果", "未检测到癫痫事件")
    
    def show_summary(self):
        """显示处理摘要"""
        summary = self.processor.get_processing_summary()
        summary_text = f"处理摘要:\n\n"
        summary_text += f"数据信息:\n"
        summary_text += f"  - 时长: {summary['data_info']['duration']:.1f}秒\n"
        summary_text += f"  - 通道数: {summary['data_info']['n_channels']}\n"
        summary_text += f"  - 采样频率: {summary['current_sfreq']}Hz\n\n"
        summary_text += f"处理历史:\n"
        for i, step in enumerate(summary['processing_history'], 1):
            summary_text += f"  {i}. {step}\n"
        summary_text += f"\n癫痫事件: {len(summary['seizure_events'])}个"
        
        messagebox.showinfo("处理摘要", summary_text)
    
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
        
        self.fig.suptitle(f'癫痫EEG数据 - 窗口: {self.processor.current_window_start:.1f}s - {self.processor.current_window_start + duration:.1f}s')
        self.fig.tight_layout()
        self.canvas.draw()
        
        self.update_time_label()
    
    def update_status(self, message):
        """更新状态栏"""
        self.status_label.config(text=message)
    
    def run(self):
        """运行应用"""
        self.root.mainloop()

    def set_montage(self):
        """设置电极位置"""
        if self.processor.set_montage_10_20():
            self.update_status("10-20系统电极位置设置完成")
            self.update_info_display()
    
    def show_electrode_info(self):
        """显示电极信息"""
        electrode_info = self.processor.get_electrode_info()
        coverage_info = self.processor.analyze_electrode_coverage()
        
        info_text = electrode_info + "\n\n电极覆盖分析:\n"
        if isinstance(coverage_info, dict):
            info_text += f"覆盖质量: {coverage_info['coverage_quality']}\n"
            info_text += f"前部电极: {coverage_info['frontal_channels']}\n"
            info_text += f"中央电极: {coverage_info['central_channels']}\n"
            info_text += f"后部电极: {coverage_info['parietal_channels']}\n"
            info_text += f"枕部电极: {coverage_info['occipital_channels']}\n"
            info_text += f"颞部电极: {coverage_info['temporal_channels']}\n"
        
        messagebox.showinfo("电极信息", info_text)
    
    def plot_electrode_positions(self):
        """绘制电极位置图"""
        fig = self.processor.plot_electrode_positions()
        if fig:
            self.update_status("电极位置图已显示")
        else:
            messagebox.showerror("错误", "无法绘制电极位置图，请先设置电极位置")

    def plot_topographic_map(self):
        """绘制地形图"""
        fig = self.processor.plot_topographic_map()
        if fig:
            self.update_status("地形图已显示")
        else:
            messagebox.showerror("错误", "无法绘制地形图")


def main():
    """主函数"""
    print("启动癫痫EEG数据处理工具...")
    app = EpilepsyEEGVisualizer()
    app.run()


if __name__ == "__main__":
    main() 