#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interactive_patient_merge.py

交互式界面，用于合并每个患者下不同试例的相同标签片段。

功能：
1. 读取患者目录下所有 _channel_merge_statistics.csv 文件
2. 展示所有通道类型和时间长度
3. 通过界面勾选要合并的片段
4. 支持修改合并后的通道组合标签
5. 执行合并并保存结果
6. 输出合并统计 CSV

使用方法:
    python interactive_patient_merge.py --root_dir "E:\DataSet\EEG\EEG dataset_SUAT_processed"
"""

import os
import csv
import argparse
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import mne
from collections import defaultdict
from pathlib import Path


class PatientMergeGUI:
    def __init__(self, root, patient_dirs):
        self.root = root
        self.root.title("患者数据合并工具")
        self.root.geometry("1200x800")
        
        self.patient_dirs = patient_dirs
        self.current_patient_idx = 0
        self.merge_history = []  # 记录所有合并历史
        self.item_to_data_idx = {}  # 映射 tree item id 到 data 索引
        
        # 创建主界面
        self.create_widgets()
        
        # 加载第一个患者
        if self.patient_dirs:
            self.load_patient(0)
    
    def create_widgets(self):
        # 顶部：患者信息
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.patient_label = ttk.Label(top_frame, text="", font=('Arial', 12, 'bold'))
        self.patient_label.pack(side=tk.LEFT)
        
        self.patient_progress = ttk.Label(top_frame, text="")
        self.patient_progress.pack(side=tk.RIGHT)
        
        # 中部：数据表格
        middle_frame = ttk.Frame(self.root)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 左侧：文件列表和数据
        left_frame = ttk.Frame(middle_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(left_frame, text="数据片段列表:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        # 创建 Treeview
        columns = ('选择', '文件名', '通道组合', '片段数', '时长(秒)', '时长(分)', '数据文件')
        self.tree = ttk.Treeview(left_frame, columns=columns, show='headings', height=20)
        
        # 设置列宽
        self.tree.column('选择', width=50)
        self.tree.column('文件名', width=150)
        self.tree.column('通道组合', width=200)
        self.tree.column('片段数', width=80)
        self.tree.column('时长(秒)', width=100)
        self.tree.column('时长(分)', width=100)
        self.tree.column('数据文件', width=200)
        
        # 设置列标题
        for col in columns:
            self.tree.heading(col, text=col)
        
        # 滚动条
        scrollbar_y = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar_x = ttk.Scrollbar(left_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 绑定双击事件切换选择状态
        self.tree.bind('<Double-1>', self.toggle_selection)
        
        # 右侧：统计和操作
        right_frame = ttk.Frame(middle_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # 统计信息
        stats_label = ttk.Label(right_frame, text="统计信息", font=('Arial', 10, 'bold'))
        stats_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.stats_text = tk.Text(right_frame, height=15, width=35)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # 合并设置
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        ttk.Label(right_frame, text="合并设置", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        ttk.Label(right_frame, text="合并后通道标签:").pack(anchor=tk.W, pady=(5, 0))
        self.merge_label_var = tk.StringVar()
        self.merge_label_entry = ttk.Entry(right_frame, textvariable=self.merge_label_var, width=30)
        self.merge_label_entry.pack(fill=tk.X, pady=(0, 10))
        
        # 自动填充建议按钮
        suggest_frame = ttk.Frame(right_frame)
        suggest_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(suggest_frame, text="使用最长片段标签", command=self.suggest_longest_label).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(suggest_frame, text="自定义", command=self.custom_label).pack(side=tk.LEFT)
        
        # 操作按钮
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill=tk.X)
        
        self.merge_btn = ttk.Button(btn_frame, text="全选", command=self.select_all)
        self.merge_btn.pack(fill=tk.X, pady=2)
        
        self.clear_btn = ttk.Button(btn_frame, text="清除选择", command=self.clear_selection)
        self.clear_btn.pack(fill=tk.X, pady=2)
        
        self.merge_btn = ttk.Button(btn_frame, text="合并选中项", command=self.merge_selected)
        self.merge_btn.pack(fill=tk.X, pady=2)
        
        # 底部：导航按钮
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.prev_btn = ttk.Button(bottom_frame, text="← 上一个患者", command=self.prev_patient)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = ttk.Button(bottom_frame, text="下一个患者 →", command=self.next_patient)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(bottom_frame, text="导出当前患者统计", command=self.export_current_stats).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="保存并退出", command=self.save_and_exit).pack(side=tk.RIGHT, padx=5)
    
    def load_patient(self, index):
        """加载指定患者的数据"""
        if index < 0 or index >= len(self.patient_dirs):
            return
        
        self.current_patient_idx = index
        patient_dir = self.patient_dirs[index]
        patient_name = os.path.basename(patient_dir)
        
        # 更新标题
        self.patient_label.config(text=f"患者: {patient_name}")
        self.patient_progress.config(text=f"第 {index + 1}/{len(self.patient_dirs)} 位患者")
        
        # 清空树形视图
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # 清空映射
        self.item_to_data_idx = {}
        
        # 查找所有 _channel_merge_statistics.csv 文件
        self.current_data = []
        for root, dirs, files in os.walk(patient_dir):
            for file in files:
                if file.endswith('_channel_merge_statistics.csv'):
                    csv_path = os.path.join(root, file)
                    self.load_statistics_csv(csv_path)
        
        # 更新统计信息
        self.update_statistics()
        
        # 更新按钮状态
        self.prev_btn.config(state=tk.NORMAL if index > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if index < len(self.patient_dirs) - 1 else tk.DISABLED)
    
    def load_statistics_csv(self, csv_path):
        """加载单个统计 CSV 文件"""
        try:
            # 读取 CSV
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # 获取文件名
            file_name = os.path.basename(csv_path).replace('_channel_merge_statistics.csv', '')
            
            # 获取数据文件所在目录
            data_dir = os.path.dirname(csv_path)
            
            # 添加到树形视图
            for _, row in df.iterrows():
                channel_combo = row['通道组合']
                count = row['片段数量']
                duration_s = row['总时长(秒)']
                duration_m = row['总时长(分钟)']
                output_file = row['输出文件名']
                
                # 构建数据文件完整路径
                data_file_path = os.path.join(data_dir, output_file)
                
                item_data = {
                    'csv_file': csv_path,
                    'file_name': file_name,
                    'channel_combo': channel_combo,
                    'count': count,
                    'duration_s': duration_s,
                    'duration_m': duration_m,
                    'data_file': data_file_path,
                    'selected': False
                }
                
                self.current_data.append(item_data)
                
                # 插入树形视图
                item_id = self.tree.insert('', tk.END, values=(
                    '☐',
                    file_name,
                    channel_combo,
                    count,
                    f"{duration_s:.2f}",
                    f"{duration_m:.2f}",
                    os.path.basename(data_file_path)
                ))
                
                # 存储数据索引映射
                self.item_to_data_idx[item_id] = len(self.current_data) - 1
        
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")

    def toggle_selection(self, event):
        """双击切换选择状态"""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = selection[0]
        # 从映射中获取数据索引
        if item not in self.item_to_data_idx:
            return
        
        data_idx = self.item_to_data_idx[item]

        if 0 <= data_idx < len(self.current_data):
            self.current_data[data_idx]['selected'] = not self.current_data[data_idx]['selected']

            # 更新显示
            current_values = list(self.tree.item(item, 'values'))
            current_values[0] = '☑' if self.current_data[data_idx]['selected'] else '☐'
            self.tree.item(item, values=current_values)

            self.update_statistics()
    
    def select_all(self):
        """全选"""
        for i, data in enumerate(self.current_data):
            data['selected'] = True
        
        # 更新显示
        for item in self.tree.get_children():
            if item in self.item_to_data_idx:
                current_values = list(self.tree.item(item, 'values'))
                current_values[0] = '☑'
                self.tree.item(item, values=current_values)
        
        self.update_statistics()
    
    def clear_selection(self):
        """清除选择"""
        for data in self.current_data:
            data['selected'] = False
        
        # 更新显示
        for item in self.tree.get_children():
            current_values = list(self.tree.item(item, 'values'))
            current_values[0] = '☐'
            self.tree.item(item, values=current_values)
        
        self.update_statistics()
    
    def update_statistics(self):
        """更新统计信息"""
        selected_items = [d for d in self.current_data if d['selected']]
        
        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert(tk.END, f"总数据项: {len(self.current_data)}\n")
        self.stats_text.insert(tk.END, f"已选择: {len(selected_items)}\n")
        self.stats_text.insert(tk.END, f"\n{'='*30}\n\n")
        
        if selected_items:
            total_duration_s = sum(d['duration_s'] for d in selected_items)
            total_duration_m = total_duration_s / 60
            
            self.stats_text.insert(tk.END, f"选中项总时长:\n")
            self.stats_text.insert(tk.END, f"  {total_duration_s:.2f} 秒\n")
            self.stats_text.insert(tk.END, f"  {total_duration_m:.2f} 分钟\n\n")
            
            # 按通道分组统计
            channel_groups = defaultdict(lambda: {'count': 0, 'duration': 0})
            for d in selected_items:
                channel_groups[d['channel_combo']]['count'] += 1
                channel_groups[d['channel_combo']]['duration'] += d['duration_s']
            
            self.stats_text.insert(tk.END, "通道组合统计:\n")
            for channel, info in sorted(channel_groups.items(), key=lambda x: x[1]['duration'], reverse=True):
                self.stats_text.insert(tk.END, f"  {channel}:\n")
                self.stats_text.insert(tk.END, f"    文件数: {info['count']}\n")
                self.stats_text.insert(tk.END, f"    总时长: {info['duration']:.2f}s\n")
    
    def suggest_longest_label(self):
        """建议使用最长片段的标签"""
        selected_items = [d for d in self.current_data if d['selected']]
        if not selected_items:
            messagebox.showwarning("警告", "请先选择要合并的数据项")
            return
        
        # 找到时长最长的
        longest = max(selected_items, key=lambda x: x['duration_s'])
        self.merge_label_var.set(longest['channel_combo'])
    
    def custom_label(self):
        """自定义标签"""
        self.merge_label_entry.focus()
    
    def merge_selected(self):
        """合并选中的数据"""
        selected_items = [d for d in self.current_data if d['selected']]
        if not selected_items:
            messagebox.showwarning("警告", "请先选择要合并的数据项")
            return
        
        merge_label = self.merge_label_var.get().strip()
        if not merge_label:
            messagebox.showwarning("警告", "请输入合并后的通道标签")
            return
        
        # 确认合并
        msg = f"将合并 {len(selected_items)} 个数据项\n"
        msg += f"合并后标签: {merge_label}\n"
        total_duration = sum(d['duration_s'] for d in selected_items)
        msg += f"总时长: {total_duration:.2f}s ({total_duration/60:.2f}min)\n\n"
        msg += "确定要合并吗？"
        
        if not messagebox.askyesno("确认合并", msg):
            return
        
        # 执行合并
        try:
            patient_name = os.path.basename(self.patient_dirs[self.current_patient_idx])
            output_file = self.perform_merge(selected_items, merge_label, patient_name)
            
            if output_file:
                # 记录合并历史
                self.merge_history.append({
                    'patient': patient_name,
                    'merge_label': merge_label,
                    'num_files': len(selected_items),
                    'total_duration_s': total_duration,
                    'output_file': output_file,
                    'source_files': [d['data_file'] for d in selected_items]
                })
                
                messagebox.showinfo("成功", f"合并完成！\n输出文件: {os.path.basename(output_file)}")
                
                # 清除选择
                self.clear_selection()
        
        except Exception as e:
            messagebox.showerror("错误", f"合并失败: {str(e)}")
    
    def perform_merge(self, selected_items, merge_label, patient_name):
        """执行实际的数据合并"""
        print(f"\n开始合并 {len(selected_items)} 个数据文件...")
        
        # 收集所有数据
        all_data_list = []
        segment_info = []
        sfreq = None
        info = None
        
        for i, item in enumerate(selected_items):
            data_file = item['data_file']
            if not os.path.exists(data_file):
                print(f"  ⚠ 文件不存在: {data_file}")
                continue
            
            print(f"  [{i+1}/{len(selected_items)}] 加载: {os.path.basename(data_file)}")
            raw = mne.io.read_raw_eeglab(data_file, preload=True, verbose='ERROR')
            
            if sfreq is None:
                sfreq = raw.info['sfreq']
                info = raw.info.copy()
            
            # 提取数据
            data = raw.get_data()
            all_data_list.append(data)
            segment_info.append({
                'source': os.path.basename(data_file),
                'channel': item['channel_combo'],
                'duration': data.shape[1] / sfreq
            })
        
        if not all_data_list:
            raise Exception("没有可用的数据文件")
        
        # 拼接所有数据
        concatenated_data = np.concatenate(all_data_list, axis=1)
        total_samples = concatenated_data.shape[1]
        total_duration = total_samples / sfreq
        
        print(f"  ✓ 合并完成: {total_samples} 个采样点 ({total_duration:.2f}秒)")
        
        # 创建新的 Raw 对象
        new_raw = mne.io.RawArray(concatenated_data, info, verbose='ERROR')
        
        # 添加注释
        descriptions = []
        onsets = []
        durations = []
        cumulative_time = 0
        
        for seg in segment_info:
            descriptions.append(f"{seg['source']}: {seg['channel']}")
            onsets.append(cumulative_time)
            durations.append(seg['duration'])
            cumulative_time += seg['duration']
        
        annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
        new_raw.set_annotations(annotations)
        
        # 保存文件
        patient_dir = self.patient_dirs[self.current_patient_idx]
        safe_label = merge_label.replace(',', '_').replace(' ', '').replace('-', '_')
        output_name = f"{patient_name}_merged_{safe_label}.set"
        output_path = os.path.join(patient_dir, output_name)
        
        print(f"  保存到: {output_path}")
        mne.export.export_raw(output_path, new_raw, fmt='eeglab', overwrite=True, verbose='ERROR')
        print(f"  ✓ 保存成功!")
        
        return output_path
    
    def export_current_stats(self):
        """导出当前患者的统计"""
        patient_name = os.path.basename(self.patient_dirs[self.current_patient_idx])
        patient_dir = self.patient_dirs[self.current_patient_idx]
        
        csv_path = os.path.join(patient_dir, f"{patient_name}_merge_summary.csv")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['文件名', '通道组合', '片段数量', '总时长(秒)', '总时长(分钟)', '数据文件'])
            
            for data in self.current_data:
                writer.writerow([
                    data['file_name'],
                    data['channel_combo'],
                    data['count'],
                    f"{data['duration_s']:.2f}",
                    f"{data['duration_m']:.2f}",
                    os.path.basename(data['data_file'])
                ])
        
        messagebox.showinfo("导出成功", f"统计已保存到:\n{csv_path}")
    
    def next_patient(self):
        """下一个患者"""
        # 保存当前患者的合并历史
        if self.merge_history:
            self.save_merge_history()
        
        self.load_patient(self.current_patient_idx + 1)
    
    def prev_patient(self):
        """上一个患者"""
        self.load_patient(self.current_patient_idx - 1)
    
    def save_merge_history(self):
        """保存合并历史到CSV"""
        if not self.merge_history:
            return
        
        patient_name = os.path.basename(self.patient_dirs[self.current_patient_idx])
        patient_dir = self.patient_dirs[self.current_patient_idx]
        
        csv_path = os.path.join(patient_dir, f"{patient_name}_merge_history.csv")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['合并标签', '源文件数量', '总时长(秒)', '总时长(分钟)', '输出文件', '源文件列表'])
            
            for record in self.merge_history:
                if record['patient'] == patient_name:
                    writer.writerow([
                        record['merge_label'],
                        record['num_files'],
                        f"{record['total_duration_s']:.2f}",
                        f"{record['total_duration_s']/60:.2f}",
                        os.path.basename(record['output_file']),
                        '; '.join([os.path.basename(f) for f in record['source_files']])
                    ])
        
        print(f"✓ 保存合并历史: {csv_path}")
    
    def save_and_exit(self):
        """保存并退出"""
        # 保存当前患者的合并历史
        if self.merge_history:
            self.save_merge_history()
        
        # 保存全局合并历史
        if self.merge_history:
            all_history_path = os.path.join(os.path.dirname(self.patient_dirs[0]), "all_patients_merge_history.csv")
            
            with open(all_history_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['患者', '合并标签', '源文件数量', '总时长(秒)', '总时长(分钟)', '输出文件'])
                
                for record in self.merge_history:
                    writer.writerow([
                        record['patient'],
                        record['merge_label'],
                        record['num_files'],
                        f"{record['total_duration_s']:.2f}",
                        f"{record['total_duration_s']/60:.2f}",
                        record['output_file']
                    ])
            
            print(f"✓ 保存全局合并历史: {all_history_path}")
        
        if messagebox.askyesno("退出", "确定要退出吗？"):
            self.root.quit()


def find_patient_directories(root_dir):
    """
    查找所有包含 _channel_merge_statistics.csv 的患者目录

    返回患者目录列表（患者目录定义为包含 _channel_merge_statistics.csv 文件的目录的上级目录）
    """
    stats_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('_channel_merge_statistics.csv'):
                stats_files.append(dirpath)

    if not stats_files:
        return []

    # 患者目录就是包含 _channel_merge_statistics.csv 文件的目录
    patient_dirs = set()
    for path in stats_files:
        patient_dirs.add(path)

    return sorted(list(patient_dirs))


def main():
    parser = argparse.ArgumentParser(
        description="交互式患者数据合并工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
功能说明:
  1. 读取每个患者下所有的 _channel_merge_statistics.csv
  2. 显示所有通道类型和时间长度
  3. 通过界面勾选要合并的片段
  4. 支持修改合并后的通道组合标签
  5. 执行合并并保存结果
  6. 跳转患者时自动保存合并统计

使用方法:
  python interactive_patient_merge.py --root_dir "E:\\DataSet\\EEG\\EEG dataset_SUAT_processed"
        """
    )
    
    parser.add_argument(
        '--root_dir',
        required=True,
        help="数据集根目录"
    )
    
    args = parser.parse_args()
    
    print("正在扫描患者目录...")
    patient_dirs = find_patient_directories(args.root_dir)
    
    if not patient_dirs:
        print("❌ 未找到包含 _channel_merge_statistics.csv 的患者目录")
        print("   请确保已运行 merge_by_channels.py 生成统计文件")
        return 1
    
    print(f"✓ 找到 {len(patient_dirs)} 个患者目录:")
    for i, d in enumerate(patient_dirs):
        print(f"  {i+1}. {os.path.basename(d)}")
    
    # 启动 GUI
    root = tk.Tk()
    app = PatientMergeGUI(root, patient_dirs)
    root.mainloop()
    
    return 0


if __name__ == "__main__":
    # 如果不提供命令行参数，使用默认值
    import sys

    if len(sys.argv) == 1:
        print("使用默认参数运行...")
        sys.argv.extend([
            '--root_dir', r'E:\DataSet\EEG\EEG dataset_SUAT_processed\头皮数据-6例'
        ])
    sys.exit(main())

