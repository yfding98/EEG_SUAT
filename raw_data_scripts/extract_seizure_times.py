import os
import csv
import mne
from pathlib import Path

"""
提取edf中的注解中的癫痫发作时间
"""


def main(root_dir, output_csv):
    results = []
    for edf_path in Path(root_dir).rglob('*.edf'):
        rel_path = str(edf_path.relative_to(root_dir))
        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=False)
            annotations = raw.annotations
            seizure_times = []
            for ann in annotations:
                if ann['description'].lower().startswith('sz'):
                    seizure_times.append(ann['onset'])
            if seizure_times:
                print(f"File: {rel_path}")
                for t in seizure_times:
                    print(f"  Seizure onset: {t}")
                for t in seizure_times:
                    results.append((rel_path, t))
        except Exception as e:
            print(f"Error processing {rel_path}: {e}")
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['relative_path', 'seizure_time'])
        for rel_path, t in results:
            writer.writerow([rel_path, t])

if __name__ == '__main__':
    # Set your root directory here
    root_dir = r'E:/DataSet/EEG/EEG dataset_SUAT'  # Current directory, change as needed
    output_csv = 'seizure_times.csv'
    main(root_dir, output_csv) 