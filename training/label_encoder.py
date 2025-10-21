"""
Label encoding utilities for channel combinations.

This module provides utilities to encode channel combinations into meaningful labels
for classification tasks.
"""

import re
from typing import List, Dict, Set, Tuple
import numpy as np
import pandas as pd


def parse_channel_combination(channel_str: str) -> List[str]:
    """
    Parse channel combination string into list of channel names.
    
    Args:
        channel_str: String like "[F7,Sph_L,T3]" or "[Fp2]"
        
    Returns:
        List of channel names
    """
    # Remove brackets and split by comma
    channels = channel_str.strip('[]').split(',')
    # Clean up whitespace
    channels = [ch.strip() for ch in channels]
    return channels


def get_channel_frequency_features(channels: List[str]) -> Dict[str, int]:
    """
    Extract frequency-based features from channel names.
    
    Args:
        channels: List of channel names
        
    Returns:
        Dictionary with frequency band counts
    """
    features = {
        'frontal': 0,
        'temporal': 0,
        'parietal': 0,
        'occipital': 0,
        'central': 0,
        'other': 0
    }
    
    for ch in channels:
        ch_upper = ch.upper()
        if ch_upper.startswith('F'):
            features['frontal'] += 1
        elif ch_upper.startswith('T'):
            features['temporal'] += 1
        elif ch_upper.startswith('P'):
            features['parietal'] += 1
        elif ch_upper.startswith('O'):
            features['occipital'] += 1
        elif ch_upper.startswith('C'):
            features['central'] += 1
        else:
            features['other'] += 1
    
    return features


def encode_channel_combination(channel_str: str, encoding_type: str = 'frequency') -> int:
    """
    Encode channel combination into integer label.
    
    Args:
        channel_str: Channel combination string
        encoding_type: Type of encoding ('frequency', 'count', 'hash', 'binary')
        
    Returns:
        Integer label
    """
    channels = parse_channel_combination(channel_str)
    
    if encoding_type == 'frequency':
        # Encode based on frequency features
        features = get_channel_frequency_features(channels)
        # Create a simple hash from frequency features
        feature_vec = [features['frontal'], features['temporal'], features['parietal'], 
                     features['occipital'], features['central'], features['other']]
        return hash(tuple(feature_vec)) % 1000
    
    elif encoding_type == 'count':
        # Simple count-based encoding
        return len(channels)
    
    elif encoding_type == 'hash':
        # Direct hash of channel combination
        return hash(tuple(sorted(channels))) % 1000
    
    elif encoding_type == 'binary':
        # Binary encoding based on presence of specific channels
        # This is a simplified example - you can extend this
        binary_features = []
        all_possible_channels = ['F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz',
                               'T3', 'T4', 'T5', 'T6', 'C3', 'C4', 'Cz',
                               'P3', 'P4', 'Pz', 'O1', 'O2', 'Oz',
                               'Sph_L', 'Sph_R']
        
        for ch in all_possible_channels:
            binary_features.append(1 if ch in channels else 0)
        
        # Convert binary to integer
        return int(''.join(map(str, binary_features)), 2) % 1000
    
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")


def create_label_mapping(df: pd.DataFrame, encoding_type: str = 'frequency') -> Dict[str, int]:
    """
    Create label mapping from DataFrame.
    
    Args:
        df: DataFrame with 'channel_combination' column
        encoding_type: Type of encoding to use
        
    Returns:
        Dictionary mapping channel combinations to labels
    """
    label_mapping = {}
    
    for _, row in df.iterrows():
        channel_combo = row['channel_combination']
        label = encode_channel_combination(channel_combo, encoding_type)
        label_mapping[channel_combo] = label
    
    return label_mapping


def analyze_channel_combinations(df: pd.DataFrame) -> Dict:
    """
    Analyze channel combinations in the dataset.
    
    Args:
        df: DataFrame with 'channel_combination' column
        
    Returns:
        Dictionary with analysis results
    """
    all_combinations = df['channel_combination'].unique()
    
    # Parse all combinations
    parsed_combinations = [parse_channel_combination(combo) for combo in all_combinations]
    
    # Count channel frequencies
    channel_counts = {}
    for combo in parsed_combinations:
        for ch in combo:
            channel_counts[ch] = channel_counts.get(ch, 0) + 1
    
    # Count combination sizes
    combo_sizes = [len(combo) for combo in parsed_combinations]
    
    # Get unique channels
    all_channels = set()
    for combo in parsed_combinations:
        all_channels.update(combo)
    
    analysis = {
        'total_combinations': len(all_combinations),
        'unique_channels': len(all_channels),
        'channel_counts': channel_counts,
        'combo_size_stats': {
            'min': min(combo_sizes),
            'max': max(combo_sizes),
            'mean': np.mean(combo_sizes),
            'std': np.std(combo_sizes)
        },
        'all_channels': sorted(list(all_channels))
    }
    
    return analysis


def create_balanced_labels(df: pd.DataFrame, target_classes: int = 5) -> Dict[str, int]:
    """
    Create balanced labels by clustering channel combinations.
    
    Args:
        df: DataFrame with 'channel_combination' column
        target_classes: Number of target classes
        
    Returns:
        Dictionary mapping channel combinations to balanced labels
    """
    from sklearn.cluster import KMeans
    
    # Extract features for clustering
    features = []
    combinations = []
    
    for _, row in df.iterrows():
        channel_combo = row['channel_combination']
        channels = parse_channel_combination(channel_combo)
        freq_features = get_channel_frequency_features(channels)
        
        # Create feature vector
        feature_vec = [
            freq_features['frontal'],
            freq_features['temporal'], 
            freq_features['parietal'],
            freq_features['occipital'],
            freq_features['central'],
            freq_features['other'],
            len(channels)  # Add combination size
        ]
        
        features.append(feature_vec)
        combinations.append(channel_combo)
    
    features = np.array(features)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=target_classes, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    
    # Create mapping
    label_mapping = {}
    for combo, label in zip(combinations, cluster_labels):
        label_mapping[combo] = int(label)
    
    return label_mapping


if __name__ == '__main__':
    # Example usage
    import pandas as pd
    
    # Load labels CSV
    df = pd.read_csv('E:/output/connectivity_features/labels.csv')
    
    # Analyze combinations
    analysis = analyze_channel_combinations(df)
    print("Channel Combination Analysis:")
    print(f"Total combinations: {analysis['total_combinations']}")
    print(f"Unique channels: {analysis['unique_channels']}")
    print(f"Combination size: {analysis['combo_size_stats']['mean']:.1f} ± {analysis['combo_size_stats']['std']:.1f}")
    print(f"All channels: {analysis['all_channels']}")
    
    # Create different encodings
    print("\nCreating label mappings...")
    
    # Frequency-based encoding
    freq_mapping = create_label_mapping(df, 'frequency')
    print(f"Frequency encoding: {len(set(freq_mapping.values()))} unique labels")
    
    # Count-based encoding
    count_mapping = create_label_mapping(df, 'count')
    print(f"Count encoding: {len(set(count_mapping.values()))} unique labels")
    
    # Balanced clustering
    balanced_mapping = create_balanced_labels(df, target_classes=5)
    print(f"Balanced encoding: {len(set(balanced_mapping.values()))} unique labels")
