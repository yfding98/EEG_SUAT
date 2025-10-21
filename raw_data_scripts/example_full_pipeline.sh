#!/bin/bash
# example_full_pipeline.sh
# 
# 完整的EEG数据处理和连接性分析流程示例
# 
# 使用方法:
#   bash example_full_pipeline.sh
# 
# 注意: 
#   1. 修改下面的路径为你的实际路径
#   2. 确保已安装所有依赖
#   3. 本脚本仅作为示例，请根据实际情况调整参数

set -e  # 遇到错误立即退出

# =============================================================================
# 配置参数
# =============================================================================

# 输入输出路径（请修改为你的路径）
DATA_ROOT="E:/DataSet/EEG/EEG dataset_SUAT_processed"
OUTPUT_ROOT="E:/output"

# 创建输出目录
mkdir -p "$OUTPUT_ROOT"

echo "=========================================="
echo "EEG完整处理流程"
echo "=========================================="
echo "数据根目录: $DATA_ROOT"
echo "输出根目录: $OUTPUT_ROOT"
echo ""

# =============================================================================
# 步骤1: 异常检测（可选 - 如果已有标注可跳过）
# =============================================================================

echo "----------------------------------------"
echo "步骤1: 异常片段检测与标注"
echo "----------------------------------------"

# 注意：这个步骤需要人工交互，可以选择跳过
# python automated_preictal_detector.py --input_file "$DATA_ROOT/patient1/data.set"

echo "跳过自动检测（假设已有标注）..."
echo ""

# =============================================================================
# 步骤2: 按通道合并片段
# =============================================================================

echo "----------------------------------------"
echo "步骤2: 按异常通道合并片段"
echo "----------------------------------------"

python merge_by_channels.py \
    --root_dir "$DATA_ROOT" \
    || { echo "错误: 合并失败"; exit 1; }

echo "✓ 合并完成"
echo ""

# =============================================================================
# 步骤3: 收集长片段（≥30秒）
# =============================================================================

echo "----------------------------------------"
echo "步骤3: 收集长片段（≥30秒）"
echo "----------------------------------------"

LONG_SEG_DIR="$OUTPUT_ROOT/long_segments_30s"

python collect_long_segments.py \
    --root_dir "$DATA_ROOT" \
    --out_dir "$LONG_SEG_DIR" \
    --min_duration 30 \
    || { echo "错误: 收集失败"; exit 1; }

echo "✓ 长片段收集完成"
echo ""

# =============================================================================
# 步骤4: 解压长片段文件（如果生成了ZIP）
# =============================================================================

echo "----------------------------------------"
echo "步骤4: 准备数据文件"
echo "----------------------------------------"

# 如果生成了ZIP文件，需要解压
ZIP_FILE="$LONG_SEG_DIR.zip"
if [ -f "$ZIP_FILE" ]; then
    echo "发现ZIP文件，正在解压..."
    unzip -o "$ZIP_FILE" -d "$LONG_SEG_DIR"
    echo "✓ 解压完成"
else
    echo "未找到ZIP文件，继续..."
fi

echo ""

# =============================================================================
# 步骤5: 提取连接性特征
# =============================================================================

echo "----------------------------------------"
echo "步骤5: 提取连接性特征"
echo "----------------------------------------"

FEATURES_DIR="$OUTPUT_ROOT/connectivity_features"

python extract_connectivity_features.py \
    --input_dir "$LONG_SEG_DIR" \
    --pattern "*_merged_*.set" \
    --output_dir "$FEATURES_DIR" \
    --window_size 30 \
    --dfc_window 2 \
    --dfc_step 1 \
    --sparsity 0.2 \
    || { echo "错误: 特征提取失败"; exit 1; }

echo "✓ 特征提取完成"
echo ""

# =============================================================================
# 步骤6: 分析和可视化结果
# =============================================================================

echo "----------------------------------------"
echo "步骤6: 分析和可视化"
echo "----------------------------------------"

# 查找所有特征目录
FEATURE_DIRS=$(find "$FEATURES_DIR" -type d -name "*_connectivity_features")

if [ -z "$FEATURE_DIRS" ]; then
    echo "警告: 未找到特征目录"
else
    # 对每个特征目录进行分析
    for FEAT_DIR in $FEATURE_DIRS; do
        echo "分析: $(basename $FEAT_DIR)"
        
        python analyze_connectivity_features.py \
            --feature_dir "$FEAT_DIR" \
            --output_dir "$FEAT_DIR/analysis" \
            || { echo "警告: 分析 $(basename $FEAT_DIR) 失败"; }
    done
    
    echo "✓ 分析完成"
fi

echo ""

# =============================================================================
# 步骤7: 汇总结果
# =============================================================================

echo "----------------------------------------"
echo "步骤7: 汇总所有结果"
echo "----------------------------------------"

# 创建汇总目录
SUMMARY_DIR="$OUTPUT_ROOT/summary"
mkdir -p "$SUMMARY_DIR"

# 收集所有scalar_features.csv
echo "收集所有标量特征文件..."
find "$FEATURES_DIR" -name "scalar_features.csv" -exec cp {} "$SUMMARY_DIR/" \;

# 统计文件数量
CSV_COUNT=$(ls -1 "$SUMMARY_DIR"/*.csv 2>/dev/null | wc -l)
echo "✓ 收集了 $CSV_COUNT 个特征文件"

echo ""

# =============================================================================
# 完成
# =============================================================================

echo "=========================================="
echo "处理完成！"
echo "=========================================="
echo ""
echo "输出目录结构:"
echo "  $OUTPUT_ROOT/"
echo "  ├── long_segments_30s/          # 长片段数据"
echo "  ├── connectivity_features/      # 连接性特征"
echo "  │   ├── file1_connectivity_features/"
echo "  │   │   ├── scalar_features.csv"
echo "  │   │   ├── connectivity_matrices_segXXX.npz"
echo "  │   │   └── analysis/           # 分析结果"
echo "  │   └── ..."
echo "  └── summary/                    # 汇总结果"
echo ""
echo "后续分析建议:"
echo "  1. 查看 summary/ 目录下的所有特征文件"
echo "  2. 使用Python进行机器学习分类"
echo "  3. 比较不同患者/条件的连接性差异"
echo "  4. 分析动态连接的时间演化"
echo ""
echo "示例分析代码:"
echo "  python -c 'import pandas as pd; df=pd.read_csv(\"$SUMMARY_DIR/scalar_features.csv\"); print(df.describe())'"
echo ""

exit 0

