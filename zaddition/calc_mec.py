import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import argparse
import os

def load_openface_csv(csv_path):
    """
    读取 OpenFace 输出的 CSV 文件，并提取 Action Units (AU) 的强度列
    OpenFace 输出列通常命名为 'AU01_r', 'AU02_r' 等 (r 代表 regression/intensity)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"OpenFace CSV not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # 筛选 AU 列
    # AU01: Inner Brow Raiser (内眉提升)
    # AU02: Outer Brow Raiser (外眉提升)
    # AU04: Brow Lowerer (皱眉)
    # AU12: Lip Corner Puller (嘴角上扬)
    # AU15: Lip Corner Depressor (嘴角下压)
    # ...
    au_cols = [col for col in df.columns if 'AU' in col and '_r' in col]
    return df[au_cols]

def calculate_mec(pred_csv, gt_csv, output_plot=None):
    """
    计算 MEC (Micro-Expression Consistency) 分数
    """
    print(f"Loading Prediction: {pred_csv}")
    print(f"Loading GroundTruth: {gt_csv}")
    
    pred_df = load_openface_csv(pred_csv)
    gt_df = load_openface_csv(gt_csv)
    
    # 对齐长度
    min_len = min(len(pred_df), len(gt_df))
    pred_df = pred_df.iloc[:min_len]
    gt_df = gt_df.iloc[:min_len]
    
    au_scores = {}
    print("\n--- AU Correlation Analysis ---")
    print(f"{'AU Code':<10} | {'Correlation (PCC)':<20}")
    print("-" * 30)
    
    for col in pred_df.columns:
        # 计算皮尔逊相关系数
        pred_signal = pred_df[col].values
        gt_signal = gt_df[col].values
        
        # 避免除零错误 (如果信号完全静止)
        if np.std(pred_signal) < 1e-6 or np.std(gt_signal) < 1e-6:
            corr = 0.0
        else:
            corr, _ = scipy.stats.pearsonr(pred_signal, gt_signal)
            
        au_scores[col] = corr
        print(f"{col:<10} | {corr:.4f}")
        
    # 计算平均 MEC
    mec_score = np.mean(list(au_scores.values()))
    print("-" * 30)
    print(f"Final MEC Score: {mec_score:.4f}")
    
    # 绘制可视化图表 (加分项)
    if output_plot:
        plt.figure(figsize=(12, 6))
        x = list(au_scores.keys())
        y = list(au_scores.values())
        plt.bar(x, y, color='skyblue')
        plt.axhline(y=mec_score, color='r', linestyle='--', label=f'Avg MEC: {mec_score:.2f}')
        plt.title('Micro-Expression Consistency by Action Unit')
        plt.xlabel('Facial Action Units (FACS)')
        plt.ylabel('Pearson Correlation')
        plt.ylim(-1, 1)
        plt.legend()
        plt.savefig(output_plot)
        print(f"Plot saved to {output_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MEC Metric for Talking Head Generation")
    parser.add_argument('--pred_csv', type=str, required=True, help='Path to OpenFace CSV of generated video')
    parser.add_argument('--gt_csv', type=str, required=True, help='Path to OpenFace CSV of ground truth video')
    parser.add_argument('--output_plot', type=str, default='mec_radar.png', help='Path to save the analysis plot')
    
    args = parser.parse_args()
    
    # 为了演示，生成一些 Mock Data (实际使用时请注释掉)
    # create_mock_data(args.pred_csv, args.gt_csv) 
    
    calculate_mec(args.pred_csv, args.gt_csv, args.output_plot)