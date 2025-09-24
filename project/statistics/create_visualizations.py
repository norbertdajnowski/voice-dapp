import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load and prepare data from CSV files"""
    print("Loading data...")
    
    # Load pre-calculated thresholds analysis data
    thresholds_df = pd.read_csv('thresholds_analysis.csv')
    print(f"Thresholds analysis: {len(thresholds_df)} rows")
    
    # Load multi-user results for iteration-based TAR plot
    multi_user_df = pd.read_csv('multi_user_results20092025.csv')
    print(f"Multi-user results: {len(multi_user_df)} rows")
    
    # Load recognition results for iteration-based TAR plot
    recognition_df = pd.read_csv('recognition_results20092025.csv')
    print(f"Recognition results: {len(recognition_df)} rows")
    
    # Extract genuine and impostor scores for iteration analysis
    genuine_scores = multi_user_df[multi_user_df['is_true_user'] == True]['similarity'].values
    impostor_scores = multi_user_df[multi_user_df['is_true_user'] == False]['similarity'].values
    
    # Extract scores from recognition data (emotion matching)
    # Genuine: train_emotion == test_emotion (same emotion)
    # Impostor: train_emotion != test_emotion (different emotions)
    rec_genuine_scores = recognition_df[recognition_df['train_emotion'] == recognition_df['test_emotion']]['similarity'].values
    rec_impostor_scores = recognition_df[recognition_df['train_emotion'] != recognition_df['test_emotion']]['similarity'].values
    
    # Combine all scores for iteration analysis
    all_genuine_scores = np.concatenate([genuine_scores, rec_genuine_scores])
    all_impostor_scores = np.concatenate([impostor_scores, rec_impostor_scores])
    
    print(f"Combined genuine scores: {len(all_genuine_scores)}")
    print(f"Combined impostor scores: {len(all_impostor_scores)}")
    
    return thresholds_df, all_genuine_scores, all_impostor_scores, multi_user_df

def create_det_curve(thresholds_df):
    """Create Detection Error Tradeoff (DET) curve - Fig. 6"""
    print("Creating DET curve...")
    
    # Use pre-calculated FAR and TAR values
    fmr_values = thresholds_df['far'].values  # FAR = FMR (False Match Rate)
    fnmr_values = 1 - thresholds_df['tar'].values  # FNMR = 1 - TAR (False Non-Match Rate)
    
    # Find EER (Equal Error Rate) point
    eer_idx = np.argmin(np.abs(fmr_values - fnmr_values))
    eer_value = fmr_values[eer_idx]
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot FMR vs FNMR curve
    plt.plot(fmr_values, fnmr_values, 'b-', linewidth=2, marker='o', markersize=4, label='FMR vs FNMR')
    
    # Plot diagonal line for EER
    plt.plot([0, 1], [0, 1], 'r-', linewidth=2, label='EER')
    
    # Mark EER point
    plt.plot(eer_value, eer_value, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2, label=f'EER = {eer_value:.2f}')
    
    # Customize plot
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Match Rate (FMR)', fontsize=12)
    plt.ylabel('False Non-Match Rate (FNMR)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Set tick marks at 0.1 intervals
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    plt.tight_layout()
    plt.savefig('det_curve_fig6.png', dpi=300, bbox_inches='tight')
    print("✓ DET curve saved as 'det_curve_fig6.png'")
    plt.close()

def create_roc_curve(thresholds_df):
    """Create ROC curve - Fig. 7"""
    print("Creating ROC curve...")
    
    # Use pre-calculated FAR and TAR values
    fpr = thresholds_df['far'].values  # FAR = False Positive Rate
    tpr = thresholds_df['tar'].values  # TAR = True Positive Rate
    
    # Calculate AUC
    roc_auc = auc(fpr, tpr)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=3, marker='o', markersize=3, 
             label=f'ROC (AUC = {roc_auc:.4f})')
    
    # Plot diagonal baseline
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='FAR=TAR')
    
    # Fill area under curve
    plt.fill_between(fpr, tpr, alpha=0.3, color='lightgreen')
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FAR', fontsize=12)
    plt.ylabel('TAR', fontsize=12)
    plt.title('Fig. 7. ROC curve plotted between FAR against TAR.', fontsize=14)
    plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Set tick marks at 0.1 intervals
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    plt.tight_layout()
    plt.savefig('roc_curve_fig7.png', dpi=300, bbox_inches='tight')
    print("✓ ROC curve saved as 'roc_curve_fig7.png'")
    plt.close()
    
    return roc_auc

def create_threshold_performance_plots(multi_user_df, thresholds_df):
    """Create separate threshold performance plots for TAR, FAR, TRR, FRR"""
    print("Creating threshold performance plots...")
    
    # Use specific iterations as shown in the image: 50, 75, 100
    target_iterations = [50, 75, 100]
    
    # Colors and markers for different iterations (matching the image)
    colors = ['blue', 'orange', 'yellow']
    linestyles = ['--', ':', '-.']
    markers = ['+', 's', 'o']
    
    # Calculate performance metrics for each iteration
    iteration_data = {}
    
    for i, iteration in enumerate(target_iterations):
        # Filter data for this iteration
        iter_data = multi_user_df[multi_user_df['iteration'] == iteration]
        
        if len(iter_data) == 0:
            print(f"Warning: No data found for iteration {iteration}")
            continue
            
        # Get genuine and impostor scores for this iteration
        genuine_scores = iter_data[iter_data['is_true_user'] == True]['similarity'].values
        impostor_scores = iter_data[iter_data['is_true_user'] == False]['similarity'].values
        
        # Calculate performance metrics for different thresholds (use range 0.15 to 0.65 as in image)
        threshold_range = np.arange(0.15, 0.66, 0.01)
        
        tar_values = []
        far_values = []
        trr_values = []
        frr_values = []
        
        for threshold in threshold_range:
            # TAR (True Acceptance Rate) = genuine_above / total_genuine
            genuine_above = np.sum(genuine_scores >= threshold)
            tar = genuine_above / len(genuine_scores) if len(genuine_scores) > 0 else 0
            
            # FAR (False Acceptance Rate) = impostor_above / total_impostor
            impostor_above = np.sum(impostor_scores >= threshold)
            far = impostor_above / len(impostor_scores) if len(impostor_scores) > 0 else 0
            
            # TRR (True Rejection Rate) = impostor_below / total_impostor = 1 - FAR
            trr = 1 - far
            
            # FRR (False Rejection Rate) = genuine_below / total_genuine = 1 - TAR
            frr = 1 - tar
            
            tar_values.append(tar)
            far_values.append(far)
            trr_values.append(trr)
            frr_values.append(frr)
        
        iteration_data[iteration] = {
            'thresholds': threshold_range,
            'tar': tar_values,
            'far': far_values,
            'trr': trr_values,
            'frr': frr_values,
            'color': colors[i % len(colors)],
            'linestyle': linestyles[i % len(linestyles)],
            'marker': markers[i % len(markers)]
        }
    
    # Create separate plots for each metric
    
    # 1. TAR vs Threshold
    plt.figure(figsize=(12, 8))
    for iteration, data in iteration_data.items():
        plt.plot(data['thresholds'], data['tar'], 
                color=data['color'], linestyle=data['linestyle'], 
                marker=data['marker'], linewidth=2, markersize=6, 
                label=f'Iteration {iteration}')
    
    plt.xlim([0.15, 0.65])
    plt.ylim([0, 1])
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('TAR (True Acceptance Rate)', fontsize=12)
    plt.title('TAR vs. Threshold for Different Iterations', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(0.15, 0.7, 0.05))
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    plt.tight_layout()
    plt.savefig('tar_vs_threshold.png', dpi=300, bbox_inches='tight')
    print("✓ TAR vs threshold plot saved as 'tar_vs_threshold.png'")
    plt.close()
    
    # 2. FAR vs Threshold
    plt.figure(figsize=(12, 8))
    for iteration, data in iteration_data.items():
        plt.plot(data['thresholds'], data['far'], 
                color=data['color'], linestyle=data['linestyle'], 
                marker=data['marker'], linewidth=2, markersize=6, 
                label=f'Iteration {iteration}')
    
    plt.xlim([0.15, 0.65])
    plt.ylim([0, 1])
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('FAR (False Acceptance Rate)', fontsize=12)
    plt.title('FAR vs. Threshold for Different Iterations', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(0.15, 0.7, 0.05))
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    plt.tight_layout()
    plt.savefig('far_vs_threshold.png', dpi=300, bbox_inches='tight')
    print("✓ FAR vs threshold plot saved as 'far_vs_threshold.png'")
    plt.close()
    
    # 3. TRR vs Threshold
    plt.figure(figsize=(12, 8))
    for iteration, data in iteration_data.items():
        plt.plot(data['thresholds'], data['trr'], 
                color=data['color'], linestyle=data['linestyle'], 
                marker=data['marker'], linewidth=2, markersize=6, 
                label=f'Iteration {iteration}')
    
    plt.xlim([0.15, 0.65])
    plt.ylim([0, 1])
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('TRR (True Rejection Rate)', fontsize=12)
    plt.title('TRR vs. Threshold for Different Iterations', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(0.15, 0.7, 0.05))
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    plt.tight_layout()
    plt.savefig('trr_vs_threshold.png', dpi=300, bbox_inches='tight')
    print("✓ TRR vs threshold plot saved as 'trr_vs_threshold.png'")
    plt.close()
    
    # 4. FRR vs Threshold
    plt.figure(figsize=(12, 8))
    for iteration, data in iteration_data.items():
        plt.plot(data['thresholds'], data['frr'], 
                color=data['color'], linestyle=data['linestyle'], 
                marker=data['marker'], linewidth=2, markersize=6, 
                label=f'Iteration {iteration}')
    
    plt.xlim([0.15, 0.65])
    plt.ylim([0, 1])
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('FRR (False Rejection Rate)', fontsize=12)
    plt.title('FRR vs. Threshold for Different Iterations', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(0.15, 0.7, 0.05))
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    plt.tight_layout()
    plt.savefig('frr_vs_threshold.png', dpi=300, bbox_inches='tight')
    print("✓ FRR vs threshold plot saved as 'frr_vs_threshold.png'")
    plt.close()
    
    # Also create the original TAR vs threshold plot (Fig. 8)
    plt.figure(figsize=(12, 8))
    for iteration, data in iteration_data.items():
        plt.plot(data['thresholds'], data['tar'], 
                color=data['color'], linestyle=data['linestyle'], 
                marker=data['marker'], linewidth=2, markersize=6, 
                label=f'Iteration {iteration}')
    
    plt.xlim([0.15, 0.65])
    plt.ylim([0, 1])
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('TAR', fontsize=12)
    plt.title('Fig. 8. TAR vs. threshold for different iterations.', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(0.15, 0.7, 0.05))
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    plt.tight_layout()
    plt.savefig('tar_threshold_fig8.png', dpi=300, bbox_inches='tight')
    print("✓ TAR vs threshold plot saved as 'tar_threshold_fig8.png'")
    plt.close()

def main():
    """Main function to create all visualizations"""
    print("="*60)
    print("CREATING VISUALIZATIONS FROM ATTACHED IMAGES")
    print("="*60)
    
    # Load data
    thresholds_df, genuine_scores, impostor_scores, multi_user_df = load_data()
    
    # Create Fig. 6: DET curve
    create_det_curve(thresholds_df)
    
    # Create Fig. 7: ROC curve
    roc_auc = create_roc_curve(thresholds_df)
    
    # Create comprehensive threshold performance plots
    create_threshold_performance_plots(multi_user_df, thresholds_df)
    
    print("\n" + "="*60)
    print("VISUALIZATION CREATION COMPLETE!")
    print("="*60)
    print(f"✓ ROC AUC Score: {roc_auc:.4f}")
    print("✓ Generated files:")
    print("  - det_curve_fig6.png (DET curve)")
    print("  - roc_curve_fig7.png (ROC curve)")
    print("  - tar_threshold_fig8.png (TAR vs threshold - Fig. 8)")
    print("  - tar_vs_threshold.png (TAR vs threshold)")
    print("  - far_vs_threshold.png (FAR vs threshold)")
    print("  - trr_vs_threshold.png (TRR vs threshold)")
    print("  - frr_vs_threshold.png (FRR vs threshold)")
    print("="*60)

if __name__ == "__main__":
    main()
