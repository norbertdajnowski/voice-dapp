import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import sys
import io

# Fix Unicode encoding issues on Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """
    Step 1: Collect Scores - Load genuine and impostor scores from both data files
    """
    print("="*80)
    print("THRESHOLD ANALYSIS - ROC CURVE FAR/TAR CALCULATION")
    print("="*80)
    
    print("\nStep 1: Collecting Scores...")
    
    # Load the multi-user results data
    multi_user_df = pd.read_csv('multi_user_results20092025.csv')
    print(f"✓ Multi-user results loaded: {len(multi_user_df)} rows")
    
    # Load the recognition results data
    recognition_df = pd.read_csv('recognition_results20092025.csv')
    print(f"✓ Recognition results loaded: {len(recognition_df)} rows")
    
    # Extract scores from multi-user data
    multi_genuine_scores = multi_user_df[multi_user_df['is_true_user'] == True]['similarity'].values
    multi_impostor_scores = multi_user_df[multi_user_df['is_true_user'] == False]['similarity'].values
    
    # Extract scores from recognition data
    # Recognition results have different structure - they're emotion-based comparisons
    print(f"\nRecognition results columns: {list(recognition_df.columns)}")
    
    # For recognition results, we'll use similarity scores
    # Success results (same emotion) can be considered "genuine-like"
    # Failed results (different emotions) can be considered "impostor-like"
    if 'similarity' in recognition_df.columns:
        # Use similarity scores from recognition results
        rec_similarity_scores = recognition_df['similarity'].values
        
        # For emotion recognition, we classify based on emotion matching:
        # - Genuine: train_emotion == test_emotion (same emotion)
        # - Impostor: train_emotion != test_emotion (different emotions)
        
        # Get genuine scores (same emotion matches)
        rec_genuine_scores = recognition_df[recognition_df['train_emotion'] == recognition_df['test_emotion']]['similarity'].values
        # Get impostor scores (different emotion matches)  
        rec_impostor_scores = recognition_df[recognition_df['train_emotion'] != recognition_df['test_emotion']]['similarity'].values
        
        print(f"Recognition emotion analysis:")
        print(f"  - Same emotion matches (genuine): {len(rec_genuine_scores)}")
        print(f"  - Different emotion matches (impostor): {len(rec_impostor_scores)}")
        
        # Show some examples of emotion matching
        print(f"  - Sample genuine emotion pairs: {recognition_df[recognition_df['train_emotion'] == recognition_df['test_emotion']][['train_emotion', 'test_emotion', 'similarity']].head(3).values.tolist()}")
        print(f"  - Sample impostor emotion pairs: {recognition_df[recognition_df['train_emotion'] != recognition_df['test_emotion']][['train_emotion', 'test_emotion', 'similarity']].head(3).values.tolist()}")
    else:
        print("Warning: No similarity column found in recognition results, using multi-user data only")
        rec_genuine_scores = np.array([])
        rec_impostor_scores = np.array([])
    
    # Combine all genuine and impostor scores
    all_genuine_scores = np.concatenate([multi_genuine_scores, rec_genuine_scores]) if len(rec_genuine_scores) > 0 else multi_genuine_scores
    all_impostor_scores = np.concatenate([multi_impostor_scores, rec_impostor_scores]) if len(rec_impostor_scores) > 0 else multi_impostor_scores
    
    print(f"✓ Combined genuine scores: {len(all_genuine_scores)} samples")
    print(f"  - From multi-user data: {len(multi_genuine_scores)}")
    print(f"  - From recognition data: {len(rec_genuine_scores)}")
    print(f"✓ Combined impostor scores: {len(all_impostor_scores)} samples")
    print(f"  - From multi-user data: {len(multi_impostor_scores)}")
    print(f"  - From recognition data: {len(rec_impostor_scores)}")
    print(f"✓ Genuine score range: {all_genuine_scores.min():.3f} - {all_genuine_scores.max():.3f}")
    print(f"✓ Impostor score range: {all_impostor_scores.min():.3f} - {all_impostor_scores.max():.3f}")
    
    return all_genuine_scores, all_impostor_scores

def create_threshold_candidates(genuine_scores, impostor_scores, num_thresholds=100):
    """
    Step 2: Combine and Sort All Scores as Threshold Candidates
    """
    print(f"\nStep 2: Creating {num_thresholds} threshold candidates...")
    
    # Method 1: Use all unique scores as thresholds (as shown in the image)
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    unique_scores = np.unique(all_scores)
    
    # Method 2: Create interpolated thresholds for smoother curve
    min_score = all_scores.min()
    max_score = all_scores.max()
    interpolated_thresholds = np.linspace(min_score, max_score, num_thresholds)
    
    # Combine both methods and sort in descending order (high = match)
    combined_thresholds = np.unique(np.concatenate([unique_scores, interpolated_thresholds]))
    thresholds = np.sort(combined_thresholds)[::-1]  # Descending order
    
    print(f"✓ Created {len(thresholds)} threshold candidates")
    print(f"✓ Threshold range: {thresholds.min():.3f} - {thresholds.max():.3f}")
    
    return thresholds

def calculate_far_tar_for_thresholds(genuine_scores, impostor_scores, thresholds):
    """
    Step 3: For Each Threshold - Calculate FAR and TAR
    """
    print(f"\nStep 3: Calculating FAR and TAR for {len(thresholds)} thresholds...")
    
    results = []
    
    for i, threshold in enumerate(thresholds):
        # TAR (True Acceptance Rate) = (Number of genuine scores >= threshold) / (Total genuine scores)
        genuine_above_threshold = np.sum(genuine_scores >= threshold)
        tar = genuine_above_threshold / len(genuine_scores)
        
        # FAR (False Acceptance Rate) = (Number of impostor scores >= threshold) / (Total impostor scores)
        impostor_above_threshold = np.sum(impostor_scores >= threshold)
        far = impostor_above_threshold / len(impostor_scores)
        
        results.append({
            'threshold': threshold,
            'tar': tar,
            'far': far,
            'genuine_above': genuine_above_threshold,
            'impostor_above': impostor_above_threshold,
            'total_genuine': len(genuine_scores),
            'total_impostor': len(impostor_scores)
        })
        
        # Progress indicator
        if (i + 1) % 20 == 0 or i == len(thresholds) - 1:
            print(f"   Processed {i + 1}/{len(thresholds)} thresholds...")
    
    results_df = pd.DataFrame(results)
    
    print(f"✓ Calculated FAR/TAR for all thresholds")
    print(f"✓ TAR range: {results_df['tar'].min():.6f} - {results_df['tar'].max():.6f}")
    print(f"✓ FAR range: {results_df['far'].min():.6f} - {results_df['far'].max():.6f}")
    
    return results_df

def find_optimal_thresholds(results_df):
    """
    Find optimal thresholds for different operating points
    """
    print(f"\nFinding optimal thresholds for different operating points...")
    
    # Find threshold that minimizes FAR while maximizing TAR (Youden's J statistic)
    results_df['youden_j'] = results_df['tar'] - results_df['far']
    optimal_youden_idx = results_df['youden_j'].idxmax()
    
    # Find threshold for specific FAR targets
    far_targets = [0.001, 0.01, 0.05, 0.1]  # 0.1%, 1%, 5%, 10%
    optimal_thresholds = {}
    
    optimal_thresholds['youden'] = {
        'threshold': results_df.loc[optimal_youden_idx, 'threshold'],
        'tar': results_df.loc[optimal_youden_idx, 'tar'],
        'far': results_df.loc[optimal_youden_idx, 'far'],
        'youden_j': results_df.loc[optimal_youden_idx, 'youden_j']
    }
    
    for far_target in far_targets:
        # Find threshold closest to target FAR
        idx = (results_df['far'] - far_target).abs().idxmin()
        optimal_thresholds[f'far_{far_target}'] = {
            'threshold': results_df.loc[idx, 'threshold'],
            'tar': results_df.loc[idx, 'tar'],
            'far': results_df.loc[idx, 'far']
        }
    
    return optimal_thresholds

def generate_comprehensive_report(results_df, optimal_thresholds, genuine_scores, impostor_scores):
    """
    Generate comprehensive threshold analysis report with detailed tables
    """
    print(f"\n" + "="*80)
    print("THRESHOLD ANALYSIS REPORT")
    print("="*80)
    
    print(f"\n1. DATA SUMMARY:")
    print("-" * 50)
    print(f"   Total Genuine Scores:     {len(genuine_scores):,}")
    print(f"   Total Impostor Scores:    {len(impostor_scores):,}")
    print(f"   Total Thresholds Tested:  {len(results_df):,}")
    print(f"   Genuine Score Range:      {genuine_scores.min():.3f} - {genuine_scores.max():.3f}")
    print(f"   Impostor Score Range:     {impostor_scores.min():.3f} - {impostor_scores.max():.3f}")
    
    print(f"\n2. OPTIMAL THRESHOLDS:")
    print("-" * 50)
    
    # Youden's J optimal
    youden = optimal_thresholds['youden']
    print(f"   Youden's J Optimal (Best Balance):")
    print(f"   - Threshold: {youden['threshold']:.6f}")
    print(f"   - TAR:       {youden['tar']:.6f} ({youden['tar']*100:.2f}%)")
    print(f"   - FAR:       {youden['far']:.6f} ({youden['far']*100:.4f}%)")
    print(f"   - Youden J:  {youden['youden_j']:.6f}")
    
    print(f"\n   FAR-Targeted Thresholds:")
    for key, point in optimal_thresholds.items():
        if key.startswith('far_'):
            far_val = float(key.split('_')[1])
            print(f"   - FAR ≤ {far_val:.1%}:")
            print(f"     Threshold: {point['threshold']:.6f}")
            print(f"     TAR:       {point['tar']:.6f} ({point['tar']*100:.2f}%)")
            print(f"     FAR:       {point['far']:.6f} ({point['far']*100:.4f}%)")
    
    print(f"\n3. PERFORMANCE ANALYSIS:")
    print("-" * 50)
    
    # Calculate AUC
    auc_score = auc(results_df['far'], results_df['tar'])
    print(f"   ROC AUC Score:            {auc_score:.6f}")
    
    # Find EER (Equal Error Rate)
    eer_idx = (results_df['far'] - (1 - results_df['tar'])).abs().idxmin()
    eer = results_df.loc[eer_idx, 'far']
    eer_threshold = results_df.loc[eer_idx, 'threshold']
    print(f"   Equal Error Rate (EER):   {eer:.6f} ({eer*100:.4f}%)")
    print(f"   EER Threshold:            {eer_threshold:.6f}")
    
    # Create comprehensive tables
    print(f"\n4. COMPLETE THRESHOLDS TABLE:")
    print("-" * 80)
    print(f"   {'#':<4} {'Threshold':<12} {'FAR':<12} {'TAR':<12} {'Genuine>=T':<10} {'Impostor>=T':<12}")
    print(f"   {'-'*4} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*12}")
    
    # Show all thresholds (limit to reasonable number for display)
    display_df = results_df[::max(1, len(results_df)//50)]  # Show up to 50 rows
    for idx, row in display_df.iterrows():
        print(f"   {idx+1:<4} {row['threshold']:<12.6f} {row['far']:<12.6f} {row['tar']:<12.6f} "
              f"{row['genuine_above']:<10} {row['impostor_above']:<12}")
    
    print(f"\n5. OPTIMAL THRESHOLDS SUMMARY TABLE:")
    print("-" * 80)
    print(f"   {'Metric':<20} {'Threshold':<12} {'FAR':<12} {'TAR':<12} {'Notes':<20}")
    print(f"   {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*20}")
    
    # Add optimal thresholds to table
    print(f"   {'Youden J Optimal':<20} {youden['threshold']:<12.6f} {youden['far']:<12.6f} "
          f"{youden['tar']:<12.6f} {'Best Balance':<20}")
    
    for key, point in optimal_thresholds.items():
        if key.startswith('far_'):
            far_val = float(key.split('_')[1])
            print(f"   {f'FAR ≤ {far_val:.1%}':<20} {point['threshold']:<12.6f} {point['far']:<12.6f} "
                  f"{point['tar']:<12.6f} {'FAR Target':<20}")
    
    print(f"\n6. EXPORT DATA:")
    print("-" * 50)
    # Save results as CSV for easy analysis
    results_df.to_csv('thresholds_analysis.csv', index=False)
    print(f"✓ Complete thresholds table saved as 'thresholds_analysis.csv'")
    
    # Save optimal thresholds as separate CSV
    optimal_data = []
    for key, value in optimal_thresholds.items():
        optimal_data.append({
            'threshold_type': key,
            'threshold': value['threshold'],
            'tar': value['tar'],
            'far': value['far'],
            'youden_j': value.get('youden_j', None)
        })
    
    optimal_df = pd.DataFrame(optimal_data)
    optimal_df.to_csv('optimal_thresholds.csv', index=False)
    print(f"✓ Optimal thresholds saved as 'optimal_thresholds.csv'")
    
    
    print(f"\n" + "="*80)
    print("THRESHOLD ANALYSIS COMPLETE!")
    print("="*80)
    
    return results_df, optimal_thresholds

def create_comprehensive_visualizations(results_df, optimal_thresholds, genuine_scores, impostor_scores):
    """
    Create comprehensive ROC curve and analysis visualizations
    """
    print(f"\n7. CREATING VISUALIZATIONS:")
    print("-" * 50)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. ROC Curve
    ax1 = plt.subplot(2, 3, 1)
    fpr, tpr, _ = roc_curve(np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))]), 
                           np.concatenate([genuine_scores, impostor_scores]))
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    # Mark optimal thresholds
    youden_threshold = optimal_thresholds['youden']['threshold']
    youden_fpr = optimal_thresholds['youden']['far']
    youden_tpr = optimal_thresholds['youden']['tar']
    plt.plot(youden_fpr, youden_tpr, 'ro', markersize=10, label=f"Youden's J (T={youden_threshold:.3f})")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (TAR)')
    plt.title('ROC Curve - Combined Dataset Analysis')
    plt.legend(loc="lower right", fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # 2. FAR vs Threshold
    ax2 = plt.subplot(2, 3, 2)
    plt.plot(results_df['threshold'], results_df['far'], 'b-', linewidth=2, label='FAR')
    plt.axhline(y=0.001, color='r', linestyle='--', alpha=0.7, label='FAR = 0.1%')
    plt.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='FAR = 1%')
    plt.axhline(y=0.05, color='g', linestyle='--', alpha=0.7, label='FAR = 5%')
    plt.xlabel('Threshold')
    plt.ylabel('False Acceptance Rate (FAR)')
    plt.title('FAR vs Threshold')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. TAR vs Threshold
    ax3 = plt.subplot(2, 3, 3)
    plt.plot(results_df['threshold'], results_df['tar'], 'g-', linewidth=2, label='TAR')
    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='TAR = 95%')
    plt.axhline(y=0.99, color='orange', linestyle='--', alpha=0.7, label='TAR = 99%')
    plt.xlabel('Threshold')
    plt.ylabel('True Acceptance Rate (TAR)')
    plt.title('TAR vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Score Distributions
    ax4 = plt.subplot(2, 3, 4)
    plt.hist(genuine_scores, bins=30, alpha=0.7, label='Genuine Scores', color='green', density=True)
    plt.hist(impostor_scores, bins=30, alpha=0.7, label='Impostor Scores', color='red', density=True)
    
    # Mark optimal thresholds
    plt.axvline(x=youden_threshold, color='black', linestyle='-', linewidth=2, label=f"Youden's J (T={youden_threshold:.3f})")
    
    plt.xlabel('Similarity Score')
    plt.ylabel('Density')
    plt.title('Score Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Performance Metrics Heatmap
    ax5 = plt.subplot(2, 3, 5)
    
    # Create performance matrix
    thresholds_sample = results_df['threshold'][::10]  # Sample every 10th threshold
    far_values = results_df['far'][::10]
    tar_values = results_df['tar'][::10]
    
    # Create a 2D grid for heatmap
    threshold_grid = np.linspace(results_df['threshold'].min(), results_df['threshold'].max(), 20)
    far_grid = np.linspace(0, 1, 20)
    performance_grid = np.zeros((20, 20))
    
    for i, t in enumerate(threshold_grid):
        for j, f in enumerate(far_grid):
            # Find closest TAR for this threshold and FAR combination
            closest_idx = np.argmin(np.abs(results_df['threshold'] - t))
            if abs(results_df.iloc[closest_idx]['far'] - f) < 0.1:  # Within 10%
                performance_grid[j, i] = results_df.iloc[closest_idx]['tar']
    
    im = plt.imshow(performance_grid, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(im, label='TAR')
    plt.xlabel('Threshold (Normalized)')
    plt.ylabel('FAR (Normalized)')
    plt.title('Performance Heatmap (TAR)')
    
    # 6. Threshold Analysis Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary text
    summary_text = f"""
THRESHOLD ANALYSIS SUMMARY

Dataset Information:
• Multi-user Results: 10,000 samples
• Recognition Results: 500 samples  
• Total Genuine: {len(genuine_scores)} samples
• Total Impostor: {len(impostor_scores)} samples

Performance Metrics:
• ROC AUC Score: {roc_auc:.4f}
• Best Balance (Youden's J): {youden_threshold:.3f}
• TAR at Youden's J: {youden_tpr:.1%}
• FAR at Youden's J: {youden_fpr:.1%}

Optimal Thresholds:
• FAR ≤ 0.1%: T = {optimal_thresholds['far_0.001']['threshold']:.3f}
• FAR ≤ 1.0%: T = {optimal_thresholds['far_0.01']['threshold']:.3f}
• FAR ≤ 5.0%: T = {optimal_thresholds['far_0.05']['threshold']:.3f}

Score Ranges:
• Genuine: {genuine_scores.min():.3f} - {genuine_scores.max():.3f}
• Impostor: {impostor_scores.min():.3f} - {impostor_scores.max():.3f}
    """
    
    plt.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('comprehensive_roc_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Comprehensive ROC analysis saved as 'comprehensive_roc_analysis.png'")
    
    # Create additional detailed ROC curve
    plt.figure(figsize=(12, 8))
    
    # Main ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'Combined Dataset ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.500)')
    
    # Mark key operating points
    colors = ['red', 'blue', 'green', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (key, point) in enumerate(optimal_thresholds.items()):
        if i < len(colors):
            plt.plot(point['far'], point['tar'], color=colors[i], marker=markers[i], 
                    markersize=8, label=f"{key.replace('_', ' ').title()} (T={point['threshold']:.3f})")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)', fontsize=12)
    plt.ylabel('True Positive Rate (TAR)', fontsize=12)
    plt.title('ROC Curve - Voice Recognition System Performance\n(Combined Multi-User + Recognition Results)', fontsize=14)
    plt.legend(loc="lower right", fontsize=9, framealpha=0.9, ncol=2, columnspacing=0.5, handletextpad=0.3)
    plt.grid(True, alpha=0.3)
    
    # Add performance annotations
    plt.text(0.6, 0.2, f'Dataset: {len(genuine_scores)} genuine, {len(impostor_scores)} impostor\n'
                       f'Score Range: {genuine_scores.min():.3f} - {genuine_scores.max():.3f}',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig('detailed_roc_curve.png', dpi=300, bbox_inches='tight')
    print("✓ Detailed ROC curve saved as 'detailed_roc_curve.png'")
    
    # Create threshold performance plot
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: FAR vs Threshold
    plt.subplot(1, 3, 1)
    plt.plot(results_df['threshold'], results_df['far'], 'b-', linewidth=2)
    plt.axhline(y=0.001, color='r', linestyle='--', alpha=0.7, label='FAR = 0.1%')
    plt.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='FAR = 1%')
    plt.xlabel('Threshold')
    plt.ylabel('False Acceptance Rate')
    plt.title('FAR vs Threshold')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: TAR vs Threshold  
    plt.subplot(1, 3, 2)
    plt.plot(results_df['threshold'], results_df['tar'], 'g-', linewidth=2)
    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='TAR = 95%')
    plt.axhline(y=0.99, color='orange', linestyle='--', alpha=0.7, label='TAR = 99%')
    plt.xlabel('Threshold')
    plt.ylabel('True Acceptance Rate')
    plt.title('TAR vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Youden's J vs Threshold
    plt.subplot(1, 3, 3)
    youden_j_values = results_df['tar'] - results_df['far']
    plt.plot(results_df['threshold'], youden_j_values, 'purple', linewidth=2)
    plt.axvline(x=youden_threshold, color='red', linestyle='--', alpha=0.7, 
                label=f"Optimal T={youden_threshold:.3f}")
    plt.xlabel('Threshold')
    plt.ylabel("Youden's J (TAR - FAR)")
    plt.title("Youden's J vs Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('threshold_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Threshold performance analysis saved as 'threshold_performance_analysis.png'")
    
    plt.close('all')  # Close all figures to free memory
    
    return roc_auc

def main():
    """
    Main function to run the complete threshold analysis
    """
    # Step 1: Collect Scores
    genuine_scores, impostor_scores = load_and_prepare_data()
    
    # Step 2: Create Threshold Candidates
    thresholds = create_threshold_candidates(genuine_scores, impostor_scores, num_thresholds=100)
    
    # Step 3: Calculate FAR and TAR for Each Threshold
    results_df = calculate_far_tar_for_thresholds(genuine_scores, impostor_scores, thresholds)
    
    # Find Optimal Thresholds
    optimal_thresholds = find_optimal_thresholds(results_df)
    
    # Generate Report
    results_df, optimal_thresholds = generate_comprehensive_report(
        results_df, optimal_thresholds, genuine_scores, impostor_scores)
    
    # Create comprehensive visualizations
    roc_auc = create_comprehensive_visualizations(results_df, optimal_thresholds, genuine_scores, impostor_scores)
    
    print(f"\n" + "="*80)
    print("COMPLETE ANALYSIS FINISHED!")
    print("="*80)
    print(f"✓ ROC AUC Score: {roc_auc:.4f}")
    print(f"✓ Generated 3 visualization files:")
    print(f"  - comprehensive_roc_analysis.png")
    print(f"  - detailed_roc_curve.png") 
    print(f"  - threshold_performance_analysis.png")
    print(f"✓ Generated 2 CSV files:")
    print(f"  - thresholds_analysis.csv")
    print(f"  - optimal_thresholds.csv")
    print("="*80)
    
    return results_df, optimal_thresholds

if __name__ == "__main__":
    results = main()
