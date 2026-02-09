"""
Statistical Analysis and Correlations - EPL 2020-21
Advanced statistical methods, correlations, and predictive insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from config import *

class StatisticalAnalyzer:
    """Perform advanced statistical analysis on EPL data"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Filter players with significant playing time for statistical validity
        self.df_qualified = df[df['Mins'] >= 500].copy()
    
    def correlation_analysis(self):
        """Analyze correlations between different performance metrics"""
        print_section_header("CORRELATION ANALYSIS")
        
        # Select numeric columns for correlation
        numeric_cols = ['Goals', 'Assists', 'xG', 'xA', 'Passes_Attempted',
                       'Perc_Passes_Completed', 'Age', 'Mins', 'Yellow_Cards',
                       'Goals+Assists', 'G+A_Per90', 'MinsPerMatch']
        
        corr_matrix = self.df_qualified[numeric_cols].corr()
        
        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_LARGE)
        fig.suptitle('Performance Metrics Correlation Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Full correlation heatmap
        ax1 = axes[0]
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, ax=ax1,
                   cbar_kws={"shrink": 0.8}, linewidths=0.5,
                   vmin=-1, vmax=1, annot_kws={'size': 8})
        ax1.set_title('Correlation Matrix', fontsize=12, fontweight='bold')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=9)
        
        # 2. Top correlations visualization
        ax2 = axes[1]
        
        # Get top positive and negative correlations (excluding self-correlations)
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs).sort_values('correlation', key=abs, ascending=False).head(15)
        
        colors = ['green' if x > 0 else 'red' for x in corr_df['correlation']]
        bars = ax2.barh(range(len(corr_df)), corr_df['correlation'], color=colors, alpha=0.7)
        ax2.set_yticks(range(len(corr_df)))
        ax2.set_yticklabels([f"{row['var1']} ↔ {row['var2']}" 
                            for _, row in corr_df.iterrows()], fontsize=8)
        ax2.set_xlabel('Correlation Coefficient', fontsize=11)
        ax2.set_title('Top 15 Variable Correlations', fontsize=12, fontweight='bold')
        ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
        
        # Add values
        for i, (bar, val) in enumerate(zip(bars, corr_df['correlation'])):
            ax2.text(val, i, f' {val:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        PlotUtils.save_figure('correlation_analysis.png')
        plt.show()
        
        print("\nStrongest Positive Correlations:")
        print(corr_df[corr_df['correlation'] > 0].head(5))
        
        print("\nStrongest Negative Correlations:")
        print(corr_df[corr_df['correlation'] < 0].head(5))
    
    def regression_analysis(self):
        """Perform regression analysis to predict goals from various factors"""
        print_section_header("REGRESSION ANALYSIS")
        
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_LARGE)
        fig.suptitle('Regression Analysis: Predicting Goals', fontsize=16, fontweight='bold', y=1.00)
        
        # 1. xG vs Actual Goals
        ax1 = axes[0, 0]
        X = self.df_qualified[['xG']].values
        y = self.df_qualified['Goals'].values
        
        model1 = LinearRegression()
        model1.fit(X, y)
        y_pred = model1.predict(X)
        
        ax1.scatter(X, y, alpha=0.5, s=50, color='steelblue', label='Actual')
        ax1.plot(X, y_pred, color='red', linewidth=2, label=f'Regression Line')
        ax1.set_xlabel('Expected Goals (xG)', fontsize=11)
        ax1.set_ylabel('Actual Goals', fontsize=11)
        ax1.set_title(f'xG vs Goals (R²={model1.score(X, y):.3f})', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Add ideal line
        max_val = max(X.max(), y.max())
        ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Ideal (1:1)')
        
        # 2. Multiple features vs Goals
        ax2 = axes[0, 1]
        features = ['xG', 'Passes_Attempted', 'Mins']
        X_multi = self.df_qualified[features].values
        y_multi = self.df_qualified['Goals'].values
        
        model2 = LinearRegression()
        model2.fit(X_multi, y_multi)
        y_pred_multi = model2.predict(X_multi)
        
        ax2.scatter(y_multi, y_pred_multi, alpha=0.5, s=50, color='darkorange')
        ax2.plot([y_multi.min(), y_multi.max()], [y_multi.min(), y_multi.max()], 
                'k--', linewidth=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Goals', fontsize=11)
        ax2.set_ylabel('Predicted Goals', fontsize=11)
        ax2.set_title(f'Multi-Feature Model (R²={model2.score(X_multi, y_multi):.3f})', 
                     fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Assists vs xA
        ax3 = axes[1, 0]
        X_assists = self.df_qualified[['xA']].values
        y_assists = self.df_qualified['Assists'].values
        
        model3 = LinearRegression()
        model3.fit(X_assists, y_assists)
        y_pred_assists = model3.predict(X_assists)
        
        ax3.scatter(X_assists, y_assists, alpha=0.5, s=50, color='green', label='Actual')
        ax3.plot(X_assists, y_pred_assists, color='red', linewidth=2, label='Regression Line')
        ax3.set_xlabel('Expected Assists (xA)', fontsize=11)
        ax3.set_ylabel('Actual Assists', fontsize=11)
        ax3.set_title(f'xA vs Assists (R²={model3.score(X_assists, y_assists):.3f})', 
                     fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Feature Importance
        ax4 = axes[1, 1]
        feature_names = features
        coefficients = np.abs(model2.coef_)
        
        # Normalize coefficients for comparison
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_multi)
        model_scaled = LinearRegression()
        model_scaled.fit(X_scaled, y_multi)
        normalized_coef = np.abs(model_scaled.coef_)
        
        colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
        bars = ax4.bar(range(len(feature_names)), normalized_coef, color=colors_bar, alpha=0.8)
        ax4.set_xticks(range(len(feature_names)))
        ax4.set_xticklabels(feature_names, rotation=0, fontsize=10)
        ax4.set_ylabel('Normalized Coefficient (Importance)', fontsize=11)
        ax4.set_title('Feature Importance for Goals Prediction', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for bar, val in zip(bars, normalized_coef):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        PlotUtils.save_figure('regression_analysis.png')
        plt.show()
        
        print("\nRegression Model Performance:")
        print(f"xG → Goals R² Score: {model1.score(X, y):.4f}")
        print(f"Multi-feature → Goals R² Score: {model2.score(X_multi, y_multi):.4f}")
        print(f"xA → Assists R² Score: {model3.score(X_assists, y_assists):.4f}")
        
        print("\nFeature Coefficients (Multi-feature model):")
        for name, coef in zip(feature_names, model2.coef_):
            print(f"{name}: {coef:.4f}")
    
    def distribution_analysis(self):
        """Analyze distributions of key metrics"""
        print_section_header("DISTRIBUTION ANALYSIS")
        
        fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE_LARGE)
        fig.suptitle('Distribution of Key Performance Metrics', fontsize=16, fontweight='bold', y=1.00)
        
        metrics = [
            ('Goals', 'Goals Distribution'),
            ('Assists', 'Assists Distribution'),
            ('G+A_Per90', 'G+A per 90 Distribution'),
            ('Perc_Passes_Completed', 'Pass Completion % Distribution'),
            ('Age', 'Age Distribution'),
            ('MinsPerMatch', 'Minutes per Match Distribution')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            data = self.df_qualified[metric].dropna()
            
            # Histogram with KDE
            ax.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
            
            # Fit and plot normal distribution
            mu, std = stats.norm.fit(data)
            x = np.linspace(data.min(), data.max(), 100)
            ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, 
                   label=f'Normal\nμ={mu:.2f}\nσ={std:.2f}')
            
            # KDE
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            ax.plot(x, kde(x), 'g--', linewidth=2, label='KDE')
            
            ax.set_xlabel(metric, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            
            # Add mean and median lines
            ax.axvline(data.mean(), color='blue', linestyle='--', alpha=0.5, 
                      linewidth=1.5, label=f'Mean: {data.mean():.2f}')
            ax.axvline(data.median(), color='orange', linestyle='--', alpha=0.5, 
                      linewidth=1.5, label=f'Median: {data.median():.2f}')
        
        plt.tight_layout()
        PlotUtils.save_figure('distribution_analysis.png')
        plt.show()
        
        # Statistical tests
        print("\nNormality Tests (Shapiro-Wilk):")
        for metric, _ in metrics:
            data = self.df_qualified[metric].dropna()
            statistic, p_value = stats.shapiro(data)
            is_normal = "Normal" if p_value > 0.05 else "Not Normal"
            print(f"{metric}: p-value = {p_value:.4f} ({is_normal})")
    
    def outlier_analysis(self):
        """Identify and analyze outliers in performance metrics"""
        print_section_header("OUTLIER ANALYSIS")
        
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_LARGE)
        fig.suptitle('Outlier Analysis', fontsize=16, fontweight='bold', y=1.00)
        
        metrics = ['Goals', 'Assists', 'G+A_Per90', 'Perc_Passes_Completed']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Box plot
            bp = ax.boxplot(self.df_qualified[metric], vert=True, patch_artist=True,
                           widths=0.5, showmeans=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2),
                           meanprops=dict(marker='D', markerfacecolor='green', markersize=8))
            
            # Calculate IQR and outliers
            Q1 = self.df_qualified[metric].quantile(0.25)
            Q3 = self.df_qualified[metric].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df_qualified[
                (self.df_qualified[metric] < lower_bound) | 
                (self.df_qualified[metric] > upper_bound)
            ]
            
            # Annotate outliers
            for _, player in outliers.iterrows():
                if player[metric] > upper_bound:  # Only annotate upper outliers
                    ax.annotate(player['Name'], 
                              xy=(1, player[metric]), 
                              xytext=(1.15, player[metric]),
                              fontsize=8, alpha=0.7,
                              arrowprops=dict(arrowstyle='->', alpha=0.5))
            
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(f'{metric} - Outliers Highlighted', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels([''])
            
            # Add statistics text
            stats_text = f'Q1: {Q1:.2f}\nMedian: {self.df_qualified[metric].median():.2f}\nQ3: {Q3:.2f}\nOutliers: {len(outliers)}'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        PlotUtils.save_figure('outlier_analysis.png')
        plt.show()
        
        print("\nOutlier Summary:")
        for metric in metrics:
            Q1 = self.df_qualified[metric].quantile(0.25)
            Q3 = self.df_qualified[metric].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df_qualified[
                (self.df_qualified[metric] < lower_bound) | 
                (self.df_qualified[metric] > upper_bound)
            ]
            
            print(f"\n{metric}: {len(outliers)} outliers detected")
            if len(outliers) > 0:
                print("Top outliers:")
                print(outliers.nlargest(3, metric)[['Name', 'Club', metric]])
    
    def comparative_statistical_tests(self):
        """Perform statistical tests comparing different groups"""
        print_section_header("COMPARATIVE STATISTICAL TESTS")
        
        # Compare performance across age groups
        age_groups = {
            'Young (≤23)': self.df_qualified[self.df_qualified['Age'] <= 23],
            'Prime (24-29)': self.df_qualified[(self.df_qualified['Age'] >= 24) & 
                                              (self.df_qualified['Age'] <= 29)],
            'Experienced (≥30)': self.df_qualified[self.df_qualified['Age'] >= 30]
        }
        
        print("\n1. Age Group Comparisons (ANOVA):")
        print("-" * 50)
        
        metrics_test = ['Goals', 'Assists', 'G+A_Per90', 'Perc_Passes_Completed']
        
        for metric in metrics_test:
            groups_data = [group[metric].dropna() for group in age_groups.values()]
            f_stat, p_value = stats.f_oneway(*groups_data)
            
            print(f"\n{metric}:")
            print(f"  F-statistic: {f_stat:.4f}")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
            
            # Print means by group
            for group_name, group_data in age_groups.items():
                print(f"  {group_name} mean: {group_data[metric].mean():.3f}")
        
        # Compare positions
        print("\n\n2. Position Comparisons (T-tests):")
        print("-" * 50)
        
        forwards = self.df_qualified[self.df_qualified['Primary_Position'] == 'FW']
        midfielders = self.df_qualified[self.df_qualified['Primary_Position'] == 'MF']
        
        comparisons = [
            ('Goals', 'FW vs MF'),
            ('Assists', 'FW vs MF'),
            ('Perc_Passes_Completed', 'FW vs MF')
        ]
        
        for metric, comparison in comparisons:
            fw_data = forwards[metric].dropna()
            mf_data = midfielders[metric].dropna()
            
            t_stat, p_value = stats.ttest_ind(fw_data, mf_data)
            
            print(f"\n{metric} ({comparison}):")
            print(f"  T-statistic: {t_stat:.4f}")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
            print(f"  FW mean: {fw_data.mean():.3f}")
            print(f"  MF mean: {mf_data.mean():.3f}")
        
        # Visualization of comparisons
        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
        fig.suptitle('Statistical Comparisons', fontsize=16, fontweight='bold')
        
        # Age group comparison
        ax1 = axes[0]
        age_group_names = list(age_groups.keys())
        goals_by_age = [group['G+A_Per90'].mean() for group in age_groups.values()]
        errors = [group['G+A_Per90'].std() for group in age_groups.values()]
        
        bars = ax1.bar(range(len(age_group_names)), goals_by_age, 
                      yerr=errors, capsize=10, alpha=0.7,
                      color=plt.cm.viridis(np.linspace(0.3, 0.9, len(age_group_names))))
        ax1.set_xticks(range(len(age_group_names)))
        ax1.set_xticklabels(age_group_names, fontsize=10)
        ax1.set_ylabel('Average G+A per 90', fontsize=11)
        ax1.set_title('Performance by Age Group', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Position comparison
        ax2 = axes[1]
        position_data = [
            forwards['G+A_Per90'].dropna(),
            midfielders['G+A_Per90'].dropna(),
            self.df_qualified[self.df_qualified['Primary_Position'] == 'DF']['G+A_Per90'].dropna()
        ]
        
        bp = ax2.boxplot(position_data, labels=['FW', 'MF', 'DF'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['red', 'blue', 'green']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax2.set_ylabel('G+A per 90', fontsize=11)
        ax2.set_title('Performance by Position', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        PlotUtils.save_figure('statistical_comparisons.png')
        plt.show()

def main():
    """Main execution function"""
    loader = DataLoader()
    df = loader.load_data()
    
    if df is not None:
        df = loader.create_derived_features(df)
        
        analyzer = StatisticalAnalyzer(df)
        
        analyzer.correlation_analysis()
        analyzer.regression_analysis()
        analyzer.distribution_analysis()
        analyzer.outlier_analysis()
        analyzer.comparative_statistical_tests()
        
        print("\n" + "="*70)
        print("  STATISTICAL ANALYSIS COMPLETED")
        print("="*70)

if __name__ == "__main__":
    main()
