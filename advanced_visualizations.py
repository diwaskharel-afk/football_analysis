"""
Advanced Visualizations - EPL 2020-21
Creative and insightful visualizations using advanced plotting techniques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle, Arc
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from config import *

class AdvancedVisualizer:
    """Create advanced and creative visualizations"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def player_radar_chart(self):
        """Create radar charts for player comparison"""
        print_section_header("PLAYER RADAR CHARTS")
        
        # Select top scorers
        top_players = self.df.nlargest(5, 'Goals')
        
        # Metrics for radar (normalized)
        metrics = ['Goals', 'Assists', 'xG', 'xA', 'Perc_Passes_Completed']
        
        # Normalize metrics to 0-100 scale
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 100))
        
        normalized_data = {}
        for idx, player in top_players.iterrows():
            values = [player[m] for m in metrics]
            normalized_data[player['Name']] = scaler.fit_transform(
                np.array(values).reshape(-1, 1)
            ).flatten()
        
        # Create radar charts
        num_players = len(top_players)
        fig = plt.figure(figsize=FIGURE_SIZE_LARGE)
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        metric_labels = ['Goals', 'Assists', 'xG', 'xA', 'Pass %']
        
        colors = plt.cm.Set2(np.linspace(0, 1, num_players))
        
        ax = fig.add_subplot(111, projection='polar')
        
        for idx, (player_name, values) in enumerate(normalized_data.items()):
            values_plot = values.tolist()
            values_plot += values_plot[:1]  # Complete the circle
            
            ax.plot(angles, values_plot, 'o-', linewidth=2, 
                   label=player_name, color=colors[idx])
            ax.fill(angles, values_plot, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, size=11)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], size=9)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        plt.title('Top 5 Scorers - Performance Radar Chart', 
                 fontsize=14, fontweight='bold', y=1.08)
        
        plt.tight_layout()
        PlotUtils.save_figure('player_radar_charts.png')
        plt.show()
    
    def bubble_chart_analysis(self):
        """Create bubble charts for multi-dimensional analysis"""
        print_section_header("BUBBLE CHART ANALYSIS")
        
        # Filter qualified players
        df_qual = self.df[self.df['Mins'] >= 900].copy()
        
        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_LARGE)
        fig.suptitle('Multi-Dimensional Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Goals vs Assists (size=minutes, color=age)
        ax1 = axes[0]
        scatter1 = ax1.scatter(df_qual['Goals'], df_qual['Assists'],
                              s=df_qual['Mins']/20, alpha=0.6,
                              c=df_qual['Age'], cmap='coolwarm',
                              edgecolors='black', linewidth=0.5)
        
        ax1.set_xlabel('Goals', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Assists', fontsize=12, fontweight='bold')
        ax1.set_title('Goals vs Assists\n(Size = Minutes, Color = Age)', fontsize=12, fontweight='bold')
        ax1.grid(alpha=0.3, linestyle='--')
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Age', rotation=270, labelpad=20, fontsize=11)
        
        # Annotate top performers
        top_ga = df_qual.nlargest(8, 'Goals+Assists')
        for _, player in top_ga.iterrows():
            ax1.annotate(player['Name'], 
                        (player['Goals'], player['Assists']),
                        fontsize=8, alpha=0.7, 
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        # 2. xG vs xA (size=actual G+A, color=efficiency)
        ax2 = axes[1]
        
        df_qual['Efficiency'] = ((df_qual['Goals'] + df_qual['Assists']) / 
                                 (df_qual['xG'] + df_qual['xA']) * 100).replace([np.inf], 0)
        df_qual['Efficiency'] = df_qual['Efficiency'].clip(0, 200)  # Cap extreme values
        
        scatter2 = ax2.scatter(df_qual['xG'], df_qual['xA'],
                              s=(df_qual['Goals+Assists']*30), alpha=0.6,
                              c=df_qual['Efficiency'], cmap='RdYlGn',
                              vmin=50, vmax=150,
                              edgecolors='black', linewidth=0.5)
        
        ax2.set_xlabel('Expected Goals (xG)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Expected Assists (xA)', fontsize=12, fontweight='bold')
        ax2.set_title('Expected vs Actual Performance\n(Size = G+A, Color = Efficiency %)', 
                     fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3, linestyle='--')
        
        # Add colorbar
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Efficiency %', rotation=270, labelpad=20, fontsize=11)
        
        # Add diagonal reference line
        max_val = max(df_qual['xG'].max(), df_qual['xA'].max())
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)
        
        plt.tight_layout()
        PlotUtils.save_figure('bubble_chart_analysis.png')
        plt.show()
    
    def heatmap_visualizations(self):
        """Create various heatmap visualizations"""
        print_section_header("HEATMAP VISUALIZATIONS")
        
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_LARGE)
        fig.suptitle('Heatmap Analyses', fontsize=16, fontweight='bold', y=1.00)
        
        # 1. Club vs Position Goals Heatmap
        ax1 = axes[0, 0]
        
        # Get top 10 clubs by goals
        top_clubs = self.df.groupby('Club')['Goals'].sum().nlargest(10).index
        df_top = self.df[self.df['Club'].isin(top_clubs)]
        
        pivot_goals = df_top.pivot_table(values='Goals', 
                                         index='Primary_Position',
                                         columns='Club', 
                                         aggfunc='sum', 
                                         fill_value=0)
        
        sns.heatmap(pivot_goals, annot=True, fmt='.0f', cmap='YlOrRd', 
                   ax=ax1, cbar_kws={'label': 'Goals'}, linewidths=0.5)
        ax1.set_title('Goals by Position and Club (Top 10)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Club', fontsize=11)
        ax1.set_ylabel('Position', fontsize=11)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        
        # 2. Age Group vs Performance Heatmap
        ax2 = axes[0, 1]
        
        self.df['AgeGroup'] = pd.cut(self.df['Age'], 
                                     bins=[0, 23, 27, 31, 100],
                                     labels=['U23', '24-27', '28-31', '32+'])
        
        pivot_performance = self.df.groupby(['AgeGroup', 'Primary_Position']).agg({
            'G+A_Per90': 'mean'
        }).unstack(fill_value=0)
        
        pivot_performance.columns = pivot_performance.columns.droplevel(0)
        
        sns.heatmap(pivot_performance, annot=True, fmt='.3f', cmap='viridis',
                   ax=ax2, cbar_kws={'label': 'Avg G+A per 90'}, linewidths=0.5)
        ax2.set_title('Performance by Age and Position', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Position', fontsize=11)
        ax2.set_ylabel('Age Group', fontsize=11)
        
        # 3. Discipline Heatmap
        ax3 = axes[1, 0]
        
        pivot_discipline = df_top.pivot_table(values=['Yellow_Cards', 'Red_Cards'],
                                             index='Club',
                                             aggfunc='sum')
        
        sns.heatmap(pivot_discipline, annot=True, fmt='.0f', cmap='Reds',
                   ax=ax3, cbar_kws={'label': 'Cards'}, linewidths=0.5)
        ax3.set_title('Discipline by Club (Top 10)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Card Type', fontsize=11)
        ax3.set_ylabel('Club', fontsize=11)
        plt.setp(ax3.get_yticklabels(), rotation=0, fontsize=9)
        
        # 4. Pass Completion by Position and Club
        ax4 = axes[1, 1]
        
        pivot_passing = df_top.pivot_table(values='Perc_Passes_Completed',
                                          index='Primary_Position',
                                          columns='Club',
                                          aggfunc='mean',
                                          fill_value=0)
        
        sns.heatmap(pivot_passing, annot=True, fmt='.1f', cmap='Blues',
                   ax=ax4, cbar_kws={'label': 'Pass %'}, linewidths=0.5,
                   vmin=70, vmax=90)
        ax4.set_title('Passing Accuracy by Position (Top 10 Clubs)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Club', fontsize=11)
        ax4.set_ylabel('Position', fontsize=11)
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        
        plt.tight_layout()
        PlotUtils.save_figure('heatmap_visualizations.png')
        plt.show()
    
    def advanced_scatter_matrix(self):
        """Create advanced scatter matrix with distributions"""
        print_section_header("SCATTER MATRIX ANALYSIS")
        
        # Select key metrics
        df_qual = self.df[self.df['Mins'] >= 900].copy()
        metrics = ['Goals', 'Assists', 'xG', 'Perc_Passes_Completed', 'Age']
        
        # Create pairplot with enhanced styling
        g = sns.pairplot(df_qual[metrics + ['Primary_Position']], 
                        hue='Primary_Position',
                        palette='Set2',
                        diag_kind='kde',
                        plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'black', 'linewidth': 0.5},
                        diag_kws={'alpha': 0.7, 'linewidth': 2})
        
        g.fig.suptitle('Performance Metrics Scatter Matrix by Position', 
                      fontsize=16, fontweight='bold', y=1.02)
        
        # Adjust layout
        plt.tight_layout()
        PlotUtils.save_figure('scatter_matrix_analysis.png')
        plt.show()
    
    def circular_bar_plots(self):
        """Create circular/radial bar plots for rankings"""
        print_section_header("CIRCULAR BAR PLOTS")
        
        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_LARGE, subplot_kw=dict(projection='polar'))
        fig.suptitle('Circular Rankings', fontsize=16, fontweight='bold')
        
        # 1. Top goal scorers circular plot
        ax1 = axes[0]
        
        top_scorers = self.df.nlargest(15, 'Goals')
        values = top_scorers['Goals'].values
        labels = top_scorers['Name'].values
        
        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
        
        # Create bars
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(values)))
        bars = ax1.bar(angles, values, color=colors, alpha=0.8, width=0.4, edgecolor='black')
        
        # Customize
        ax1.set_xticks(angles)
        ax1.set_xticklabels(labels, size=8)
        ax1.set_ylim(0, max(values) * 1.1)
        ax1.set_theta_offset(np.pi / 2)
        ax1.set_theta_direction(-1)
        ax1.set_title('Top 15 Goal Scorers', fontsize=12, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for angle, value, bar in zip(angles, values, bars):
            rotation = np.rad2deg(angle)
            alignment = "right" if angle > np.pi else "left"
            ax1.text(angle, value + 1, str(int(value)), 
                    ha=alignment, va='center', fontsize=8, fontweight='bold')
        
        # 2. Top assist providers circular plot
        ax2 = axes[1]
        
        top_assists = self.df.nlargest(15, 'Assists')
        values2 = top_assists['Assists'].values
        labels2 = top_assists['Name'].values
        
        angles2 = np.linspace(0, 2 * np.pi, len(values2), endpoint=False).tolist()
        
        colors2 = plt.cm.Greens(np.linspace(0.4, 0.9, len(values2)))
        bars2 = ax2.bar(angles2, values2, color=colors2, alpha=0.8, width=0.4, edgecolor='black')
        
        ax2.set_xticks(angles2)
        ax2.set_xticklabels(labels2, size=8)
        ax2.set_ylim(0, max(values2) * 1.1)
        ax2.set_theta_offset(np.pi / 2)
        ax2.set_theta_direction(-1)
        ax2.set_title('Top 15 Assist Providers', fontsize=12, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3)
        
        for angle, value, bar in zip(angles2, values2, bars2):
            rotation = np.rad2deg(angle)
            alignment = "right" if angle > np.pi else "left"
            ax2.text(angle, value + 0.5, str(int(value)), 
                    ha=alignment, va='center', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        PlotUtils.save_figure('circular_bar_plots.png')
        plt.show()
    
    def advanced_comparison_plots(self):
        """Create advanced comparison visualizations"""
        print_section_header("ADVANCED COMPARISON PLOTS")
        
        fig = plt.figure(figsize=FIGURE_SIZE_LARGE)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Advanced Performance Comparisons', fontsize=16, fontweight='bold')
        
        # 1. Lollipop chart - Top performers
        ax1 = fig.add_subplot(gs[0, 0])
        
        top_perf = self.df.nlargest(15, 'Goals+Assists')
        y_pos = np.arange(len(top_perf))
        
        ax1.hlines(y=y_pos, xmin=0, xmax=top_perf['Goals+Assists'], 
                  color='steelblue', alpha=0.7, linewidth=2)
        ax1.plot(top_perf['Goals+Assists'], y_pos, "o", 
                markersize=10, color='darkblue', alpha=0.8)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_perf['Name'], fontsize=9)
        ax1.set_xlabel('Goals + Assists', fontsize=11, fontweight='bold')
        ax1.set_title('Top 15 Total Goal Contributions', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        ax1.invert_yaxis()
        
        # Add values
        for i, (value, name) in enumerate(zip(top_perf['Goals+Assists'], top_perf['Name'])):
            ax1.text(value + 0.5, i, str(int(value)), 
                    va='center', fontsize=9, fontweight='bold')
        
        # 2. Diverging bar chart - Over/underperformance vs xG
        ax2 = fig.add_subplot(gs[0, 1])
        
        df_qual = self.df[self.df['Mins'] >= 1000].copy()
        df_qual['xG_diff'] = df_qual['Goals'] - df_qual['xG']
        
        top_over = df_qual.nlargest(8, 'xG_diff')
        top_under = df_qual.nsmallest(8, 'xG_diff')
        combined = pd.concat([top_over, top_under]).sort_values('xG_diff')
        
        colors_div = ['red' if x < 0 else 'green' for x in combined['xG_diff']]
        y_pos2 = np.arange(len(combined))
        
        ax2.barh(y_pos2, combined['xG_diff'], color=colors_div, alpha=0.7)
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(combined['Name'], fontsize=8)
        ax2.set_xlabel('Goals - xG', fontsize=11, fontweight='bold')
        ax2.set_title('Top Over/Underperformers vs xG', fontsize=12, fontweight='bold')
        ax2.axvline(0, color='black', linewidth=1)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 3. Violin plot - Performance distribution by age group
        ax3 = fig.add_subplot(gs[1, :])
        
        age_groups_data = []
        age_labels = []
        for age_range, label in [((16, 21), 'U21'), ((21, 25), '21-25'), 
                                 ((25, 29), '25-29'), ((29, 40), '29+')]:
            group_data = self.df[
                (self.df['Age'] >= age_range[0]) & 
                (self.df['Age'] < age_range[1]) &
                (self.df['Mins'] >= 500)
            ]['G+A_Per90']
            if len(group_data) > 0:
                age_groups_data.append(group_data)
                age_labels.append(label)
        
        parts = ax3.violinplot(age_groups_data, positions=range(len(age_groups_data)),
                              showmeans=True, showmedians=True)
        
        # Color the violins
        colors_violin = plt.cm.viridis(np.linspace(0.2, 0.8, len(age_groups_data)))
        for pc, color in zip(parts['bodies'], colors_violin):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax3.set_xticks(range(len(age_labels)))
        ax3.set_xticklabels(age_labels, fontsize=11)
        ax3.set_xlabel('Age Group', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Goals + Assists per 90 minutes', fontsize=12, fontweight='bold')
        ax3.set_title('Performance Distribution by Age Group', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        PlotUtils.save_figure('advanced_comparison_plots.png')
        plt.show()

def main():
    """Main execution function"""
    loader = DataLoader()
    df = loader.load_data()
    
    if df is not None:
        df = loader.create_derived_features(df)
        
        visualizer = AdvancedVisualizer(df)
        
        visualizer.player_radar_chart()
        visualizer.bubble_chart_analysis()
        visualizer.heatmap_visualizations()
        visualizer.advanced_scatter_matrix()
        visualizer.circular_bar_plots()
        visualizer.advanced_comparison_plots()
        
        print("\n" + "="*70)
        print("  ADVANCED VISUALIZATIONS COMPLETED")
        print("="*70)

if __name__ == "__main__":
    main()
