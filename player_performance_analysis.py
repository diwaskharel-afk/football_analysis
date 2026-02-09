"""
Advanced Player Performance Analysis - EPL 2020-21
This module focuses on individual player metrics, efficiency, and performance patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from config import *

class PlayerPerformanceAnalyzer:
    """Analyze individual player performance metrics"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def top_performers_analysis(self):
        """Comprehensive analysis of top performers across multiple metrics"""
        print_section_header("TOP PERFORMERS ANALYSIS")
        
        # Multi-metric top performers
        fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE_LARGE)
        fig.suptitle('Top 10 Players - Multiple Performance Metrics', fontsize=16, fontweight='bold', y=1.02)
        
        metrics = [
            ('Goals', 'Goals'),
            ('Assists', 'Assists'),
            ('Goals+Assists', 'G+A'),
            ('G+A_Per90', 'G+A per 90'),
            ('xG', 'Expected Goals (xG)'),
            ('Perc_Passes_Completed', 'Pass Completion %')
        ]
        
        for idx, (metric, label) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            top_data = self.df.nlargest(10, metric)
            
            bars = ax.barh(range(len(top_data)), top_data[metric], 
                          color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_data))))
            ax.set_yticks(range(len(top_data)))
            ax.set_yticklabels(top_data['Name'], fontsize=9)
            ax.set_xlabel(label, fontsize=10)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            
            # Add values on bars
            for i, (bar, value) in enumerate(zip(bars, top_data[metric])):
                ax.text(value, i, f' {value:.2f}', va='center', fontsize=8)
        
        plt.tight_layout()
        PlotUtils.save_figure('player_top_performers_multimet.png')
        plt.show()
        
    def efficiency_analysis(self):
        """Analyze player efficiency metrics"""
        print_section_header("PLAYER EFFICIENCY ANALYSIS")
        
        # Filter players with minimum playing time (at least 900 minutes)
        df_qualified = self.df[self.df['Mins'] >= 900].copy()
        
        # Create efficiency scatter plots
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_LARGE)
        fig.suptitle('Player Efficiency Metrics (Min 900 minutes played)', 
                     fontsize=16, fontweight='bold', y=1.00)
        
        # 1. Goals vs xG (overperformance)
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(df_qualified['xG'], df_qualified['Goals'], 
                              c=df_qualified['Mins'], cmap='coolwarm', alpha=0.6, s=100)
        ax1.plot([0, df_qualified['xG'].max()], [0, df_qualified['xG'].max()], 
                'k--', alpha=0.3, label='Expected line')
        ax1.set_xlabel('Expected Goals (xG)', fontsize=11)
        ax1.set_ylabel('Actual Goals', fontsize=11)
        ax1.set_title('Goal Conversion: Actual vs Expected', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Minutes Played')
        
        # Annotate top overperformers
        df_qualified['xG_diff'] = df_qualified['Goals'] - df_qualified['xG']
        top_over = df_qualified.nlargest(3, 'xG_diff')
        for _, player in top_over.iterrows():
            ax1.annotate(player['Name'], (player['xG'], player['Goals']),
                        fontsize=8, alpha=0.7, xytext=(5, 5), 
                        textcoords='offset points')
        
        # 2. Assists vs xA
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(df_qualified['xA'], df_qualified['Assists'], 
                              c=df_qualified['Mins'], cmap='coolwarm', alpha=0.6, s=100)
        ax2.plot([0, df_qualified['xA'].max()], [0, df_qualified['xA'].max()], 
                'k--', alpha=0.3, label='Expected line')
        ax2.set_xlabel('Expected Assists (xA)', fontsize=11)
        ax2.set_ylabel('Actual Assists', fontsize=11)
        ax2.set_title('Assist Creation: Actual vs Expected', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Minutes Played')
        
        # 3. Pass completion vs Passes attempted
        ax3 = axes[1, 0]
        forwards = df_qualified[df_qualified['Primary_Position'] == 'FW']
        midfielders = df_qualified[df_qualified['Primary_Position'] == 'MF']
        defenders = df_qualified[df_qualified['Primary_Position'] == 'DF']
        
        ax3.scatter(forwards['PassesPerMatch'], forwards['Perc_Passes_Completed'], 
                   label='Forwards', alpha=0.6, s=80)
        ax3.scatter(midfielders['PassesPerMatch'], midfielders['Perc_Passes_Completed'], 
                   label='Midfielders', alpha=0.6, s=80)
        ax3.scatter(defenders['PassesPerMatch'], defenders['Perc_Passes_Completed'], 
                   label='Defenders', alpha=0.6, s=80)
        ax3.set_xlabel('Passes Per Match', fontsize=11)
        ax3.set_ylabel('Pass Completion %', fontsize=11)
        ax3.set_title('Passing Efficiency by Position', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Goals+Assists per 90 vs Minutes
        ax4 = axes[1, 1]
        scatter4 = ax4.scatter(df_qualified['Mins'], df_qualified['G+A_Per90'], 
                              c=df_qualified['Age'], cmap='viridis', alpha=0.6, s=100)
        ax4.set_xlabel('Total Minutes Played', fontsize=11)
        ax4.set_ylabel('Goals + Assists per 90 min', fontsize=11)
        ax4.set_title('Productivity vs Playing Time', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
        plt.colorbar(scatter4, ax=ax4, label='Age')
        
        plt.tight_layout()
        PlotUtils.save_figure('player_efficiency_analysis.png')
        plt.show()
        
        # Print top overperformers
        print("\nTop 5 Goal Overperformers (Goals - xG):")
        print(df_qualified.nlargest(5, 'xG_diff')[['Name', 'Club', 'Goals', 'xG', 'xG_diff']])
        
        print("\nTop 5 Assist Overperformers (Assists - xA):")
        df_qualified['xA_diff'] = df_qualified['Assists'] - df_qualified['xA']
        print(df_qualified.nlargest(5, 'xA_diff')[['Name', 'Club', 'Assists', 'xA', 'xA_diff']])
    
    def positional_performance_comparison(self):
        """Compare performance across different positions"""
        print_section_header("POSITIONAL PERFORMANCE COMPARISON")
        
        # Filter main positions only
        main_positions = ['FW', 'MF', 'DF']
        df_pos = self.df[self.df['Primary_Position'].isin(main_positions)].copy()
        
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_LARGE)
        fig.suptitle('Performance Metrics by Position', fontsize=16, fontweight='bold', y=1.00)
        
        # 1. Goals distribution by position
        ax1 = axes[0, 0]
        df_pos.boxplot(column='Goals', by='Primary_Position', ax=ax1, patch_artist=True)
        ax1.set_title('Goals by Position', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Position', fontsize=11)
        ax1.set_ylabel('Goals', fontsize=11)
        plt.sca(ax1)
        plt.xticks(rotation=0)
        
        # 2. Assists distribution by position
        ax2 = axes[0, 1]
        df_pos.boxplot(column='Assists', by='Primary_Position', ax=ax2, patch_artist=True)
        ax2.set_title('Assists by Position', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Position', fontsize=11)
        ax2.set_ylabel('Assists', fontsize=11)
        plt.sca(ax2)
        plt.xticks(rotation=0)
        
        # 3. Pass completion by position
        ax3 = axes[1, 0]
        position_order = ['DF', 'MF', 'FW']
        sns.violinplot(data=df_pos, x='Primary_Position', y='Perc_Passes_Completed', 
                      order=position_order, ax=ax3, palette='Set2')
        ax3.set_title('Pass Completion % by Position', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Position', fontsize=11)
        ax3.set_ylabel('Pass Completion %', fontsize=11)
        ax3.axhline(y=df_pos['Perc_Passes_Completed'].mean(), color='r', 
                   linestyle='--', alpha=0.5, label='Overall Mean')
        ax3.legend()
        
        # 4. Discipline by position
        ax4 = axes[1, 1]
        discipline_data = df_pos.groupby('Primary_Position').agg({
            'Yellow_Cards': 'mean',
            'Red_Cards': 'mean'
        }).reindex(position_order)
        
        x = np.arange(len(position_order))
        width = 0.35
        ax4.bar(x - width/2, discipline_data['Yellow_Cards'], width, 
               label='Avg Yellow Cards', color='gold')
        ax4.bar(x + width/2, discipline_data['Red_Cards'], width, 
               label='Avg Red Cards', color='red')
        ax4.set_xlabel('Position', fontsize=11)
        ax4.set_ylabel('Average Cards', fontsize=11)
        ax4.set_title('Discipline by Position', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(position_order)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        PlotUtils.save_figure('positional_performance_comparison.png')
        plt.show()
        
        # Statistical summary by position
        print("\nStatistical Summary by Position:")
        summary = df_pos.groupby('Primary_Position').agg({
            'Goals': ['mean', 'median', 'std'],
            'Assists': ['mean', 'median', 'std'],
            'Perc_Passes_Completed': ['mean', 'std'],
            'Yellow_Cards': 'mean',
            'Red_Cards': 'mean'
        }).round(2)
        print(summary)
    
    def young_talent_analysis(self):
        """Identify and analyze young talented players"""
        print_section_header("YOUNG TALENT ANALYSIS")
        
        # Players under 23 with significant playing time
        young_players = self.df[(self.df['Age'] <= 23) & (self.df['Mins'] >= 900)].copy()
        
        # Create talent score (weighted combination of metrics)
        young_players['TalentScore'] = (
            young_players['G+A_Per90'] * 30 +
            young_players['Perc_Passes_Completed'] * 0.5 +
            (young_players['xG+xA'] / young_players['Matches']) * 20
        )
        
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_LARGE)
        fig.suptitle('Young Talent Analysis (Age ≤ 23, Min 900 minutes)', 
                     fontsize=16, fontweight='bold', y=1.00)
        
        # 1. Top young talents by talent score
        ax1 = axes[0, 0]
        top_young = young_players.nlargest(15, 'TalentScore')
        bars = ax1.barh(range(len(top_young)), top_young['TalentScore'],
                       color=plt.cm.plasma(np.linspace(0.3, 0.9, len(top_young))))
        ax1.set_yticks(range(len(top_young)))
        ax1.set_yticklabels([f"{name} ({age})" for name, age in 
                            zip(top_young['Name'], top_young['Age'])], fontsize=9)
        ax1.set_xlabel('Talent Score', fontsize=11)
        ax1.set_title('Top Young Talents', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Age vs Performance
        ax2 = axes[0, 1]
        scatter = ax2.scatter(young_players['Age'], young_players['G+A_Per90'],
                            s=young_players['Mins']/30, alpha=0.6,
                            c=young_players['Goals+Assists'], cmap='coolwarm')
        ax2.set_xlabel('Age', fontsize=11)
        ax2.set_ylabel('G+A per 90 minutes', fontsize=11)
        ax2.set_title('Age vs Goal Contribution Rate', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Total G+A')
        
        # Annotate standout performers
        standouts = young_players.nlargest(5, 'G+A_Per90')
        for _, player in standouts.iterrows():
            ax2.annotate(player['Name'], (player['Age'], player['G+A_Per90']),
                        fontsize=8, alpha=0.7, xytext=(5, 5),
                        textcoords='offset points')
        
        # 3. Club-wise young talent distribution
        ax3 = axes[1, 0]
        young_by_club = young_players.groupby('Club').size().sort_values(ascending=False).head(10)
        ax3.bar(range(len(young_by_club)), young_by_club.values, 
               color=sns.color_palette('Set3', len(young_by_club)))
        ax3.set_xticks(range(len(young_by_club)))
        ax3.set_xticklabels(young_by_club.index, rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('Number of Young Talents', fontsize=11)
        ax3.set_title('Clubs with Most Young Talents', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Position distribution
        ax4 = axes[1, 1]
        position_dist = young_players['Primary_Position'].value_counts()
        colors_pie = sns.color_palette('Set2', len(position_dist))
        ax4.pie(position_dist.values, labels=position_dist.index, autopct='%1.1f%%',
               colors=colors_pie, startangle=90)
        ax4.set_title('Position Distribution', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        PlotUtils.save_figure('young_talent_analysis.png')
        plt.show()
        
        print("\nTop 10 Young Talents:")
        print(young_players.nlargest(10, 'TalentScore')[
            ['Name', 'Age', 'Club', 'Goals', 'Assists', 'G+A_Per90', 'TalentScore']
        ].round(2))

def main():
    """Main execution function"""
    # Load and prepare data
    loader = DataLoader()
    df = loader.load_data()
    
    if df is not None:
        df = loader.create_derived_features(df)
        
        # Initialize analyzer
        analyzer = PlayerPerformanceAnalyzer(df)
        
        # Run all analyses
        analyzer.top_performers_analysis()
        analyzer.efficiency_analysis()
        analyzer.positional_performance_comparison()
        analyzer.young_talent_analysis()
        
        print("\n" + "="*70)
        print("  PLAYER PERFORMANCE ANALYSIS COMPLETED")
        print("="*70)

if __name__ == "__main__":
    main()
