"""
Club and Team Performance Analysis - EPL 2020-21
This module analyzes team-level metrics, squad composition, and club comparisons
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from config import *

class ClubPerformanceAnalyzer:
    """Analyze club-level performance and squad metrics"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.club_stats = self._calculate_club_stats()
    
    def _calculate_club_stats(self):
        """Calculate comprehensive club statistics"""
        club_stats = self.df.groupby('Club').agg({
            'Goals': 'sum',
            'Assists': 'sum',
            'xG': 'sum',
            'xA': 'sum',
            'Passes_Attempted': 'sum',
            'Perc_Passes_Completed': 'mean',
            'Yellow_Cards': 'sum',
            'Red_Cards': 'sum',
            'Age': 'mean',
            'Name': 'count',
            'Mins': 'sum'
        }).round(2)
        
        club_stats.columns = ['Total_Goals', 'Total_Assists', 'Total_xG', 'Total_xA',
                             'Total_Passes', 'Avg_Pass_Completion', 'Yellow_Cards',
                             'Red_Cards', 'Avg_Age', 'Squad_Size', 'Total_Minutes']
        
        # Derived metrics
        club_stats['Goal_Efficiency'] = (club_stats['Total_Goals'] / club_stats['Total_xG'] * 100).round(2)
        club_stats['Assist_Efficiency'] = (club_stats['Total_Assists'] / club_stats['Total_xA'] * 100).round(2)
        club_stats['Attacking_Output'] = club_stats['Total_Goals'] + club_stats['Total_Assists']
        club_stats['Discipline_Index'] = club_stats['Yellow_Cards'] + (club_stats['Red_Cards'] * 3)
        
        return club_stats
    
    def attacking_performance_analysis(self):
        """Analyze attacking metrics across clubs"""
        print_section_header("CLUB ATTACKING PERFORMANCE")
        
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_LARGE)
        fig.suptitle('Club Attacking Performance Metrics', fontsize=16, fontweight='bold', y=1.00)
        
        # 1. Total Goals by Club
        ax1 = axes[0, 0]
        sorted_goals = self.club_stats.sort_values('Total_Goals', ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_goals)))
        bars1 = ax1.barh(range(len(sorted_goals)), sorted_goals['Total_Goals'], color=colors)
        ax1.set_yticks(range(len(sorted_goals)))
        ax1.set_yticklabels(sorted_goals.index, fontsize=9)
        ax1.set_xlabel('Total Goals', fontsize=11)
        ax1.set_title('Total Goals by Club', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add league average line
        avg_goals = sorted_goals['Total_Goals'].mean()
        ax1.axvline(avg_goals, color='red', linestyle='--', alpha=0.5, label=f'Avg: {avg_goals:.1f}')
        ax1.legend()
        
        # 2. Actual vs Expected Goals (xG)
        ax2 = axes[0, 1]
        clubs = self.club_stats.index
        x = np.arange(len(clubs))
        width = 0.35
        
        sorted_clubs = self.club_stats.sort_values('Total_Goals', ascending=False)
        bars_actual = ax2.bar(x - width/2, sorted_clubs['Total_Goals'], width, 
                             label='Actual Goals', color='steelblue', alpha=0.8)
        bars_expected = ax2.bar(x + width/2, sorted_clubs['Total_xG'], width,
                               label='Expected Goals (xG)', color='coral', alpha=0.8)
        
        ax2.set_xlabel('Club', fontsize=11)
        ax2.set_ylabel('Goals', fontsize=11)
        ax2.set_title('Actual Goals vs Expected Goals', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(sorted_clubs.index, rotation=45, ha='right', fontsize=8)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Goal Efficiency (Goals/xG ratio)
        ax3 = axes[1, 0]
        efficiency_sorted = self.club_stats.sort_values('Goal_Efficiency', ascending=True)
        efficiency_colors = ['green' if x > 100 else 'red' for x in efficiency_sorted['Goal_Efficiency']]
        
        bars3 = ax3.barh(range(len(efficiency_sorted)), efficiency_sorted['Goal_Efficiency'], 
                        color=efficiency_colors, alpha=0.7)
        ax3.set_yticks(range(len(efficiency_sorted)))
        ax3.set_yticklabels(efficiency_sorted.index, fontsize=9)
        ax3.set_xlabel('Goal Efficiency %', fontsize=11)
        ax3.set_title('Goal Efficiency (Goals/xG %)', fontsize=12, fontweight='bold')
        ax3.axvline(100, color='black', linestyle='--', alpha=0.5, label='100% (On target)')
        ax3.legend()
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Goals + Assists (Total Attacking Output)
        ax4 = axes[1, 1]
        attacking_sorted = self.club_stats.sort_values('Attacking_Output', ascending=False)
        
        # Stacked bar chart
        ax4.bar(range(len(attacking_sorted)), attacking_sorted['Total_Goals'], 
               label='Goals', color='crimson', alpha=0.8)
        ax4.bar(range(len(attacking_sorted)), attacking_sorted['Total_Assists'], 
               bottom=attacking_sorted['Total_Goals'], label='Assists', 
               color='gold', alpha=0.8)
        
        ax4.set_xlabel('Club', fontsize=11)
        ax4.set_ylabel('Total Attacking Output', fontsize=11)
        ax4.set_title('Goals + Assists by Club', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(attacking_sorted)))
        ax4.set_xticklabels(attacking_sorted.index, rotation=45, ha='right', fontsize=8)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        PlotUtils.save_figure('club_attacking_performance.png')
        plt.show()
        
        # Print top/bottom performers
        print("\nTop 5 Attacking Clubs:")
        print(self.club_stats.nlargest(5, 'Attacking_Output')[
            ['Total_Goals', 'Total_Assists', 'Attacking_Output', 'Goal_Efficiency']
        ])
        
        print("\nMost Efficient Clubs (Goals vs xG):")
        print(self.club_stats.nlargest(5, 'Goal_Efficiency')[
            ['Total_Goals', 'Total_xG', 'Goal_Efficiency']
        ])
    
    def squad_composition_analysis(self):
        """Analyze squad composition and demographics"""
        print_section_header("SQUAD COMPOSITION ANALYSIS")
        
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_LARGE)
        fig.suptitle('Squad Composition & Demographics', fontsize=16, fontweight='bold', y=1.00)
        
        # 1. Squad Size
        ax1 = axes[0, 0]
        squad_size_sorted = self.club_stats.sort_values('Squad_Size', ascending=True)
        bars1 = ax1.barh(range(len(squad_size_sorted)), squad_size_sorted['Squad_Size'],
                        color=plt.cm.viridis(np.linspace(0.3, 0.9, len(squad_size_sorted))))
        ax1.set_yticks(range(len(squad_size_sorted)))
        ax1.set_yticklabels(squad_size_sorted.index, fontsize=9)
        ax1.set_xlabel('Squad Size', fontsize=11)
        ax1.set_title('Squad Size by Club', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add values
        for i, v in enumerate(squad_size_sorted['Squad_Size']):
            ax1.text(v, i, f' {int(v)}', va='center', fontsize=8)
        
        # 2. Average Age
        ax2 = axes[0, 1]
        age_sorted = self.club_stats.sort_values('Avg_Age', ascending=True)
        bars2 = ax2.barh(range(len(age_sorted)), age_sorted['Avg_Age'],
                        color=plt.cm.coolwarm(np.linspace(0.2, 0.8, len(age_sorted))))
        ax2.set_yticks(range(len(age_sorted)))
        ax2.set_yticklabels(age_sorted.index, fontsize=9)
        ax2.set_xlabel('Average Age', fontsize=11)
        ax2.set_title('Average Squad Age', fontsize=12, fontweight='bold')
        ax2.axvline(self.club_stats['Avg_Age'].mean(), color='black', 
                   linestyle='--', alpha=0.5, label='League Avg')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Position Distribution by Club (sample top clubs)
        ax3 = axes[1, 0]
        top_clubs = self.club_stats.nlargest(6, 'Total_Goals').index
        
        position_data = []
        for club in top_clubs:
            club_df = self.df[self.df['Club'] == club]
            pos_counts = club_df['Primary_Position'].value_counts()
            position_data.append([
                pos_counts.get('FW', 0),
                pos_counts.get('MF', 0),
                pos_counts.get('DF', 0),
                pos_counts.get('GK', 0)
            ])
        
        position_data = np.array(position_data)
        x = np.arange(len(top_clubs))
        width = 0.6
        
        bottom = np.zeros(len(top_clubs))
        colors_pos = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        labels_pos = ['FW', 'MF', 'DF', 'GK']
        
        for i, (data, color, label) in enumerate(zip(position_data.T, colors_pos, labels_pos)):
            ax3.bar(x, data, width, bottom=bottom, label=label, color=color, alpha=0.8)
            bottom += data
        
        ax3.set_xlabel('Club', fontsize=11)
        ax3.set_ylabel('Number of Players', fontsize=11)
        ax3.set_title('Position Distribution (Top 6 Clubs)', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(top_clubs, rotation=45, ha='right', fontsize=9)
        ax3.legend(loc='upper right')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Nationality Diversity
        ax4 = axes[1, 1]
        diversity_data = []
        clubs_sample = self.club_stats.index[:15]
        
        for club in clubs_sample:
            club_df = self.df[self.df['Club'] == club]
            unique_nationalities = club_df['Nationality'].nunique()
            diversity_data.append(unique_nationalities)
        
        bars4 = ax4.bar(range(len(clubs_sample)), diversity_data,
                       color=plt.cm.Spectral(np.linspace(0.2, 0.8, len(clubs_sample))))
        ax4.set_xlabel('Club', fontsize=11)
        ax4.set_ylabel('Number of Nationalities', fontsize=11)
        ax4.set_title('Nationality Diversity', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(clubs_sample)))
        ax4.set_xticklabels(clubs_sample, rotation=45, ha='right', fontsize=8)
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        PlotUtils.save_figure('squad_composition_analysis.png')
        plt.show()
        
        print("\nSquad Statistics Summary:")
        print(self.club_stats[['Squad_Size', 'Avg_Age']].describe().round(2))
    
    def discipline_analysis(self):
        """Analyze discipline metrics across clubs"""
        print_section_header("CLUB DISCIPLINE ANALYSIS")
        
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_LARGE)
        fig.suptitle('Club Discipline Metrics', fontsize=16, fontweight='bold', y=1.00)
        
        # 1. Total Yellow Cards
        ax1 = axes[0, 0]
        yellow_sorted = self.club_stats.sort_values('Yellow_Cards', ascending=True)
        bars1 = ax1.barh(range(len(yellow_sorted)), yellow_sorted['Yellow_Cards'],
                        color='gold', alpha=0.7, edgecolor='orange')
        ax1.set_yticks(range(len(yellow_sorted)))
        ax1.set_yticklabels(yellow_sorted.index, fontsize=9)
        ax1.set_xlabel('Yellow Cards', fontsize=11)
        ax1.set_title('Total Yellow Cards by Club', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Total Red Cards
        ax2 = axes[0, 1]
        red_sorted = self.club_stats.sort_values('Red_Cards', ascending=True)
        bars2 = ax2.barh(range(len(red_sorted)), red_sorted['Red_Cards'],
                        color='red', alpha=0.7, edgecolor='darkred')
        ax2.set_yticks(range(len(red_sorted)))
        ax2.set_yticklabels(red_sorted.index, fontsize=9)
        ax2.set_xlabel('Red Cards', fontsize=11)
        ax2.set_title('Total Red Cards by Club', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Discipline Index
        ax3 = axes[1, 0]
        discipline_sorted = self.club_stats.sort_values('Discipline_Index', ascending=False)
        colors_disc = plt.cm.Reds(np.linspace(0.4, 0.9, len(discipline_sorted)))
        bars3 = ax3.bar(range(len(discipline_sorted)), discipline_sorted['Discipline_Index'],
                       color=colors_disc, alpha=0.8)
        ax3.set_xlabel('Club', fontsize=11)
        ax3.set_ylabel('Discipline Index', fontsize=11)
        ax3.set_title('Discipline Index (Yellow + Red×3)', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(len(discipline_sorted)))
        ax3.set_xticklabels(discipline_sorted.index, rotation=45, ha='right', fontsize=8)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Cards vs Performance
        ax4 = axes[1, 1]
        scatter = ax4.scatter(self.club_stats['Discipline_Index'], 
                            self.club_stats['Total_Goals'],
                            s=self.club_stats['Squad_Size']*20, 
                            alpha=0.6, c=self.club_stats['Avg_Age'],
                            cmap='viridis')
        ax4.set_xlabel('Discipline Index', fontsize=11)
        ax4.set_ylabel('Total Goals', fontsize=11)
        ax4.set_title('Discipline vs Goals (size=squad, color=age)', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Avg Age')
        
        # Annotate outliers
        for club in self.club_stats.index:
            if (self.club_stats.loc[club, 'Discipline_Index'] > 
                self.club_stats['Discipline_Index'].quantile(0.85)):
                ax4.annotate(club, 
                           (self.club_stats.loc[club, 'Discipline_Index'],
                            self.club_stats.loc[club, 'Total_Goals']),
                           fontsize=8, alpha=0.7, xytext=(5, 5),
                           textcoords='offset points')
        
        plt.tight_layout()
        PlotUtils.save_figure('club_discipline_analysis.png')
        plt.show()
        
        print("\nMost Disciplined Clubs (Lowest Discipline Index):")
        print(self.club_stats.nsmallest(5, 'Discipline_Index')[
            ['Yellow_Cards', 'Red_Cards', 'Discipline_Index']
        ])
        
        print("\nLeast Disciplined Clubs:")
        print(self.club_stats.nlargest(5, 'Discipline_Index')[
            ['Yellow_Cards', 'Red_Cards', 'Discipline_Index']
        ])
    
    def club_clustering_analysis(self):
        """Perform clustering analysis to group similar clubs"""
        print_section_header("CLUB CLUSTERING ANALYSIS")
        
        # Prepare data for clustering
        features = ['Total_Goals', 'Total_Assists', 'Avg_Pass_Completion', 
                   'Avg_Age', 'Squad_Size', 'Discipline_Index']
        X = self.club_stats[features].copy()
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_LARGE)
        fig.suptitle('Club Similarity Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Hierarchical Clustering Dendrogram
        ax1 = axes[0]
        linkage_matrix = linkage(X_scaled, method='ward')
        dendrogram(linkage_matrix, labels=self.club_stats.index, ax=ax1, 
                  orientation='right', leaf_font_size=9)
        ax1.set_xlabel('Distance', fontsize=11)
        ax1.set_title('Hierarchical Clustering of Clubs', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. PCA-based visualization (simplified)
        ax2 = axes[1]
        
        # Use correlation-based grouping for visualization
        attacking_strength = (self.club_stats['Total_Goals'] + self.club_stats['Total_Assists']) / 2
        defensive_discipline = 100 - (self.club_stats['Discipline_Index'] / 
                                     self.club_stats['Discipline_Index'].max() * 100)
        
        scatter = ax2.scatter(attacking_strength, defensive_discipline,
                            s=self.club_stats['Squad_Size']*15,
                            c=self.club_stats['Avg_Age'],
                            cmap='coolwarm', alpha=0.6, edgecolors='black')
        
        ax2.set_xlabel('Attacking Strength (Avg G+A)', fontsize=11)
        ax2.set_ylabel('Discipline Score (0-100)', fontsize=11)
        ax2.set_title('Club Positioning: Attack vs Discipline', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Avg Age')
        
        # Add club labels
        for club in self.club_stats.index:
            ax2.annotate(club, 
                       (attacking_strength[club], defensive_discipline[club]),
                       fontsize=7, alpha=0.6)
        
        # Add quadrant lines
        ax2.axhline(defensive_discipline.median(), color='gray', linestyle='--', alpha=0.3)
        ax2.axvline(attacking_strength.median(), color='gray', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        PlotUtils.save_figure('club_clustering_analysis.png')
        plt.show()
        
        print("\nClub Performance Quadrants:")
        print("\nHigh Attack, High Discipline:")
        high_attack_high_disc = self.club_stats[
            (attacking_strength > attacking_strength.median()) & 
            (defensive_discipline > defensive_discipline.median())
        ].index.tolist()
        print(high_attack_high_disc)
        
        print("\nHigh Attack, Low Discipline:")
        high_attack_low_disc = self.club_stats[
            (attacking_strength > attacking_strength.median()) & 
            (defensive_discipline <= defensive_discipline.median())
        ].index.tolist()
        print(high_attack_low_disc)

def main():
    """Main execution function"""
    loader = DataLoader()
    df = loader.load_data()
    
    if df is not None:
        df = loader.create_derived_features(df)
        
        analyzer = ClubPerformanceAnalyzer(df)
        
        analyzer.attacking_performance_analysis()
        analyzer.squad_composition_analysis()
        analyzer.discipline_analysis()
        analyzer.club_clustering_analysis()
        
        print("\n" + "="*70)
        print("  CLUB PERFORMANCE ANALYSIS COMPLETED")
        print("="*70)

if __name__ == "__main__":
    main()
