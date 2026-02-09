"""
Configuration and Utility Functions for EPL 2020-21 Analysis Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Set styling defaults
sns.set_palette("husl")
plt.style.use('seaborn-v0_8-darkgrid')

# Configuration
DATA_PATH = 'EPL_20_21.csv'
OUTPUT_DIR = 'outputs/'
FIGURE_SIZE = (14, 8)
FIGURE_SIZE_SMALL = (10, 6)
FIGURE_SIZE_LARGE = (16, 10)

# Color palettes for different visualizations
COLORS = {
    'primary': sns.color_palette("Set2"),
    'sequential': sns.color_palette("rocket"),
    'diverging': sns.color_palette("coolwarm"),
    'categorical': sns.color_palette("tab20"),
    'pastel': sns.color_palette("pastel"),
    'dark': sns.color_palette("dark"),
    'muted': sns.color_palette("muted")
}

class DataLoader:
    """Class to load and preprocess EPL data"""
    
    @staticmethod
    def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
        """Load EPL dataset"""
        try:
            df = pd.read_csv(filepath)
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None
    
    @staticmethod
    def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for analysis"""
        df = df.copy()
        
        # Basic metrics
        df['MinsPerMatch'] = (df['Mins'] / df['Matches']).replace([np.inf, -np.inf], 0)
        df['GoalsPerMatch'] = (df['Goals'] / df['Matches']).replace([np.inf, -np.inf], 0)
        df['AssistsPerMatch'] = (df['Assists'] / df['Matches']).replace([np.inf, -np.inf], 0)
        
        # Advanced metrics
        df['Goals+Assists'] = df['Goals'] + df['Assists']
        df['G+A_Per90'] = ((df['Goals+Assists'] / df['Mins']) * 90).replace([np.inf, -np.inf], 0)
        df['xG+xA'] = df['xG'] + df['xA']
        df['GoalConversionRate'] = (df['Goals'] / df['Passes_Attempted'] * 100).replace([np.inf, -np.inf], 0)
        df['NonPenaltyGoals'] = df['Goals'] - df['Penalty_Goals']
        df['xG_Overperformance'] = df['Goals'] - df['xG']
        df['xA_Overperformance'] = df['Assists'] - df['xA']
        df['Penalty_Conversion'] = np.where(df['Penalty_Attempted'] > 0, 
                                             df['Penalty_Goals'] / df['Penalty_Attempted'] * 100, 0)
        
        # Efficiency metrics
        df['PassesPerMatch'] = (df['Passes_Attempted'] / df['Matches']).replace([np.inf, -np.inf], 0)
        df['SuccessfulPasses'] = (df['Passes_Attempted'] * df['Perc_Passes_Completed'] / 100).round()
        
        # Discipline metrics
        df['CardsPerMatch'] = ((df['Yellow_Cards'] + df['Red_Cards']) / df['Matches']).replace([np.inf, -np.inf], 0)
        df['DisciplineScore'] = df['Yellow_Cards'] + (df['Red_Cards'] * 3)  # Red cards weighted 3x
        
        # Age groups
        df['AgeGroup'] = pd.cut(df['Age'], 
                                bins=[0, 20, 25, 30, 100], 
                                labels=['U20', '21-25', '26-30', '30+'])
        
        # Position simplification
        df['Primary_Position'] = df['Position'].apply(lambda x: x.split(',')[0])
        
        return df

class PlotUtils:
    """Utility functions for plotting"""
    
    @staticmethod
    def save_figure(filename: str, dpi: int = 300):
        """Save figure with high quality"""
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}{filename}', dpi=dpi, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    @staticmethod
    def add_value_labels(ax, spacing: int = 5, format_str: str = '{:.1f}'):
        """Add value labels on bar charts"""
        for rect in ax.patches:
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
            label = format_str.format(y_value)
            ax.annotate(label, (x_value, y_value), 
                       xytext=(0, spacing), 
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    @staticmethod
    def create_comparison_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                              title: str, xlabel: str, ylabel: str,
                              n_items: int = 15, palette: str = 'viridis'):
        """Create a standardized comparison bar plot"""
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        data = df.nlargest(n_items, y_col)
        sns.barplot(data=data, x=x_col, y=y_col, palette=palette, ax=ax)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        return fig, ax

def print_section_header(title: str):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")
