"""
Master Script - EPL 2020-21 Complete Analysis
Run all analysis modules and generate comprehensive report
"""

import os
import pandas as pd
import time
from datetime import datetime
from config import DataLoader, print_section_header

def create_output_directory():
    """Create output directory if it doesn't exist"""
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        print("Created 'outputs' directory for saving visualizations")

def generate_summary_report(df):
    """Generate a summary report of the dataset"""
    print_section_header("DATASET SUMMARY REPORT")
    
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nDataset Overview:")
    print(f"  Total Players: {len(df)}")
    print(f"  Total Clubs: {df['Club'].nunique()}")
    print(f"  Nationalities Represented: {df['Nationality'].nunique()}")
    print(f"  Positions Covered: {df['Position'].nunique()}")
    
    print(f"\nAge Statistics:")
    print(f"  Youngest Player: {df['Age'].min()} years")
    print(f"  Oldest Player: {df['Age'].max()} years")
    print(f"  Average Age: {df['Age'].mean():.2f} years")
    
    print(f"\nPerformance Highlights:")
    print(f"  Total Goals Scored: {df['Goals'].sum()}")
    print(f"  Total Assists: {df['Assists'].sum()}")
    print(f"  Total Penalty Goals: {df['Penalty_Goals'].sum()}")
    print(f"  Total Yellow Cards: {df['Yellow_Cards'].sum()}")
    print(f"  Total Red Cards: {df['Red_Cards'].sum()}")
    
    top_scorer = df.nlargest(1, 'Goals').iloc[0]
    print(f"\n  Top Scorer: {top_scorer['Name']} ({top_scorer['Club']}) - {int(top_scorer['Goals'])} goals")
    
    top_assist = df.nlargest(1, 'Assists').iloc[0]
    print(f"  Top Assist Provider: {top_assist['Name']} ({top_assist['Club']}) - {int(top_assist['Assists'])} assists")
    
    print(f"\nTop 5 Clubs by Total Goals:")
    club_goals = df.groupby('Club')['Goals'].sum().sort_values(ascending=False).head(5)
    for i, (club, goals) in enumerate(club_goals.items(), 1):
        print(f"  {i}. {club}: {int(goals)} goals")

def run_all_analyses():
    """Run all analysis modules sequentially"""
    
    print("\n" + "="*70)
    print("  EPL 2020-21 SEASON - COMPREHENSIVE ANALYSIS SUITE")
    print("="*70)
    
    # Create output directory
    create_output_directory()
    
    # Load data
    print("\n[1/5] Loading and preparing data...")
    loader = DataLoader()
    df = loader.load_data()
    
    if df is None:
        print("Error: Could not load data. Please ensure 'EPL_20_21.csv' is in the current directory.")
        return
    
    df = loader.create_derived_features(df)
    print(f"✓ Data loaded successfully: {len(df)} players")
    
    # Generate summary report
    generate_summary_report(df)
    
    # Run analysis modules
    try:
        print("\n" + "="*70)
        print("[2/5] Running Player Performance Analysis...")
        print("="*70)
        from player_performance_analysis import PlayerPerformanceAnalyzer
        player_analyzer = PlayerPerformanceAnalyzer(df)
        player_analyzer.top_performers_analysis()
        player_analyzer.efficiency_analysis()
        player_analyzer.positional_performance_comparison()
        player_analyzer.young_talent_analysis()
        print("✓ Player Performance Analysis completed")
        
    except Exception as e:
        print(f"✗ Error in Player Performance Analysis: {str(e)}")
    
    try:
        print("\n" + "="*70)
        print("[3/5] Running Club Performance Analysis...")
        print("="*70)
        from club_performance_analysis import ClubPerformanceAnalyzer
        club_analyzer = ClubPerformanceAnalyzer(df)
        club_analyzer.attacking_performance_analysis()
        club_analyzer.squad_composition_analysis()
        club_analyzer.discipline_analysis()
        club_analyzer.club_clustering_analysis()
        print("✓ Club Performance Analysis completed")
        
    except Exception as e:
        print(f"✗ Error in Club Performance Analysis: {str(e)}")
    
    try:
        print("\n" + "="*70)
        print("[4/5] Running Statistical Analysis...")
        print("="*70)
        from statistical_analysis import StatisticalAnalyzer
        stats_analyzer = StatisticalAnalyzer(df)
        stats_analyzer.correlation_analysis()
        stats_analyzer.regression_analysis()
        stats_analyzer.distribution_analysis()
        stats_analyzer.outlier_analysis()
        stats_analyzer.comparative_statistical_tests()
        print("✓ Statistical Analysis completed")
        
    except Exception as e:
        print(f"✗ Error in Statistical Analysis: {str(e)}")
    
    try:
        print("\n" + "="*70)
        print("[5/5] Running Advanced Visualizations...")
        print("="*70)
        from advanced_visualizations import AdvancedVisualizer
        viz_analyzer = AdvancedVisualizer(df)
        viz_analyzer.player_radar_chart()
        viz_analyzer.bubble_chart_analysis()
        viz_analyzer.heatmap_visualizations()
        viz_analyzer.advanced_scatter_matrix()
        viz_analyzer.circular_bar_plots()
        viz_analyzer.advanced_comparison_plots()
        print("✓ Advanced Visualizations completed")
        
    except Exception as e:
        print(f"✗ Error in Advanced Visualizations: {str(e)}")
    
    # Final summary
    print("\n" + "="*70)
    print("  ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll visualizations have been saved to the 'outputs/' directory.")
    print(f"Total analyses performed: 5 modules")
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List generated files
    if os.path.exists('outputs'):
        files = [f for f in os.listdir('outputs') if f.endswith('.png')]
        if files:
            print(f"\nGenerated {len(files)} visualization files:")
            for f in sorted(files):
                print(f"  - {f}")

def run_individual_module(module_number):
    """Run a specific analysis module"""
    
    create_output_directory()
    
    loader = DataLoader()
    df = loader.load_data()
    
    if df is None:
        print("Error: Could not load data.")
        return
    
    df = loader.create_derived_features(df)
    
    modules = {
        1: ("Player Performance Analysis", "player_performance_analysis", "PlayerPerformanceAnalyzer"),
        2: ("Club Performance Analysis", "club_performance_analysis", "ClubPerformanceAnalyzer"),
        3: ("Statistical Analysis", "statistical_analysis", "StatisticalAnalyzer"),
        4: ("Advanced Visualizations", "advanced_visualizations", "AdvancedVisualizer")
    }
    
    if module_number in modules:
        name, module_name, class_name = modules[module_number]
        print(f"\nRunning {name}...")
        exec(f"from {module_name} import {class_name}")
        exec(f"{class_name}(df)")
    else:
        print(f"Invalid module number. Choose from 1-4.")

if __name__ == "__main__":
    import sys
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║         EPL 2020-21 SEASON - ADVANCED ANALYSIS SUITE            ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    This comprehensive analysis suite includes:
    
    1. Player Performance Analysis
       - Top performers across multiple metrics
       - Efficiency analysis (xG vs actual, xA vs actual)
       - Positional performance comparisons
       - Young talent identification
    
    2. Club Performance Analysis
       - Attacking performance metrics
       - Squad composition and demographics
       - Discipline analysis
       - Club clustering and similarity
    
    3. Statistical Analysis
       - Correlation analysis
       - Regression modeling
       - Distribution analysis
       - Outlier detection
       - Comparative statistical tests
    
    4. Advanced Visualizations
       - Player radar charts
       - Multi-dimensional bubble charts
       - Heatmaps (position, age, performance)
       - Scatter matrices
       - Circular bar plots
       - Advanced comparison plots
    
    ──────────────────────────────────────────────────────────────────
    """)
    
    # Check if running all or specific module
    if len(sys.argv) > 1:
        try:
            module_num = int(sys.argv[1])
            run_individual_module(module_num)
        except ValueError:
            print("Usage: python master_analysis.py [module_number]")
            print("Module numbers: 1-4, or run without arguments for all modules")
    else:
        # Run all analyses
        start_time = time.time()
        run_all_analyses()
        elapsed_time = time.time() - start_time
        print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
