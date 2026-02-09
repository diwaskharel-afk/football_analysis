# EPL 2020-21 Season - Advanced Analysis Suite

## 🏆 Project Overview

This is a comprehensive, production-grade analysis of the English Premier League 2020-21 season data. The project features advanced statistical analysis, machine learning techniques, and sophisticated visualizations to extract meaningful insights about player and team performance.

## 📊 Features

### 1. Player Performance Analysis (`1_player_performance_analysis.py`)
- **Top Performers Analysis**: Multi-metric evaluation of elite players
- **Efficiency Analysis**: 
  - Actual vs Expected Goals (xG) analysis
  - Actual vs Expected Assists (xA) analysis
  - Position-based passing efficiency
  - Productivity vs playing time correlation
- **Positional Comparisons**: Statistical comparison across FW, MF, DF positions
- **Young Talent Identification**: Custom talent scoring algorithm for players ≤23

### 2. Club Performance Analysis (`2_club_performance_analysis.py`)
- **Attacking Metrics**: 
  - Total goals and assists by club
  - Goal efficiency (Actual/xG ratio)
  - Expected vs actual performance
- **Squad Composition**:
  - Squad size and average age
  - Position distribution
  - Nationality diversity index
- **Discipline Analysis**: Yellow/red card patterns and discipline index
- **Club Clustering**: Hierarchical clustering to identify similar teams

### 3. Statistical Analysis (`3_statistical_analysis.py`)
- **Correlation Analysis**: 
  - Heatmaps showing relationships between 12+ metrics
  - Top positive and negative correlations
- **Regression Modeling**:
  - Linear regression for goal prediction
  - Multi-feature models with R² scoring
  - Feature importance analysis
- **Distribution Analysis**: 
  - Normality testing (Shapiro-Wilk test)
  - KDE plots with statistical parameters
- **Outlier Detection**: IQR-based outlier identification
- **Comparative Tests**:
  - ANOVA for age group comparisons
  - T-tests for positional differences

### 4. Advanced Visualizations (`4_advanced_visualizations.py`)
- **Radar Charts**: Multi-dimensional player comparisons
- **Bubble Charts**: 4-dimensional performance visualization
- **Heatmaps**: 
  - Club vs Position performance matrices
  - Age group vs performance
  - Discipline patterns
- **Scatter Matrices**: Pairwise relationship exploration
- **Circular Bar Plots**: Radial rankings for top performers
- **Advanced Comparisons**:
  - Lollipop charts
  - Diverging bar charts
  - Violin plots for distribution comparisons

## 🗂️ Project Structure

```
EPL_Analysis/
│
├── config.py                          # Configuration and utility functions
├── master_analysis.py                 # Main script to run all analyses
├── player_performance_analysis.py     # Player-focused analysis
├── club_performance_analysis.py       # Club/team-focused analysis
├── statistical_analysis.py            # Statistical methods and tests
├── advanced_visualizations.py         # Creative visualizations
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
│
├── EPL_20_21.csv                      # Input data (place your file here)
│
└── outputs/                           # Generated visualizations
    ├── player_top_performers_multimet.png
    ├── player_efficiency_analysis.png
    ├── positional_performance_comparison.png
    ├── young_talent_analysis.png
    ├── club_attacking_performance.png
    ├── squad_composition_analysis.png
    ├── club_discipline_analysis.png
    ├── club_clustering_analysis.png
    ├── correlation_analysis.png
    ├── regression_analysis.png
    ├── distribution_analysis.png
    ├── outlier_analysis.png
    ├── statistical_comparisons.png
    ├── player_radar_charts.png
    ├── bubble_chart_analysis.png
    ├── heatmap_visualizations.png
    ├── scatter_matrix_analysis.png
    ├── circular_bar_plots.png
    └── advanced_comparison_plots.png
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or download this project**

2. **Install required packages**:
```bash
pip install -r requirements.txt
```

3. **Place your data file**:
   - Ensure `EPL_20_21.csv` is in the project root directory
   - The CSV should have the following columns:
     - Name, Club, Nationality, Position, Age, Matches, Starts, Mins
     - Goals, Assists, Passes_Attempted, Perc_Passes_Completed
     - Penalty_Goals, Penalty_Attempted, xG, xA
     - Yellow_Cards, Red_Cards

## 📖 Usage

### Option 1: Run Complete Analysis Suite
```bash
python master_analysis.py
```
This will execute all four analysis modules sequentially and generate all visualizations.

### Option 2: Run Individual Modules
```bash
# Run only player performance analysis
python master_analysis.py 1

# Run only club performance analysis
python master_analysis.py 2

# Run only statistical analysis
python master_analysis.py 3

# Run only advanced visualizations
python master_analysis.py 4
```

### Option 3: Run Specific Analysis Files Directly
```bash
python player_performance_analysis.py
python club_performance_analysis.py
python statistical_analysis.py
python advanced_visualizations.py
```

## 📈 Key Metrics & Derived Features

### Custom Calculated Metrics:
- **MinsPerMatch**: Average minutes played per match
- **GoalsPerMatch**: Goal scoring rate
- **AssistsPerMatch**: Assist providing rate
- **Goals+Assists**: Total goal contributions
- **G+A_Per90**: Goal contributions per 90 minutes
- **xG+xA**: Total expected goal contributions
- **GoalConversionRate**: Goals per pass attempted (%)
- **NonPenaltyGoals**: Goals excluding penalties
- **xG_Overperformance**: Actual goals minus expected goals
- **xA_Overperformance**: Actual assists minus expected assists
- **Penalty_Conversion**: Penalty success rate (%)
- **PassesPerMatch**: Average passes per match
- **CardsPerMatch**: Disciplinary issues rate
- **DisciplineScore**: Weighted card score (Yellow + Red×3)
- **TalentScore**: Composite talent metric for young players

## 🎯 Analysis Highlights

### Player Insights:
- Identifies top performers across 6+ different metrics
- Compares actual performance vs expected metrics (xG, xA)
- Analyzes efficiency by position and age
- Discovers hidden talent among young players (U23)
- Detects statistical outliers and exceptional performers

### Club Insights:
- Benchmarks attacking output across all 20 clubs
- Evaluates goal efficiency (over/underperformance vs xG)
- Analyzes squad demographics and composition
- Identifies discipline patterns
- Groups similar clubs using hierarchical clustering

### Statistical Insights:
- Reveals strongest correlations between performance metrics
- Builds predictive models for goals and assists
- Tests normality of key distributions
- Identifies significant differences between positions and age groups
- Provides statistical validation for observed patterns

## 📊 Sample Outputs

All visualizations are saved as high-resolution PNG files (300 DPI) in the `outputs/` directory:

1. **Player Performance**: 4 comprehensive visualizations
2. **Club Analysis**: 4 detailed comparison charts
3. **Statistical Analysis**: 5 statistical visualizations
4. **Advanced Viz**: 6 creative and insightful plots

**Total: 19 high-quality visualizations**

## 🔧 Customization

### Modify Analysis Parameters:

Edit `config.py` to customize:
- Figure sizes and DPI
- Color palettes
- Minimum playing time thresholds
- Output directory location

### Add New Metrics:

In `config.py`, modify the `create_derived_features()` method to add custom calculated metrics.

### Adjust Visualizations:

Each analysis file is modular - you can comment out specific visualizations or modify parameters within each method.

## 📝 Data Requirements

The input CSV file should contain data for all EPL players from the 2020-21 season with these required columns:

**Player Information:**
- Name, Club, Nationality, Position, Age

**Match Statistics:**
- Matches, Starts, Mins (minutes played)

**Performance Metrics:**
- Goals, Assists, Passes_Attempted, Perc_Passes_Completed
- Penalty_Goals, Penalty_Attempted
- xG (expected goals), xA (expected assists)
- Yellow_Cards, Red_Cards

## 🐛 Troubleshooting

### Common Issues:

1. **ModuleNotFoundError**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **FileNotFoundError**: Verify `EPL_20_21.csv` is in the correct location
   - Check the file path in `config.py` (DATA_PATH variable)

3. **Memory Issues**: If running on limited RAM:
   - Run individual modules instead of the complete suite
   - Reduce figure sizes in `config.py`

4. **Import Errors**: Use Python 3.8+
   ```bash
   python --version
   ```

## 🎓 Learning Resources

This project demonstrates:
- Data cleaning and feature engineering
- Statistical analysis and hypothesis testing
- Regression modeling and prediction
- Advanced data visualization techniques
- Object-oriented programming in data science
- Modular code organization





## 🙏 Acknowledgments

- Data from EPL 2020-21 season
- Built with Python, Pandas, Matplotlib, Seaborn, Scikit-learn, and SciPy

---

**Last Updated**: 2026-02-09
**Version**: 1.0.0

For questions or suggestions, feel free to open an issue or contribute to the project!
