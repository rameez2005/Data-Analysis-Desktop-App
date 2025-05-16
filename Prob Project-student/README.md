# Student Habits and Performance Analysis Application

A modern, interactive data analysis application built with PyQt5 that provides in-depth analysis and visualization of student habits and their correlation with academic performance.

## Project Description

This application analyzes the relationship between student habits (study hours, sleep patterns, social media usage, etc.) and academic performance. It provides various statistical analyses, visualizations, and predictive models to help understand the factors that influence student success.

The application offers a sleek, modern user interface with an intuitive sidebar navigation and real-time data visualization capabilities. Users can explore different aspects of student behavior through pie charts, regression analyses, statistical distributions, and more.

## Features

### 1. Categorical Analysis
- Create pie charts for categorical variables (gender, diet quality, etc.)
- View detailed percentage breakdowns and mode analysis
- Identify distribution patterns in categorical student data

### 2. Regression Analysis
- Predict exam scores based on study hours
- Predict attendance percentage based on sleep hours
- View detailed regression statistics including R² scores
- Interactive prediction tool for "what-if" scenarios

### 3. Exploratory Data Analysis (EDA)
- Statistical analysis of numerical variables
- Visualization through box plots
- Detailed statistical measures including min, max, quartiles, variance, etc.
- Interpretive insights for better understanding

### 4. Probability Distributions
- Normal distribution analysis of exam scores
- Binomial distribution of study habits
- Poisson distribution of mental health ratings
- Uniform distribution visualizations

### 5. Confidence Intervals
- Calculate prediction intervals for exam scores based on study hours
- 95% confidence interval visualization
- Statistical validation of predictions

### 6. Tabular Data View
- View the complete dataset in a structured, tabular format
- Apply filters to focus on specific student segments (high performers, students with more study hours, etc.)
- Color-coded exam scores for quick performance assessment
- Summary statistics for the filtered data
- Interactive table with sortable columns

## Installation and Setup

### Prerequisites
- Python 3.6 or higher
- Pip package manager

### Installation Steps

1. Clone the repository:
```
git clone https://github.com/rameez2005/student-habits-analysis.git
cd student-habits-analysis
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Run the application:
```
python run.py
```

### Dependencies
The application requires the following Python packages:
```
PyQt5>=5.15.0
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
scikit-learn>=0.24.0
statsmodels>=0.13.0
Pillow>=8.0.0
```

## Technical Implementation

### Core Libraries
- **PyQt5**: Provides the UI framework and interactive components
- **pandas**: Used for data management and manipulation
- **matplotlib/seaborn**: Handles data visualization
- **scikit-learn**: Implements machine learning models for regression analysis
- **scipy**: Provides statistical distributions and testing
- **statsmodels**: Used for confidence interval calculations and statistical modeling

### Key Components

#### 1. Data Processing
The application reads student data from `student_habits_performance.csv` and processes it using pandas for statistical analysis.

#### 2. User Interface 
The UI is built with PyQt5, featuring:
- Modern styling with gradient backgrounds
- Interactive sidebar with cards for different analyses
- Real-time visualization display area
- Loading animations for longer calculations
- Social media links to GitHub and LinkedIn profiles

#### 3. Statistical Models
- **Linear Regression**: Used to predict exam scores and attendance percentages
- **Probability Distributions**: Normal, Binomial, Poisson, and Uniform distributions
- **Confidence Intervals**: Calculated using statsmodels for prediction validation

#### 4. Visualization Techniques
- **Pie Charts**: For categorical variable analysis
- **Scatter Plots with Regression Lines**: For regression analysis
- **Box Plots**: For numerical variable distribution analysis
- **Distribution Plots**: For probability distribution visualization

## Project Structure

- `run.py` - Entry point for the application
- `modern_project.py` - Contains the core analysis logic and processing
- `modern_ui.py` - Defines the UI components and styling
- `student_habits_performance.csv` - Dataset containing student information
- `requirements.txt` - List of project dependencies

## Usage Examples

1. **Analyzing Study Habits:**
   - Select "study_hours_per_day" in the EDA section
   - Run EDA to view statistical distribution of study patterns

2. **Predicting Exam Scores:**
   - Select "Predict exam score" in the Regression section
   - Input study hours to get predicted scores
   - View the confidence intervals for predictions

3. **Examining Categorical Distributions:**
   - Select a categorical variable like "diet_quality"
   - Generate a pie chart to view the distribution

## Contributors
- Muhammad Rameez (23F-0636)
- Malik Kamran Ali (23F-0674)

## Contact
- GitHub: [github.com/rameez2005](https://github.com/rameez2005)
- LinkedIn: [linkedin.com/in/rameez2005](https://www.linkedin.com/in/rameez2005/) 