# 📊 Student Habits and Performance Analysis

A modern, interactive data analytics application built with **PyQt5**, **Pandas**, and **Matplotlib**, designed to analyze and predict student performance based on lifestyle habits such as study time, sleep, diet, and mental health.

---

## 🧠 Project Overview

This desktop GUI application allows users to:
- Load and explore a dataset of student habits.
- Visualize data with pie charts, distribution plots, and boxplots.
- Analyze categorical and numerical trends using probability distributions.
- Perform linear regression to predict academic performance.
- Calculate and visualize 95% confidence intervals for predictions.
- View filtered student records in a styled data table.

---

## 📁 Dataset Description

The dataset used is `student_habits_performance.csv` and contains the following features:

- `study_hours_per_day`
- `exam_score`
- `sleep_hours`
- `attendance_percentage`
- `mental_health_rating`
- `social_media_hours`
- `netflix_hours`
- `gender`
- `diet_quality`
- `internet_quality`
- `parental_education_level`
- `part_time_job`
- `extracurricular_participation`

---

## ✨ Key Features

### 📊 Visualizations
- **Pie Charts** for categorical variable distribution.
- **PMF and PDF** plots for Binomial, Poisson, Normal, and Uniform distributions.
- **Boxplots** and descriptive statistics for numeric columns.

### 📈 Regression & Prediction
- **Linear regression** to predict:
  - Exam Score based on Study Hours
  - Attendance Percentage based on Sleep Hours
- **Prediction input** with model-generated values.
- **R² score** evaluation of model performance.

### 🎯 Confidence Intervals
- 95% prediction intervals for new inputs using t-distribution.
- Interval visualized on regression graph.

### 📋 Filtered Data Tables
- Filter views:
  - High Performers (Exam > 80)
  - Study Hours > 4
  - Sleep Hours > 7
- Styled PyQt5 table with color-coded scores.

---

## 🛠️ Technologies Used

- **Python 3**
- **PyQt5** – GUI framework
- **Pandas & NumPy** – Data manipulation
- **Matplotlib & Seaborn** – Visualization
- **Scikit-learn** – Linear regression models
- **SciPy & Statsmodels** – Probability distributions & confidence intervals

---

## 🚀 How to Run

### 📦 Prerequisites

Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy pyqt5

## Run the Project
python run.py


📚 Statistical Concepts Applied
Descriptive Statistics (mean, median, mode, IQR, etc.)

Probability Mass Function (PMF) – for Binomial & Poisson distributions

Probability Density Function (PDF) – for Normal & Uniform distributions

Linear Regression (with R² evaluation)

95% Confidence Intervals (based on t-distribution)

🙋‍♂️ Author
Muhammad Rameez
🎓 FAST-NUCES, Computer Science
