import pandas as pd
import numpy as np

# Load the student data
data = pd.read_csv('student_habits_performance.csv')

# Print basic information
print("Data shape:", data.shape)
print("\nData types:")
print(data.dtypes)

# Print first 5 rows
print("\nFirst 5 rows:")
print(data.head())

# Print summary statistics for numerical columns
print("\nSummary statistics for numerical columns:")
print(data.describe())

# Check unique values in categorical columns
categorical_cols = ['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 
                   'internet_quality', 'extracurricular_participation']
print("\nUnique values in categorical columns:")
for col in categorical_cols:
    print(f"{col}: {data[col].unique()}") 