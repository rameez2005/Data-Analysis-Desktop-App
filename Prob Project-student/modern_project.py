import sys
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import norm, poisson, uniform, binom
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QDialog, QVBoxLayout, QHeaderView, QPushButton, QLabel, QHBoxLayout, QSizePolicy
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QColor
from modern_ui import ModernMainWindow

class StudentHabitsAnalysisApp(ModernMainWindow):
    def __init__(self):
        super().__init__()
        
        # Load the dataset
        self.data = pd.read_csv('student_habits_performance.csv')
        
        # Connect button signals to slots
        self.connect_signals()
        
        # Set window title
        self.setWindowTitle("Student Habits and Performance Analysis")
    
    def connect_signals(self):
        # Connect all button click events to their respective functions
        self.SearchPie.clicked.connect(self.get_pie_chart)
        self.SearchPie_distribution.clicked.connect(self.get_distribution)
        self.SearchPie_EDA.clicked.connect(self.get_eda)
        self.SearchPie_Regression.clicked.connect(self.get_regression)
        self.SearchPie_Regression_predict.clicked.connect(self.get_regression_predict)
        self.SearchPie_Confidence.clicked.connect(self.get_Confidence)
        
        # Connect the data table button
        self.data_table_button.clicked.connect(self.get_data_table)
    
    def show_loading_and_execute(self, func):
        """Helper method to show loading animation and execute a function"""
        self.show_loading()
        
        # Use a timer to delay execution slightly to show the loading animation
        QTimer.singleShot(500, lambda: self.execute_with_loading(func))
    
    def execute_with_loading(self, func):
        """Execute the function and hide loading when done"""
        try:
            func()
        finally:
            self.hide_loading()
    
    def get_pie_chart(self):
        """Generate a pie chart for the selected categorical variable"""
        self.show_loading_and_execute(self._get_pie_chart)
    
    def _get_pie_chart(self):
        # Get the selected variable
        var = self.comboPie.currentText()
        
        # Calculate value counts
        value_counts = self.data[var].value_counts()
        
        # Select top categories and group the rest into "Other"
        top_categories = value_counts.head(10)
        other_sum = value_counts[10:].sum() if len(value_counts) > 10 else 0
        
        if other_sum != 0:
            top_categories['Other'] = other_sum
        
        # Calculate percentages
        total = top_categories.sum()
        percentages = (top_categories / total * 100).round(1)
        
        # Create labels with percentages
        labels = [f'{name}: {percent}%' for name, percent in zip(top_categories.index, percentages)]
        
        # Create a pie chart
        plt.figure(figsize=(13, 6))
        plt.pie(top_categories, labels=labels, autopct='%1.1f%%', startangle=90, 
                colors=plt.cm.Paired(range(len(top_categories))))
        plt.title(f'Pie Chart of {var} with Percentages')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        
        # Calculate the mode
        mode_value = self.data[var].mode()[0]
        
        # Create description text
        description = (
            f"Detailed Percentages for Each Category in {var}:\n" +
            "\n".join([f'{name}: {percent}%' for name, percent in zip(top_categories.index, percentages)]) +
            f"\n\nThe mode for {var} is: {mode_value}\nWhich means that in {var}\n {mode_value} is most popular"
        )
        
        # Update the output text
        self.output_text.setText(description)
    
    def get_distribution(self):
        """Generate a probability distribution for the selected type"""
        self.show_loading_and_execute(self._get_distribution)
    
    def _get_distribution(self):
        # Get the selected distribution type
        dist_type = self.comboPie_distribution.currentText()
        
        if dist_type == "Binomial Distribution":
            n = len(self.data)  # number of trials
            p = np.mean(self.data['study_hours_per_day'] > 4)  # probability of success (studying more than 4 hours)
            
            if 0 < p < 1:
                binomial_dist = binom(n=n, p=p)
                x = np.arange(binom.ppf(0.01, n, p), binom.ppf(0.99, n, p))
                plt.figure(figsize=(10, 6))
                plt.plot(x, binomial_dist.pmf(x), 'bo', ms=8, label='Binomial PMF')
                plt.title('Binomial Distribution for study hours > 4')
                plt.xlabel('Number of successes')
                plt.ylabel('Probability Mass Function (PMF)')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.show()
                
                description = f"Binomial Distribution with n={n} and p={p:.4f}"
                self.output_text.setText(description)
            else:
                self.output_text.setText(f"Error: Probability 'p' must be between 0 and 1 but got: {p}")
            
        elif dist_type == "Poisson Distribution":
            data_sample = self.data['mental_health_rating']  # Using mental health rating for Poisson
            rate = np.mean(data_sample)  # The rate parameter (lambda) for Poisson
            
            poisson_dist = poisson(rate)
            x = np.arange(poisson.ppf(0.01, rate), poisson.ppf(0.99, rate))
            
            plt.figure(figsize=(10, 6))
            plt.plot(x, poisson_dist.pmf(x), 'bo', ms=8, label='Poisson PMF')
            plt.title(f'Poisson Distribution with λ={rate:.2f}')
            plt.xlabel('Mental Health Rating')
            plt.ylabel('Probability Mass Function (PMF)')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()
            
            description = f"Poisson Distribution with λ (rate) = {rate:.2f}"
            self.output_text.setText(description)
            
        elif dist_type == "Normal Distribution":
            data_sample = self.data['exam_score']
            mu, std = norm.fit(data_sample)
            
            plt.figure(figsize=(10, 6))
            sns.histplot(data_sample, kde=False, color='blue', stat="density")
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2)
            title = f"Normal Distribution Fit: μ = {mu:.2f}, σ = {std:.2f}"
            plt.title(title)
            plt.xlabel('Exam Score')
            plt.ylabel('Density')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()
            
            description = f"Normal Distribution with μ (mean) = {mu:.2f} and σ (std) = {std:.2f}"
            self.output_text.setText(description)

        elif dist_type == "Uniform Distribution":
            data_sample = self.data['study_hours_per_day']
            min_val = min(data_sample)
            max_val = max(data_sample)
            width = max_val - min_val
            
            uniform_dist = uniform(loc=min_val, scale=width)
            x = np.linspace(min_val, max_val, 100)
            
            plt.figure(figsize=(10, 6))
            plt.plot(x, uniform_dist.pdf(x), 'r-', lw=5, alpha=0.6, label='Uniform PDF')
            plt.title(f'Uniform Distribution: min={min_val:.2f}, max={max_val:.2f}')
            plt.xlabel('Study Hours Per Day')
            plt.ylabel('Probability Density Function (PDF)')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()
            
            description = f"Uniform Distribution with min={min_val:.2f} and max={max_val:.2f}"
            self.output_text.setText(description)
    
    def get_eda(self):
        """Perform exploratory data analysis on the selected variable"""
        self.show_loading_and_execute(self._get_eda)
    
    def _get_eda(self):
        # Get the selected variable
        var = self.comboPie_EDA.currentText()
        
        # Calculate descriptive statistics
        min_val = self.data[var].min()
        max_val = self.data[var].max()
        range_val = max_val - min_val
        quartiles = self.data[var].quantile([0.25, 0.5, 0.75])
        mode = self.data[var].mode()[0]
        iqr = quartiles[0.75] - quartiles[0.25]
        variance = self.data[var].var()
        std_dev = self.data[var].std()
        
        # Store results in dictionary
        stats_dict = {
            'Min': min_val,
            'Max': max_val,
            'Range': range_val,
            'Q1': quartiles[0.25],
            'Q2 (Median)': quartiles[0.5],
            'Q3': quartiles[0.75],
            'Mode': mode,
            'IQR': iqr,
            'Variance': variance,
            'Standard Deviation': std_dev
        }
        
        # Generate box plot
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=self.data[var])
        plt.title(f'Box Plot of {var}')
        plt.xlabel(var)
        plt.show()
        
        # Create description text
        description = (
            f"Statistical Analysis for {var}:\n\n" +
            "\n".join([f"{key}: {value:.4f}" for key, value in stats_dict.items()]) +
            "\n\nINTERPRETATION:" +
            f"\n• 25% of the values are below {stats_dict['Q1']:.4f}" +
            f"\n• 50% of the values are below {stats_dict['Q2 (Median)']:.4f}, and 50% are above" +
            f"\n• 75% of the values fall below {stats_dict['Q3']:.4f}" +
            f"\n• The interquartile range (IQR) is {stats_dict['IQR']:.4f}, indicating the spread of the middle 50% of the data"
        )
        
        # Update the output text
        self.output_text.setText(description)
    
    def get_regression(self):
        """Generate a regression model for the selected prediction type"""
        self.show_loading_and_execute(self._get_regression)
    
    def _get_regression(self):
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        
        # Get the selected prediction type
        pred_type = self.comboPie_Regression.currentText()
        
        if pred_type == "Predict exam score":
            X = self.data[['study_hours_per_day']]
            y = self.data['exam_score']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create and train the model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Generate prediction points for plotting
            hours_points = np.linspace(self.data['study_hours_per_day'].min(), 
                                      self.data['study_hours_per_day'].max(), 100).reshape(-1, 1)
            predicted_values = model.predict(hours_points)
            
            # Calculate where the regression line crosses 100 (if it does)
            if model.coef_[0] > 0:  # Only if the slope is positive
                max_hours_for_100 = (100 - model.intercept_) / model.coef_[0]
                
                # If this point is within our visible range, modify the plotting
                if max_hours_for_100 <= hours_points.max() and max_hours_for_100 >= hours_points.min():
                    # Display a note about where the line is capped
                    capped_note = f"Note: Predictions capped at 100 (maximum score) at {max_hours_for_100:.2f} hours of study"
                    
                    # Create two segments - one uncapped, one capped
                    before_cap_mask = hours_points.flatten() <= max_hours_for_100
                    hours_before_cap = hours_points[before_cap_mask]
                    predicted_before_cap = predicted_values[before_cap_mask]
                    
                    hours_after_cap = hours_points[~before_cap_mask]
                    predicted_after_cap = np.full_like(hours_after_cap, 100.0)
                    
                    # Plot the results with two line segments
                    plt.figure(figsize=(12, 6))
                    sns.scatterplot(x=X.squeeze(), y=y, color='blue', alpha=0.5, label='Actual Data')
                    
                    # Plot uncapped segment
                    if len(hours_before_cap) > 0:
                        plt.plot(hours_before_cap, predicted_before_cap, color='red', linewidth=2, label='Regression Line')
                    
                    # Plot capped segment
                    if len(hours_after_cap) > 0:
                        plt.plot(hours_after_cap, predicted_after_cap, color='red', linewidth=2, linestyle='dashed', 
                                label='Capped at 100 (Maximum Score)')
                    
                    # Add annotation about capping
                    plt.annotate(capped_note, xy=(max_hours_for_100, 100), xytext=(max_hours_for_100-1, 90),
                                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7))
                else:
                    # If the cap point is outside our range, just cap the values
                    capped_values = np.minimum(predicted_values, 100.0)
                    
                    # Plot the results
                    plt.figure(figsize=(12, 6))
                    sns.scatterplot(x=X.squeeze(), y=y, color='blue', alpha=0.5, label='Actual Data')
                    plt.plot(hours_points, capped_values, color='red', linewidth=2, label='Regression Line')
            else:
                # If the slope is not positive, just plot normally with a cap at 100
                capped_values = np.minimum(predicted_values, 100.0)
                
                plt.figure(figsize=(12, 6))
                sns.scatterplot(x=X.squeeze(), y=y, color='blue', alpha=0.5, label='Actual Data')
                plt.plot(hours_points, capped_values, color='red', linewidth=2, label='Regression Line')
            
            # Set y-axis limits to make the plot more relevant (0-100 for exam score)
            plt.ylim(0, 105)  # A bit higher to make room for annotations
            
            plt.title('Exam Score vs. Study Hours Regression')
            plt.xlabel('Study Hours Per Day')
            plt.ylabel('Exam Score')
            plt.legend()
            plt.show()
            
            # Calculate model performance
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Create description text
            description = (
                f"Regression Analysis: Predicting Exam Score from Study Hours\n\n" +
                f"Model Performance:\n" +
                f"• Training R² Score: {train_score:.4f}\n" +
                f"• Testing R² Score: {test_score:.4f}\n\n" +
                f"Regression Equation:\n" +
                f"Exam Score = {model.coef_[0]:.2f} × Study Hours + {model.intercept_:.2f}\n\n" +
                f"Interpretation:\n" +
                f"• For each additional hour of study, exam score changes by {model.coef_[0]:.2f} points\n" +
                f"• The model explains {train_score*100:.1f}% of the variance in exam scores\n" +
                f"• Predictions are capped at 100 (maximum possible exam score)"
            )
            
            # Update the output text
            self.output_text.setText(description)
            
        elif pred_type == "Predict attendance":
            X = self.data[['sleep_hours']]
            y = self.data['attendance_percentage']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create and train the model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Generate prediction points for plotting
            sleep_points = np.linspace(self.data['sleep_hours'].min(), 
                                      self.data['sleep_hours'].max(), 100).reshape(-1, 1)
            predicted_values = model.predict(sleep_points)
            
            # Calculate where the regression line crosses 100% (if it does)
            if model.coef_[0] > 0:  # Only if the slope is positive
                max_sleep_for_100 = (100 - model.intercept_) / model.coef_[0]
                
                # If this point is within our visible range, modify the plotting
                if max_sleep_for_100 <= sleep_points.max() and max_sleep_for_100 >= sleep_points.min():
                    # Display a note about where the line is capped
                    capped_note = f"Note: Predictions capped at 100% at {max_sleep_for_100:.2f} hours of sleep"
                    
                    # Create two segments - one uncapped, one capped
                    before_cap_mask = sleep_points.flatten() <= max_sleep_for_100
                    sleep_before_cap = sleep_points[before_cap_mask]
                    predicted_before_cap = predicted_values[before_cap_mask]
                    
                    sleep_after_cap = sleep_points[~before_cap_mask]
                    predicted_after_cap = np.full_like(sleep_after_cap, 100.0)
                    
                    # Plot the results with two line segments
                    plt.figure(figsize=(12, 6))
                    sns.scatterplot(x=X.squeeze(), y=y, color='green', alpha=0.5, label='Actual Data')
                    
                    # Plot uncapped segment
                    if len(sleep_before_cap) > 0:
                        plt.plot(sleep_before_cap, predicted_before_cap, color='red', linewidth=2, label='Regression Line')
                    
                    # Plot capped segment
                    if len(sleep_after_cap) > 0:
                        plt.plot(sleep_after_cap, predicted_after_cap, color='red', linewidth=2, linestyle='dashed', 
                                label='Capped at 100% (Maximum Attendance)')
                    
                    # Add annotation about capping
                    plt.annotate(capped_note, xy=(max_sleep_for_100, 100), xytext=(max_sleep_for_100-1, 90),
                                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7))
                else:
                    # If the cap point is outside our range, just cap the values
                    capped_values = np.minimum(predicted_values, 100.0)
                    
                    # Plot the results
                    plt.figure(figsize=(12, 6))
                    sns.scatterplot(x=X.squeeze(), y=y, color='green', alpha=0.5, label='Actual Data')
                    plt.plot(sleep_points, capped_values, color='red', linewidth=2, label='Regression Line')
            else:
                # If the slope is not positive, just plot normally with a cap at 100
                capped_values = np.minimum(predicted_values, 100.0)
                
                plt.figure(figsize=(12, 6))
                sns.scatterplot(x=X.squeeze(), y=y, color='green', alpha=0.5, label='Actual Data')
                plt.plot(sleep_points, capped_values, color='red', linewidth=2, label='Regression Line')
            
            # Set y-axis limits to make the plot more relevant (0-100 for attendance)
            plt.ylim(0, 105)  # A bit higher to make room for annotations
            
            plt.title('Attendance Percentage vs. Sleep Hours Regression')
            plt.xlabel('Sleep Hours')
            plt.ylabel('Attendance Percentage')
            plt.legend()
            plt.show()
            
            # Calculate model performance
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Create description text
            description = (
                f"Regression Analysis: Predicting Attendance Percentage from Sleep Hours\n\n" +
                f"Model Performance:\n" +
                f"• Training R² Score: {train_score:.4f}\n" +
                f"• Testing R² Score: {test_score:.4f}\n\n" +
                f"Regression Equation:\n" +
                f"Attendance Percentage = {model.coef_[0]:.2f} × Sleep Hours + {model.intercept_:.2f}\n\n" +
                f"Interpretation:\n" +
                f"• For each additional hour of sleep, attendance percentage changes by {model.coef_[0]:.2f}%\n" +
                f"• The model explains {train_score*100:.1f}% of the variance in attendance percentage\n" +
                f"• Predictions are capped at 100% (maximum possible attendance)"
            )
            
            # Update the output text
            self.output_text.setText(description)
    
    def get_regression_predict(self):
        """Make a prediction using the regression model"""
        self.show_loading_and_execute(self._get_regression_predict)
    
    def _get_regression_predict(self):
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        
        # Get the selected prediction type and input value
        pred_type = self.comboPie_Regression_predict.currentText()
        input_value = self.spinBox_predict.value()
        
        # Predicting Exam Score based on Study Hours
        X_exam = self.data[['study_hours_per_day']]
        y_exam = self.data['exam_score']
        
        # Splitting the data for the exam score model
        X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_exam, y_exam, test_size=0.2, random_state=42)
        
        # Create and train the exam score model
        model_exam = LinearRegression()
        model_exam.fit(X_train_e, y_train_e)
        
        # Predicting Attendance based on Sleep Hours
        X_attend = self.data[['sleep_hours']]
        y_attend = self.data['attendance_percentage']
        
        # Splitting the data for the attendance model
        X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_attend, y_attend, test_size=0.2, random_state=42)
        
        # Create and train the attendance model
        model_attend = LinearRegression()
        model_attend.fit(X_train_a, y_train_a)
        
        # Use the models to predict based on the input value
        input_point = [[input_value]]
        
        if pred_type == "Predict exam score":
            # Get raw prediction
            predicted_exam = model_exam.predict(input_point)
            
            # Cap the prediction at 100 (maximum possible exam score)
            capped_exam_score = min(predicted_exam[0], 100.0)
            
            description = (
                f"Prediction Results:\n\n" +
                f"For {input_value} hours of study per day:\n" +
                f"• Predicted Exam Score: {capped_exam_score:.2f}\n\n" +
                f"Model Performance:\n" +
                f"• Training R² Score: {model_exam.score(X_train_e, y_train_e):.4f}\n" +
                f"• Testing R² Score: {model_exam.score(X_test_e, y_test_e):.4f}"
            )
            
            # Add note if prediction was capped
            if predicted_exam[0] > 100.0:
                description += f"\n\nNote: Raw prediction ({predicted_exam[0]:.2f}) exceeded maximum possible score and was capped at 100."
                
            self.output_text.setText(description)
            
        elif pred_type == "Predict attendance":
            predicted_attend = model_attend.predict(input_point)
            
            # Cap attendance percentage at 100%
            capped_attendance = min(predicted_attend[0], 100.0)
            
            description = (
                f"Prediction Results:\n\n" +
                f"For {input_value} hours of sleep per day:\n" +
                f"• Predicted Attendance Percentage: {capped_attendance:.2f}%\n\n" +
                f"Model Performance:\n" +
                f"• Training R² Score: {model_attend.score(X_train_a, y_train_a):.4f}\n" +
                f"• Testing R² Score: {model_attend.score(X_test_a, y_test_a):.4f}"
            )
            
            # Add note if prediction was capped
            if predicted_attend[0] > 100.0:
                description += f"\n\nNote: Raw prediction ({predicted_attend[0]:.2f}%) exceeded 100% and was capped."
                
            self.output_text.setText(description)
    
    def get_Confidence(self):
        """Calculate and display confidence intervals for predictions"""
        self.show_loading_and_execute(self._get_Confidence)
    
    def _get_Confidence(self):
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        import scipy.stats as stats
        
        # Get the selected confidence interval type
        interval_type = self.comboPie_Confidence.currentText()
        
        if interval_type == "Exam Score vs Study Hours":
            # Get study hours from user input
            study_hours = self.spinBox_km.value()
            
            # Get the data for regression
            x = self.data['study_hours_per_day']
            y = self.data['exam_score']
            
            # Manually calculate regression parameters
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Calculate predicted value for the input study hours
            raw_predicted_score = intercept + slope * study_hours
            
            # Cap the predicted score at 100 (maximum possible exam score)
            predicted_score = min(raw_predicted_score, 100.0)
            
            # Calculate the standard error of the regression
            n = len(x)
            y_pred = intercept + slope * x
            residuals = y - y_pred
            residual_std = np.sqrt(np.sum(residuals**2) / (n-2))
            
            # Calculate prediction interval
            x_mean = np.mean(x)
            x_std = np.std(x)
            # Standard error of prediction
            se_pred = residual_std * np.sqrt(1 + 1/n + (study_hours - x_mean)**2 / ((n-1) * x_std**2))
            
            # t-value for 95% confidence
            t_value = stats.t.ppf(0.975, n-2)
            
            # Lower and upper bounds of prediction interval
            raw_lower = raw_predicted_score - t_value * se_pred
            raw_upper = raw_predicted_score + t_value * se_pred
            
            # Cap the prediction interval at 100
            lower = max(0, raw_lower)  # Also prevent negative scores
            upper = min(raw_upper, 100.0)
            
            capped_message = ""
            if raw_predicted_score > 100.0:
                capped_message = f"\n• Note: Raw prediction ({raw_predicted_score:.2f}) exceeded maximum possible score and was capped at 100."
            
            if raw_upper > 100.0:
                capped_message += f"\n• Note: Upper bound of prediction interval ({raw_upper:.2f}) was capped at 100."
            
            description = (
                f"Predicted Exam Score for {study_hours:,.1f} hours of study per day:\n"
                f"     {predicted_score:.2f} points\n\n"
                f"95% prediction interval for a student studying {study_hours:,.1f} hours per day: \n"
                f"     {lower:.2f} to {upper:.2f} points\n\n"
                f"Interpretation:\n"
                f"• With 95% confidence, a student studying {study_hours:,.1f} hours per day will score\n"
                f"  between {lower:.2f} and {upper:.2f} on the exam.\n"
                f"• The correlation between study hours and exam score is: {r_value:.4f}\n"
                f"• The regression equation is: Score = {intercept:.2f} + {slope:.2f} × Study Hours"
                f"{capped_message}"
            )
            
            # Plot the confidence interval
            plt.figure(figsize=(10, 6))
            
            # Plot the actual data points
            plt.scatter(x, y, alpha=0.5, color='blue', label='Actual Data')
            
            # Generate x values for the regression line
            x_line = np.linspace(min(x), max(x), 100)
            # Calculate y values for the regression line
            y_line = intercept + slope * x_line
            
            # Cap the regression line at 100
            y_line = np.minimum(y_line, 100.0)
            
            # Plot the regression line
            plt.plot(x_line, y_line, color='red', label='Regression Line')
            
            # Highlight the prediction point
            plt.scatter([study_hours], [predicted_score], color='green', s=100, 
                        label=f'Prediction: {predicted_score:.2f}')
            
            # Add a vertical interval line for the prediction
            plt.vlines(x=study_hours, ymin=lower, ymax=upper, 
                      colors='green', linestyles='dashed', label='95% Prediction Interval')
            
            plt.axhline(y=lower, color='gray', linestyle='dotted')
            plt.axhline(y=upper, color='gray', linestyle='dotted')
            
            # Set y-axis limits to make the plot more relevant (0-100 for exam score)
            plt.ylim(0, 100)
            
            plt.xlabel('Study Hours Per Day')
            plt.ylabel('Exam Score')
            plt.title('Exam Score Prediction with 95% Confidence Interval')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        elif interval_type == "Attendance vs Sleep Hours":
            # Get sleep hours from user input
            sleep_hours = self.spinBox_km.value()
            
            # Get the data for regression
            x = self.data['sleep_hours']
            y = self.data['attendance_percentage']
            
            # Manually calculate regression parameters
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Calculate predicted value for the input sleep hours
            raw_predicted_attendance = intercept + slope * sleep_hours
            
            # Cap the attendance percentage at 100%
            predicted_attendance = min(raw_predicted_attendance, 100.0)
            
            # Calculate the standard error of the regression
            n = len(x)
            y_pred = intercept + slope * x
            residuals = y - y_pred
            residual_std = np.sqrt(np.sum(residuals**2) / (n-2))
            
            # Calculate prediction interval
            x_mean = np.mean(x)
            x_std = np.std(x)
            # Standard error of prediction
            se_pred = residual_std * np.sqrt(1 + 1/n + (sleep_hours - x_mean)**2 / ((n-1) * x_std**2))
            
            # t-value for 95% confidence
            t_value = stats.t.ppf(0.975, n-2)
            
            # Lower and upper bounds of prediction interval
            raw_lower = raw_predicted_attendance - t_value * se_pred
            raw_upper = raw_predicted_attendance + t_value * se_pred
            
            # Cap the prediction interval at 100% and prevent negative values
            lower = max(0, raw_lower)
            upper = min(raw_upper, 100.0)
            
            capped_message = ""
            if raw_predicted_attendance > 100.0:
                capped_message = f"\n• Note: Raw prediction ({raw_predicted_attendance:.2f}%) exceeded 100% and was capped."
            
            if raw_upper > 100.0:
                capped_message += f"\n• Note: Upper bound of prediction interval ({raw_upper:.2f}%) was capped at 100%."
            
            description = (
                f"Predicted Attendance Percentage for {sleep_hours:,.1f} hours of sleep per day:\n"
                f"     {predicted_attendance:.2f}%\n\n"
                f"95% prediction interval for a student sleeping {sleep_hours:,.1f} hours per day: \n"
                f"     {lower:.2f}% to {upper:.2f}%\n\n"
                f"Interpretation:\n"
                f"• With 95% confidence, a student sleeping {sleep_hours:,.1f} hours per day will have\n"
                f"  between {lower:.2f}% and {upper:.2f}% attendance.\n"
                f"• The correlation between sleep hours and attendance is: {r_value:.4f}\n"
                f"• The regression equation is: Attendance = {intercept:.2f} + {slope:.2f} × Sleep Hours"
                f"{capped_message}"
            )
            
            # Plot the confidence interval
            plt.figure(figsize=(10, 6))
            
            # Plot the actual data points
            plt.scatter(x, y, alpha=0.5, color='blue', label='Actual Data')
            
            # Generate x values for the regression line
            x_line = np.linspace(min(x), max(x), 100)
            # Calculate y values for the regression line
            y_line = intercept + slope * x_line
            
            # Cap the regression line at 100%
            y_line = np.minimum(y_line, 100.0)
            
            # Plot the regression line
            plt.plot(x_line, y_line, color='red', label='Regression Line')
            
            # Highlight the prediction point
            plt.scatter([sleep_hours], [predicted_attendance], color='green', s=100, 
                        label=f'Prediction: {predicted_attendance:.2f}%')
            
            # Add a vertical interval line for the prediction
            plt.vlines(x=sleep_hours, ymin=lower, ymax=upper, 
                      colors='green', linestyles='dashed', label='95% Prediction Interval')
            
            plt.axhline(y=lower, color='gray', linestyle='dotted')
            plt.axhline(y=upper, color='gray', linestyle='dotted')
            
            # Set y-axis limits to make the plot more relevant (0-100 for attendance percentage)
            plt.ylim(0, 100)
            
            plt.xlabel('Sleep Hours Per Day')
            plt.ylabel('Attendance Percentage')
            plt.title('Attendance Prediction with 95% Confidence Interval')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        self.output_text.setText(description)

    def get_data_table(self):
        """Display the data in a tabular format based on the selected filter"""
        self.show_loading_and_execute(self._get_data_table)
    
    def _get_data_table(self):
        # Get the selected filter
        filter_option = self.data_table_filter_combo.currentText()
        
        # Apply filter to the data
        if filter_option == "All Data":
            filtered_data = self.data
        elif filter_option == "High Performers (Exam > 80)":
            filtered_data = self.data[self.data['exam_score'] > 80]
        elif filter_option == "Study Hours > 4":
            filtered_data = self.data[self.data['study_hours_per_day'] > 4]
        elif filter_option == "Sleep Hours > 7":
            filtered_data = self.data[self.data['sleep_hours'] > 7]
        else:
            filtered_data = self.data
        
        # Create a dialog to display the table
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Student Data - {filter_option}")
        dialog.setMinimumSize(800, 600)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
            }
            QTableWidget {
                background-color: #242424;
                color: white;
                gridline-color: #3d5afe;
                border: 1px solid #3d5afe;
                border-radius: 10px;
                selection-background-color: #3d5afe;
                selection-color: white;
            }
            QTableWidget::item {
                padding: 5px;
                border-bottom: 1px solid rgba(61, 90, 254, 0.3);
            }
            QTableWidget::item:selected {
                background-color: #3d5afe;
            }
            QHeaderView::section {
                background-color: #121212;
                color: white;
                padding: 6px;
                border: 1px solid #3d5afe;
                font-weight: bold;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3d5afe, stop:1 #1a237e);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                min-height: 35px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #536dfe, stop:1 #283593);
            }
            QLabel {
                color: white;
                font-size: 12px;
            }
        """)
        
        # Create a layout for the dialog
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Add a header with stats about the filtered data
        header_layout = QHBoxLayout()
        record_count = QLabel(f"Records: {len(filtered_data)}")
        avg_study = QLabel(f"Avg Study: {filtered_data['study_hours_per_day'].mean():.2f} hrs")
        avg_score = QLabel(f"Avg Score: {filtered_data['exam_score'].mean():.2f}")
        
        header_layout.addWidget(record_count)
        header_layout.addWidget(avg_study)
        header_layout.addWidget(avg_score)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Create a table widget to display the data
        table = QTableWidget()
        
        # Set the number of rows and columns
        rows, cols = filtered_data.shape
        table.setRowCount(rows)
        table.setColumnCount(cols)
        
        # Set the column headers
        table.setHorizontalHeaderLabels(filtered_data.columns)
        
        # Fill the table with data
        for i in range(rows):
            for j in range(cols):
                value = str(filtered_data.iloc[i, j])
                item = QTableWidgetItem(value)
                
                # Center align the text
                item.setTextAlignment(Qt.AlignCenter)
                
                # Color high exam scores in green
                if j == 15:  # exam_score column
                    try:
                        score = float(value)
                        if score >= 90:
                            item.setForeground(QColor(0, 255, 0))  # Green for high scores
                        elif score >= 70:
                            item.setForeground(QColor(255, 165, 0))  # Orange for medium scores
                        elif score < 50:
                            item.setForeground(QColor(255, 0, 0))  # Red for low scores
                    except:
                        pass
                
                table.setItem(i, j, item)
        
        # Resize columns to contents
        header = table.horizontalHeader()
        for i in range(cols):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        
        # Make the table scrollable and stretch to fill available space
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(table)
        
        # Add a close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)
        
        # Show the dialog
        dialog.exec_()
        
        # Add summary to output text
        summary = (
            f"Data Summary for {filter_option}:\n\n"
            f"Total Records: {len(filtered_data)}\n"
            f"Average Study Hours: {filtered_data['study_hours_per_day'].mean():.2f}\n"
            f"Average Sleep Hours: {filtered_data['sleep_hours'].mean():.2f}\n"
            f"Average Exam Score: {filtered_data['exam_score'].mean():.2f}\n"
            f"Highest Exam Score: {filtered_data['exam_score'].max():.2f}\n"
            f"Lowest Exam Score: {filtered_data['exam_score'].min():.2f}\n\n"
            f"The data table has been displayed in a separate window."
        )
        
        self.output_text.setText(summary)

def main():
    app = QApplication(sys.argv)
    window = StudentHabitsAnalysisApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 































































