# Forecasting Energy Consumption for PJM Interconnection LLC (PJM) using XGBoost Algorithm

This code aims to forecast the energy consumption for PJM Interconnection LLC (PJM), a regional transmission organization (RTO) in the United States, utilizing the XGBoost algorithm. 

## Overview

The code performs the following steps:

1. Data Loading and Preprocessing:
   - Loads the energy consumption data from 'PJME_hourly.csv'.
   - Sets the 'Datetime' column as the index and converts it to datetime format.

2. Data Visualization:
   - Plots the energy consumption data over time to visualize the trends.

3. Train-Test Split:
   - Splits the data into training and test sets, with the cutoff date set at '01-01-2016'.

4. Feature Creation:
   - Creates additional time-related features from the datetime index to improve model performance.

5. Model Creation and Training:
   - Initializes an XGBoost regressor with specified parameters.
   - Fits the XGBoost model to the training data and evaluates its performance on both training and test sets.

6. Feature Importance Visualization:
   - Calculates and visualizes the importance of features in predicting energy consumption.

7. Evaluation Metrics:
   - Calculates evaluation metrics including Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and R-squared (Coefficient of Determination).

8. Visualization of Predictions:
   - Plots the true energy consumption data against the predicted values to assess model performance visually.

## Requirements
- pandas
- numpy
- matplotlib
- seaborn
- xgboost
- scikit-learn

## Usage
1. Ensure all required libraries are installed.
2. Place the 'PJME_hourly.csv' file in the same directory as the code.
3. Run the code.


## Output:

- R-squared (Coefficient of Determination): 0.62
- Mean Squared Error (MSE): 15396624.75
- Root Mean Squared Error (RMSE): 3923.85
- Mean Absolute Percentage Error (MAPE): 10.03
