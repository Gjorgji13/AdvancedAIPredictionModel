# Automated Machine Learning Pipeline for Time Series Forecasting and Analysis

This project implements an automated machine learning pipeline designed for time series data analysis, forecasting, and model management.  It combines data preprocessing, feature engineering, model training, evaluation, and deployment within a single notebook environment.  The primary goal is to simplify the process of building and deploying machine learning models for time series forecasting, while incorporating features for dataset discovery and reuse.

## Key Features

*   **Automated Dataset Discovery:**  The pipeline automatically analyzes uploaded datasets to identify suitable target variables and features.
*   **Intelligent Feature Engineering:**  The code generates lagged and rolling window features, along with cyclical time features, to enhance model predictive power.
*   **Model Selection & Training:**  The pipeline supports multiple machine learning algorithms (Random Forest, XGBoost, simple neural network) and automatically trains models based on available data and configurations.
*   **Dataset Reuse:** This script incorporates features for re-using similar datasets by saving metadata which can be re-used to train, load, or deploy similar models.
*   **Model Management:** It supports saving and loading trained models for later use, reducing the need for repeated training.
*   **Visualization and Evaluation:** Provides visualizations for residual analysis, model evaluation, and forecasting results.
*   **Automated Pipeline and Workflow:**  It allows a streamlined automated pipeline for building ML models.

## Architecture

The notebook is structured around a main `on_process_clicked` function, which orchestrates the entire workflow.  The pipeline follows these stages:

1.  **Data Loading and Analysis:** Loads data from uploaded files (CSV, Excel) and performs initial analysis to identify target variables and features.
2.  **Data Preprocessing:** Handles missing values, performs feature engineering (lagged, rolling window, cyclical time features).
3.  **Model Training:** Trains machine learning models (Random Forest, XGBoost, MLPRegressor) using the preprocessed data.
4.  **Model Evaluation:** Evaluates model performance using various metrics and visualizations.
5.  **Model Deployment:** Saves trained models for later use and generates predictions for new data.

## Dependencies

This project requires the following Python libraries:

*   pandas
*   numpy
*   scikit-learn
*   xgboost
*   pytorch
*   pytorch-lightning
*   torch
*   xlsxwriter
*   statsmodels
*   google-colab
*   ipywidgets
*   seaborn
*   matplotlib

You can install these dependencies using:
```bash
!pip install -r requirements.txt
