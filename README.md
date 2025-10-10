Customer Churn Prediction using Machine Learning
**Project Goal**
The primary goal of this project is to build and compare robust machine learning models to accurately predict customer churn from structured, multi-dimensional business data. By identifying customers who are likely to leave, businesses can implement proactive retention strategies to maximize customer lifetime value and minimize revenue loss.

**Key Features & Techniques**
This project encompasses the entire machine learning pipeline, from multi-source data integration to model optimization and evaluation.

Data Integration & Preprocessing: The workflow combines five distinct datasets—Customer Demographics, Transaction History, Customer Service, Online Activity, and Churn Status—from a large-scale Excel file (Customer_Churn_Data_Large.xlsx) into a single, comprehensive DataFrame.

Feature Engineering: Categorical features are encoded using one-hot encoding, and numerical features are standardized using StandardScaler for optimal model performance.

Model Comparison: The project compares the performance of three powerful classification algorithms:

XGBoost (Extreme Gradient Boosting)

Random Forest Classifier

Logistic Regression

Model Optimization: The notebook utilizes RandomizedSearchCV for efficient hyperparameter tuning to find the best model configuration for each algorithm.

Imbalance Handling: The dataset exhibits a significant class imbalance (4:1 ratio of No Churn to Churn cases). The XGBoost model specifically addresses this by incorporating a calculated scale_pos_weight to focus attention on the minority (churn) class.

Evaluation Metric: Model performance is primarily evaluated using the ROC-AUC (Receiver Operating Characteristic - Area Under Curve) score, which is ideal for binary classification on imbalanced datasets.

**Project Structure & Files**
File Name	Description
**Lloyds.ipynb**	Data Preparation and Feature Engineering. This Jupyter Notebook contains the steps for reading the multi-sheet Excel data, checking for missing values, merging the five data sources, and applying preprocessing techniques like encoding and scaling.
**Churn Prediction**	Model Training, Optimization, and Evaluation. This file builds upon the preprocessed data, defining the prediction task, splitting the data, implementing imbalance handling, performing hyperparameter tuning (RandomizedSearchCV), and comparing the final performance of the three models.
**Customer_Churn_Data_Large.xlsx**	The raw, multi-sheet input data file used for the analysis.
**Customer_Churn_Processed.csv**	The processed dataset used as input for the model training phase.

Export to Sheets
**Installation and Setup**
This project requires a standard Python data science environment.

Clone the repository (or download the files).

Install the necessary libraries:

Bash

pip install pandas numpy scikit-learn xgboost matplotlib seaborn openpyxl
(Note: openpyxl is needed to read the .xlsx file)

Ensure data files are present: Place Customer_Churn_Data_Large.xlsx and Customer_Churn_Processed.csv in the root directory alongside the notebooks, or update the file paths within the notebooks.

**How to Run the Project**
The project is executed sequentially through the two main notebooks:

Run Lloyds.ipynb:

Execute all cells in this notebook to perform data loading, merging, cleaning, and the initial feature engineering.

Run Churn Prediction:

Execute all cells in this notebook. It will load the processed data, split it into training and testing sets, train the Logistic Regression, Random Forest, and XGBoost models, optimize them, and output performance metrics (classification reports and ROC-AUC scores) for comparison.
