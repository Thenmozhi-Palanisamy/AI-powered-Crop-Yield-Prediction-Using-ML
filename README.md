# AI-powered-Crop-Yield-Prediction-Using-MACHINE LEARINING

# AIM
The aim of AI-powered crop yield prediction using machine learning is to accurately forecast crop production by analyzing environmental factors like temperature, rainfall, and pesticide use, helping farmers make data-driven decisions. This approach enhances agricultural efficiency, reduces risks, and promotes sustainable farming practices.

## MATERIALS REQUIRED:
Datasets,CPU,Python,Libraries.

## THEORY:

AI-powered crop yield prediction leverages machine learning algorithms to analyze various factors affecting agricultural productivity. By incorporating historical data on crop yields, climate conditions, soil health, and pesticide usage, models can identify patterns and relationships that influence yields. Machine learning techniques, such as regression analysis and decision trees, can forecast future yields based on current and historical input data.

The process begins with data collection, followed by preprocessing to handle missing values and normalize data. Features are engineered to enhance model performance, and data is divided into training and testing sets to validate the model’s accuracy. Once trained, the model can predict yields for different crops in various regions under changing environmental conditions.

This predictive capability allows farmers to make informed decisions about resource allocation, planting strategies, and crop selection, ultimately improving productivity and sustainability in agriculture. By reducing risks associated with crop failures, AI enhances food security and supports efficient farming practices. The integration of AI in agriculture represents a significant advancement toward data-driven farming solutions.

## STEPS:
## STEP 1:
Read the Given Data

## STEP 2: 
Clean the Data Set Using Data Cleaning Process

## STEP 3: 
Apply Feature Generation Techniques to All Features of the Data Set

## STEP 4: 
Split the Data into Training and Testing Sets

## STEP 5: 
Train the Machine Learning Model

## STEP 6: 
Evaluate the Model Performance

## STEP 7: 
Save the Model and Processed Data

## PROGRAM:

# Import Libraries
```
import numpy as np
import pandas as pd
import os
import seaborn as sns
import missingno
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
```

# Read Datasets
```
yield_data = pd.read_csv('yield.csv')
temp_data = pd.read_csv('temp.csv')
rainfall_data = pd.read_csv('rainfall.csv')
pesticides_data = pd.read_csv('pesticides.csv')
```
# Data Cleaning and Preprocessing
```
yield_data = yield_data[['Area', 'Item', 'Year', 'Value']]
temp_data.rename(columns={'year': 'Year', 'country': 'Area'}, inplace=True)
pesticides_data = pesticides_data[['Area', 'Year', 'Value']]
yield_data = yield_data[(yield_data['Item'] == "Rice, paddy") & (yield_data['Year'] >= 1961)]
```
# Merge Datasets
```
yield_final = pd.merge(yield_data, temp_data, on=['Year', 'Area'])
yield_final = pd.merge(yield_final, pesticides_data, on=['Year', 'Area'], suffixes=('_yield', '_pesticide'))
rainfall_data.rename(columns={' Area': 'Area'}, inplace=True)
yield_final = pd.merge(yield_final, rainfall_data, on=['Year', 'Area'])
```
# Handle Missing Values and Convert Data Types
```
yield_final['average_rain'] = pd.to_numeric(yield_final['average_rain_fall_mm_per_year'], errors='coerce')
yield_final.dropna(inplace=True)
```
# Feature Engineering
```
yield_final['Pesticides_log'] = np.log1p(yield_final['Value_pesticide'])  # Log transform to avoid -inf
yield_final['rain_temp'] = yield_final['avg_temp'] * yield_final['average_rain']
```
# Encode Categorical Variables
```
datacorr = yield_final.copy()
for column in datacorr.select_dtypes(include=['object']).columns:
    datacorr[column] = LabelEncoder().fit_transform(datacorr[column])
```
# Visualize Data
```
sns.heatmap(datacorr.corr(), cmap='PuOr', annot=True, fmt=".2f", linewidths=0.5)
plt.show()
```
# Create Categories for Yield Value
```
datacorr['Yield_Value_Cat'] = pd.cut(
    datacorr['Value_yield'], bins=[0, 32500, 50000, 75000, 90000, np.inf], labels=[1, 2, 3, 4, 5]
)
```

# Train-Test Split with Stratified Sampling
```
X = datacorr.drop(['Value_yield', 'Yield_Value_Cat'], axis=1)
y = datacorr['Value_yield']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=datacorr['Yield_Value_Cat'], random_state=1812
)
```

# Feature Scaling
```
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
```
# Initialize Models
```
models = [
    ('Linear Regression', LinearRegression()),
    ('Decision Tree', DecisionTreeRegressor(random_state=1812)),
    ('Random Forest', RandomForestRegressor(random_state=1812))
]
```

# Train and Evaluate Models
````
for name, model in models:
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
`````
# Metrics Calculation
`
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    print(f'{name} - MSE: {mse:.2f}, R²: {r2:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%')
    `
# Scatter Plot of Actual vs Predicted
``
    plt.scatter(y_test, y_pred, s=10, color='#3c7b9b')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', linewidth=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{name} Evaluation')
    plt.show()
`
``
## OUTPUT:








