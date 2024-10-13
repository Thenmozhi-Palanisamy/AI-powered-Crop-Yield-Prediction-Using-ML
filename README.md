# AI-powered-Crop-Yield-Prediction-Using-MACHINE LEARINING

# AIM
The aim of AI-powered crop yield prediction using machine learning is to accurately forecast crop production by analyzing environmental factors like temperature, rainfall, and pesticide use, helping farmers make data-driven decisions. This approach enhances agricultural efficiency, reduces risks, and promotes sustainable farming practices.

## MATERIALS REQUIRED:
Datasets,CPU,Python,Libraries.

## THEORY:

AI-powered crop yield prediction leverages machine learning algorithms to analyze various factors affecting agricultural productivity. By incorporating historical data on crop yields, climate conditions, soil health, and pesticide usage, models can identify patterns and relationships that influence yields. Machine learning techniques, such as regression analysis and decision trees, can forecast future yields based on current and historical input data.

The process begins with data collection, followed by preprocessing to handle missing values and normalize data. Features are engineered to enhance model performance, and data is divided into training and testing sets to validate the model’s accuracy. Once trained, the model can predict yields for different crops in various regions under changing environmental conditions.

This predictive capability allows farmers to make informed decisions about resource allocation, planting strategies, and crop selection, ultimately improving productivity and sustainability in agriculture. By reducing risks associated with crop failures, AI enhances food security and supports efficient farming practices. The integration of AI in agriculture represents a significant advancement toward data-driven farming solutions.

## ALGORITHMS

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
```
## DEVELOPED BY:THENMOZHI P(212221230116)
## DEVELOPED BY:BALAJI J(212221243001)
```
# Import Libraries
```
import numpy as np
import pandas as pd
import os
import missingno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, LabelEncoder
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# Ensure plots display inline in Jupyter notebooks
%matplotlib inline  

# List files in the directory (if using Kaggle environment)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Read CSV files
yield_data = pd.read_csv('yield.csv')
temp_data = pd.read_csv('temp.csv')
rainfall_data = pd.read_csv('rainfall.csv')
pesticides_data = pd.read_csv('pesticides.csv')
yield_df = pd.read_csv('yield_df.csv')

# Keep only needed columns in yield_data
yield_data = yield_data[['Area', 'Item', 'Year', 'Value']]
yield_data.head(1)

# Match column titles with yield_data
temp_data.rename(columns={'year': 'Year', 'country': "Area"}, inplace=True)
temp_data.head(1)

# Keep all rainfall_data
rainfall_data.head(1)

# Keep only needed columns in pesticides_data
pesticides_data = pesticides_data[['Area', 'Year', 'Value']]
pesticides_data.head(1)

# Filter data
temp_data = temp_data[temp_data.Year >= 1961]
yield_data = yield_data[yield_data.Item == "Rice, paddy"]

# Merge datasets
yield_final = pd.merge(yield_data, temp_data, on=['Year', 'Area'])
yield_final = pd.merge(yield_final, pesticides_data, on=['Year', 'Area'])
yield_final.rename(columns={'Value_x': "Yield_Value", 'Value_y': 'Pesticides_Value'}, inplace=True)
rainfall_data.rename(columns={' Area': 'Area'}, inplace=True)
yield_final = pd.merge(yield_final, rainfall_data, on=['Year', 'Area'])
yield_final.rename(columns={'average_rain_fall_mm_per_year': 'average_rain'}, inplace=True)
yield_final["average_rain"] = pd.to_numeric(yield_final["average_rain"], errors='coerce')  # Replace non-numbers with NaN

# Visualize missing data
missingno.bar(yield_final, figsize=(5, 3))
missingno.matrix(yield_final, figsize=(5, 3))

# Drop rows with NaN values and keep relevant columns
yield_final = yield_final.dropna()
yield_final = yield_final[['Area', 'Item', 'Year', "avg_temp", "Pesticides_Value", "average_rain", "Yield_Value"]]
yield_final.info()

# Visualize missing data again
missingno.bar(yield_final, figsize=(5, 3))
missingno.matrix(yield_final, figsize=(5, 3))
yield_final.describe()

# Plot histograms
yield_final.hist(bins=25, figsize=(20, 15))

# Log transformation and feature engineering
yield_final['Pesticides_log'] = np.log(yield_final['Pesticides_Value'])
yield_final['Pesticides_log'].hist(bins=25, figsize=(5, 3))

yield_final['rain_temp'] = yield_final['avg_temp'] * yield_final['average_rain']
yield_final['Pesticides_rain'] = np.log(yield_final['Pesticides_Value'] / yield_final['average_rain'])
yield_final['Pesticides_temp'] = np.log(yield_final['Pesticides_Value'] / yield_final['avg_temp'])
yield_final['Pesticides_temp_rain'] = (yield_final['Pesticides_temp'] / yield_final['Pesticides_rain'])
yield_final['rain_log'] = np.log(yield_final['avg_temp'])
yield_final['temp_rainlog'] = yield_final['avg_temp'] / yield_final['rain_log']

# Calculate correlation
num_cols = ['avg_temp', 'Pesticides_Value', 'average_rain', 'rain_temp', 'Pesticides_rain', 
            'Pesticides_temp', 'Pesticides_temp_rain', 'Pesticides_log', 'rain_log', 
            'temp_rainlog', 'Yield_Value']
corr_matrix = yield_final[num_cols].corr()
corr_matrix["Yield_Value"].sort_values(ascending=False)

# Data correlation visualization
datacorr = yield_final.copy()
categorical_columns = datacorr.select_dtypes(include=['object']).columns.tolist()
label_encoder = LabelEncoder()
for column in categorical_columns:
    datacorr[column] = label_encoder.fit_transform(datacorr[column])

sns.heatmap(datacorr.corr(), cmap='PuOr')
scatter_matrix(yield_final[num_cols], figsize=(12, 8))
plt.show()

# Yield value categorization
datacorr['Yield_Value_Cat'] = pd.cut(datacorr['Yield_Value'],
                                      bins=[0., 32500, 50000, 75000, 90000, np.inf],
                                      labels=[1, 2, 3, 4, 5])
datacorr['Yield_Value_Cat'].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Yield_Value_Cat")
plt.ylabel("Number of districts")
plt.show()

# Train-test split with stratification
strat_train_set, strat_test_set = train_test_split(
    datacorr, test_size=0.1, stratify=datacorr["Yield_Value_Cat"], random_state=1812)

# Check percentage distribution
strat_test_set["Yield_Value_Cat"].value_counts() / len(strat_test_set)
datacorr["Yield_Value_Cat"].value_counts() / len(datacorr)

# Prepare data for modeling
X = datacorr.drop("Yield_Value", axis=1)
Y = datacorr["Yield_Value"].copy()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=datacorr["Yield_Value_Cat"], random_state=1812)

# Scaling features
std_scaler = StandardScaler()
X_train_sc = std_scaler.fit_transform(X_train)
X_test_sc = std_scaler.transform(X_test)

# Model training and evaluation
results = []
models = [
    ('Linear Regression', LinearRegression()),
    ('Decision Tree', DecisionTreeRegressor(random_state=1812)),
    ('Random Forest', RandomForestRegressor(random_state=1812))
]

for name, model in models:
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    
    # Metrics evaluation
    MSE = mean_squared_error(y_test, y_pred)
    R2_score = r2_score(y_test, y_pred)
    
    print(f'The mean_squared_error of the {name} Model is {MSE:.2f}')
    print(f'The r2_score of the {name} Model is {R2_score:.2f}')
    
    # Training and testing accuracy
    acc_train = model.score(X_train_sc, y_train) * 100
    acc_test = model.score(X_test_sc, y_test) * 100
    print(f'The accuracy of the {name} Model Train is {acc_train:.2f}')
    print(f'The accuracy of the {name} Model Test is {acc_test:.2f}')
    
    # Scatter plot for predictions vs actual values
    plt.scatter(y_test, y_pred, s=10, color='#3c7b9b')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{name} Evaluation')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', linewidth=4)
    plt.show()

```
## OUTPUT:
## Missing Data:

![missing data](https://github.com/user-attachments/assets/219bdef2-5cc4-4705-bff4-d9ea86658fa6)

## yield:

![yield](https://github.com/user-attachments/assets/94cc3734-ea67-4c8a-90a4-9e7a020a2e5b)


![yield 1](https://github.com/user-attachments/assets/28832958-4f97-4bd4-8008-ee4bff8b9c25)


![yield value](https://github.com/user-attachments/assets/bbb38662-af67-4c24-b925-91c108fc1852)

## Pesticide

![pesticides](https://github.com/user-attachments/assets/1264be8d-379b-4f7e-8225-a3a5bbedce5e)


## HeatMap

![heat  map](https://github.com/user-attachments/assets/37f13279-2b2e-4918-aeeb-c5bca5f79760)

# Scatter Plot:

![scatter matrix](https://github.com/user-attachments/assets/fce44e76-3cb0-4c4f-becd-76824ab2605d)

## Linear Regression;

![p2](https://github.com/user-attachments/assets/9d8d5013-6313-4d2f-ba2c-38f8c1d42427)

## Decisiom Tree

![p3](https://github.com/user-attachments/assets/5bd611a0-cb29-46c8-8693-a46449b2301d)

## Random Forest

![p4](https://github.com/user-attachments/assets/bd80e913-9e48-4239-9d4e-c07a8b3d6ac1)

## Results:
Thus AI-powered crop yield prediction using machine learning effectively analyzes environmental factors, resulting in improved accuracy and insights for farmers. This technology enables better decision-making, leading to enhanced productivity and sustainable agricultural practices.


