import warnings
warnings.filterwarnings("ignore")

import time

import pandas as pd
import numpy as np
import joblib 

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error, mean_absolute_error

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from xgboost import XGBRegressor  

# Load the data
# ------------------------------------------------------------------------------------
df = pd.read_csv('data_modified/engineered_data.csv', encoding='utf-8')
df = df.drop(columns=['price', 'square_feet'])


# Columns to be treated as factor
cols = ["bathrooms", "bedrooms", "has_photo", "cityname", "state", "source"]
df[cols] = df[cols].astype('category')

# One-hot encode categorical variables
data_encoded = pd.get_dummies(df, columns=cols, drop_first=True)

# Define features and target variable
X = data_encoded.drop(['log1p_price'], axis=1)
y = data_encoded['log1p_price'] 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models to Tune
# ------------------------------------------------------------------------------------
# Define models to test with hyperparameter tuning
models = {
    "Ridge Regression": (Ridge(), {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'max_iter': [1000,10000] 
    }),
    "Lasso Regression": (Lasso(), {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'max_iter': [1000,10000] 
    }),
    "Decision Tree": (DecisionTreeRegressor(random_state=42), {
        'max_depth': [None, 5, 15, 25],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 5]
    }),
    "Random Forest": (RandomForestRegressor(random_state=42, n_jobs=-1), {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }),
    "Hist Gradient Boosting": (HistGradientBoostingRegressor(random_state=42), {
        'max_iter': [100, 200, 500],
        'learning_rate': [0.001, 0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9]
    }),
    "XGBoost": (XGBRegressor(random_state=42, n_jobs=-1), {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 5, 15],
        'learning_rate': [0.001, 0.01, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    })
}

# Test each model with hyperparameter tuning
print("--Training Models with Hyperparameter Tuning--")
results = {}
best_models = {}
best_params = {}
for name, (model, params) in models.items():
    print(f"    Tuning {name}")
    
    start_time = time.time()        # Start timing
    
    # Perform grid search
    grid_search = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    time_taken = (time.time() - start_time) / 60  # End timing and calculate time taken (convert to minutes)
    
    best_model = grid_search.best_estimator_    # Get the best model
    best_models[name] = best_model
    best_params[name] = grid_search.best_params_

    
    # joblib.dump(best_model, f"models/tuned/model_{name.replace(' ', '_')}.pkl")  # Save the best model
    
    y_pred_log  = best_model.predict(X_test)                  # Make predictions
    
    # Back-transform predictions to the original price scale
    y_pred_original = np.expm1(y_pred_log)      # Apply exp(y_pred) - 1
    y_test_original = np.expm1(y_test)          # Apply exp(y_test) - 1 to the true values
    
    # Calculate metrics on the original scale
    r_squared = r2_score(y_test, y_pred_log)                            # Calculate R-squared
    mse = mean_squared_error(y_test_original, y_pred_original)          # Calculate MSE
    medae = median_absolute_error(y_test_original, y_pred_original)     # Calculate Median AE
    mae = mean_absolute_error(y_test_original, y_pred_original)         # Calculate MAE
    
    results[name] = (r_squared, mse, medae, mae, time_taken)    # Store results with the time taken

# Print the results
print("--Results for Models with tuned parameters--")
print(f"\n{'Model Name':<25} {'R-squared':<10} {'MSE':<15} {'MedAE':<15} {'MAE':<15} {'Time (min)':<15}")
print("-" * 100)

for model_name, metrics in results.items():
    r2, mse, medae, mae, time_taken = metrics
    print(f"{model_name:<25} {r2:<10.4f} {mse:<15.4f} {medae:<15.4f} {mae:<15.4f} {time_taken:<15.2f}")
    
