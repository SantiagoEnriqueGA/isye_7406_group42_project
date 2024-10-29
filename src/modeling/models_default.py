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

# Base models
# ------------------------------------------------------------------------------------
# Define models to test with n_jobs set to -1 for parallel processing where applicable
models = {
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, n_jobs=-1)
}

# Test each model
print("--Training Models with default parameters--")
results = {}
for name, model in models.items():
    print(f"    Training {name}")
    
    start_time = time.time()                        # Start timing
    model.fit(X_train, y_train)                     # Train the model
    time_taken = (time.time() - start_time) / 60    # End timing and calculate time taken (convert to minutes)
    
    joblib.dump(model, f"models/base/model_{name.replace(' ', '_')}.pkl")   # Save the trained model
    
    y_pred_log  = model.predict(X_test)                  # Make predictions
    
    # Back-transform predictions to the original price scale
    y_pred_original = np.expm1(y_pred_log)      # Apply exp(y_pred) - 1
    y_test_original = np.expm1(y_test)          # Apply exp(y_test) - 1 to the true values
    
    # Calculate metrics on the original scale
    r_squared = r2_score(y_test, y_pred_log)                            # Calculate R-squared
    mse = mean_squared_error(y_test_original, y_pred_original)          # Calculate MSE
    medae = median_absolute_error(y_test_original, y_pred_original)     # Calculate Median AE
    mae = mean_absolute_error(y_test_original, y_pred_original)         # Calculate MAE
    
    results[name] = (r_squared, mse, medae, mae, time_taken)            # Store results with the time taken

# Print the results
print("--Results for Models with default parameters--")
print(f"\n{'Model Name':<25} {'R-squared':<10} {'MSE':<15} {'MedAE':<15} {'MAE':<15} {'Time (min)':<15}")
print("-" * 100)

for model_name, metrics in results.items():
    r2, mse, medae, mae, time_taken = metrics
    print(f"{model_name:<25} {r2:<10.4f} {mse:<15.4f} {medae:<15.4f} {mae:<15.4f} {time_taken:<15.2f}")

# Store the results in a DataFrame and save to CSV
results_df = pd.DataFrame.from_dict(
    results,
    orient='index',
    columns=['R-squared', 'MSE', 'MedAE', 'MAE', 'Time (min)']
)

# Save to CSV
results_df.to_csv('models/base/model_performance_metrics.csv')
print("--Model performance metrics saved to 'model_performance_metrics.csv'--")