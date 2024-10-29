import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import time

def generate_larger_dataset(test_data_path, desired_rows):
    # Load the test data
    test_data = pd.read_csv(test_data_path, encoding='utf-8')

    # Get the current number of rows
    current_rows = test_data.shape[0]

    # If the current number of rows is greater than or equal to the desired number, just return the subset
    if current_rows >= desired_rows:
        return test_data.head(desired_rows)
    
    # Calculate how many times we need to repeat the data to reach the desired number of rows
    multiplier = (desired_rows // current_rows) + 1

    # Repeat the data
    larger_data = pd.concat([test_data] * multiplier, ignore_index=True)

    # Shuffle the data and select only the desired number of rows
    larger_data = larger_data.sample(n=desired_rows, random_state=42).reset_index(drop=True)

    return larger_data

def predict_new_data(model_path, original_data_path, test_data_path, gen_test_rows = None):
    
    # Load the original data (used to align columns)
    df = pd.read_csv(original_data_path, encoding='utf-8')
    df = df.drop(columns=['price', 'square_feet'])

    # Columns to be treated as categorical
    cols = ["bathrooms", "bedrooms", "has_photo", "cityname", "state", "source"]
    df[cols] = df[cols].astype('category')

    # One-hot encode categorical variables
    data_encoded = pd.get_dummies(df, columns=cols, drop_first=True)

    # Define features
    X = data_encoded.drop(['log1p_price'], axis=1)

    # Load new data for prediction
    if gen_test_rows:
        new_data = generate_larger_dataset(test_data_path, gen_test_rows)
    else:
        new_data = pd.read_csv(test_data_path, encoding='utf-8')
    new_data = new_data.drop(columns=['price', 'square_feet'])
        
    # Columns to be treated as categorical
    new_data[cols] = new_data[cols].astype('category')

    # One-hot encode the new data
    new_data_encoded = pd.get_dummies(new_data, columns=cols, drop_first=True)

    # Ensure new data columns align with the training data columns
    new_data_encoded = new_data_encoded.reindex(columns=X.columns, fill_value=0)

    # Standardize the features using the same scaler that was used for training
    scaler = StandardScaler()
    new_data_scaled = scaler.fit_transform(new_data_encoded)

    # List of model names to load and predict
    model_names = ["Ridge Regression", "Lasso Regression", "Decision Tree", 
                   "Random Forest", "Hist Gradient Boosting", "XGBoost"]
    
    predictions = {}
    prediction_times = {}

    # Predict with each model and measure time taken
    for model_name in model_names:
        # Load the saved model
        model = joblib.load(f"{model_path}/model_{model_name.replace(' ', '_')}.pkl")

        # Start timing the prediction process
        start_time = time.time()

        # Make predictions in log scale
        y_pred_log = model.predict(new_data_scaled)

        # End timing the prediction process
        time_taken = time.time() - start_time

        # Back-transform predictions to the original scale
        y_pred_original = np.expm1(y_pred_log)  # Apply exp(y_pred) - 1

        predictions[model_name] = y_pred_original
        prediction_times[model_name] = time_taken

    # Store predictions in a DataFrame for comparison
    predictions_df = pd.DataFrame(predictions)

    return predictions_df, prediction_times


# Define paths
base_path = 'models/base'
tuned_path = 'models/tuned'
original_data_path = 'data_modified/engineered_data.csv'
test_data_path = 'src/modeling/test_data.csv'

# Predict for both base and tuned models
base_predictions_df, base_prediction_times = predict_new_data(base_path, original_data_path, test_data_path, gen_test_rows=1000000)
tuned_predictions_df, tuned_prediction_times = predict_new_data(tuned_path, original_data_path, test_data_path, gen_test_rows=1000000)

# Combine base and tuned predictions for comparison
comparison_df = pd.DataFrame({
    'Base_Ridge': base_predictions_df['Ridge Regression'],
    'Tuned_Ridge': tuned_predictions_df['Ridge Regression'],
    'Base_Lasso': base_predictions_df['Lasso Regression'],
    'Tuned_Lasso': tuned_predictions_df['Lasso Regression'],
    'Base_DecisionTree': base_predictions_df['Decision Tree'],
    'Tuned_DecisionTree': tuned_predictions_df['Decision Tree'],
    'Base_RandomForest': base_predictions_df['Random Forest'],
    'Tuned_RandomForest': tuned_predictions_df['Random Forest'],
    'Base_HistGradientBoosting': base_predictions_df['Hist Gradient Boosting'],
    'Tuned_HistGradientBoosting': tuned_predictions_df['Hist Gradient Boosting'],
    'Base_XGBoost': base_predictions_df['XGBoost'],
    'Tuned_XGBoost': tuned_predictions_df['XGBoost']
})

# View combined predictions
print(comparison_df.head())

# Combine base and tuned prediction times for comparison
prediction_times_df = pd.DataFrame({
    'Model': base_prediction_times.keys(),
    'Base_Times': base_prediction_times.values(),
    'Tuned_Times': tuned_prediction_times.values()
})

# View prediction times
print("--Prediction times comparison (in seconds)--")
print(prediction_times_df)