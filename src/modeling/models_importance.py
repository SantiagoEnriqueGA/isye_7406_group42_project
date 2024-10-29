import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# Function to plot feature importance
def plot_feature_importance(importances, model_name, feature_names, top_n=20):
    # Get indices of top n features
    indices = np.argsort(importances)[-top_n:]
    
    # Plot feature importances
    plt.figure(figsize=(10, 8))
    plt.title(f"Top {top_n} Feature Importances for {model_name}", fontsize=15)
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='viridis')
    plt.xlabel("Feature Importance")
    plt.show()

# Load the data
# ------------------------------------------------------------------------------------
df = pd.read_csv('data_modified/engineered_data.csv', encoding='utf-8')
df = df.drop(columns=['price', 'square_feet'])

# Columns that were treated as categorical and one-hot encoded
cols = ["bathrooms", "bedrooms", "has_photo", "cityname", "state", "source"]
df[cols] = df[cols].astype('category')

# One-hot encode categorical variables
data_encoded = pd.get_dummies(df, columns=cols, drop_first=True)

# Define features and target variable
X = data_encoded.drop(['log1p_price'], axis=1)
y = np.log1p(data_encoded['log1p_price'])    

# Feature names after encoding
feature_names = X.columns

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# List of models to evaluate
# ------------------------------------------------------------------------------------
model_names = ["Ridge_Regression", "Lasso_Regression", "Decision_Tree", 
               "Random_Forest", "XGBoost", "Hist_Gradient_Boosting"] 

# model_source = 'models/base'
model_source = 'models/tuned'

# Initialize a DataFrame to store feature importances
feature_importance_df = pd.DataFrame(index=feature_names)

# Loop through each model and calculate feature importance
for model_name in model_names:
    print(f"-- Getting feature importances for {model_name}")
    
    model = joblib.load(model_source + f"/model_{model_name}.pkl")     # Load the trained model
    
    # Compute permutation-based feature importance for Hist Gradient Boosting
    # "randomly shuffling the values of a single feature and observing the resulting degradation of the model’s score"
    if model_name == "Hist_Gradient_Boosting":
        print(f"Computing permutation importance for {model_name}...")
        result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, scoring='neg_mean_squared_error', n_jobs=-1)
        importances = result.importances_mean
    
    # Tree-based models (Decision Tree, Random Forest, XGBoost)
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    
    # Coefficient-based models (Ridge, Lasso)
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)  # Coefficients can be negative, so we take absolute values
    
    # Skip models without feature importance
    else:
        print(f"Model {model_name} does not support feature importance.")
        continue
    
    
    feature_importance_df[model_name] = importances                     # Add the importances as a new column in the DataFrame
    # plot_feature_importance(importances, model_name, feature_names)     # Plot the top 20 feature importances

# Save the feature importance results to a CSV
feature_importance_df.to_csv(model_source + '/feature_importance_results.csv')

print("\n-- Feature importances saved to "+model_source+"/feature_importance_results.csv --")
