import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# Load the data
df = pd.read_csv('ai4i2020 (1).csv')

# Show original column names
print("Original columns:", df.columns.tolist())

# Clean column names by removing brackets
df.columns = [col.replace('[', '_').replace(']', '').replace(' ', '_') for col in df.columns]

# Show cleaned column names
print("\nCleaned columns:", df.columns.tolist())

# Prepare features and target
feature_cols = [col for col in df.columns if any(x in col.lower() for x in ['temperature', 'speed', 'torque', 'wear'])]
print("\nSelected features:", feature_cols)

X = df[feature_cols]
y = df['Machine_failure']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple XGBoost model
model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model.pkl')
print("Model successfully saved as 'model.pkl'")