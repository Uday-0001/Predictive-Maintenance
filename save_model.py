import joblib
from xgboost import XGBClassifier

# Load the model from the notebook's pickle file (created by Jupyter)
model = joblib.load('.ipynb_checkpoints/Predictive-checkpoint.ipynb')

# Save the model in the format expected by the Streamlit app
joblib.dump(model, 'model.pkl')
print("Model saved as 'model.pkl'")