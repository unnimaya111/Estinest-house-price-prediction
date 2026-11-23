
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load cleaned dataset
df = pd.read_csv("data/cleaned_house_prices.csv")

# Separate features and target
X = df.drop(columns=["Price_INR", "Property_ID"])
y = df["Price_INR"]

# Recreate the same testing split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model file names
model_files = {
    "Linear Regression": "models/linear_regression.joblib",
    "Random Forest": "models/random_forest.joblib",
    "Gradient Boosting": "models/gradient_boosting.joblib"
}

results = []

# Evaluate each trained model
for name, filename in model_files.items():

    print(f"\nEvaluating {name}...")

    model = joblib.load(filename)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

# Create comparison table
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
)

print("\nMODEL EVALUATION RESULTS")
print(results_df.to_string(index=False))

# Save evaluation results
results_df.to_csv("model_comparison.csv", index=False)

# Select the model with the lowest RMSE
best_model_name = results_df.iloc[0]["Model"]

best_model_path = model_files[best_model_name]

best_model = joblib.load(best_model_path)

# Save the best model
joblib.dump(best_model, "models/best_model.joblib")

print("\nBest Model:", best_model_name)
print("Best Model RMSE:", results_df.iloc[0]["RMSE"])
print("Best Model R2 Score:", results_df.iloc[0]["R2 Score"])

print("\nBest model saved as models/best_model.joblib")
print("Evaluation results saved as model_comparison.csv")
