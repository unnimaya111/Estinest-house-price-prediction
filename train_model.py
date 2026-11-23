
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from preprocessing import create_preprocessor


# Load cleaned dataset
df = pd.read_csv("data/cleaned_house_prices.csv")

# Separate features and target
X = df.drop(columns=["Price_INR", "Property_ID"])
y = df["Price_INR"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Define three machine learning models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

# Create models folder
Path("models").mkdir(exist_ok=True)

# Train and save each model
for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    filename = name.lower().replace(" ", "_") + ".joblib"

    joblib.dump(pipeline, f"models/{filename}")

    print(f"{name} trained and saved successfully!")

print("\nAll three models have been trained successfully!")

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))
