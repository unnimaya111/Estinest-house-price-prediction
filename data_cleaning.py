
import pandas as pd
from pathlib import Path

df = pd.read_csv("03_house_price_prediction.csv")

print("Original Dataset Shape:", df.shape)

df_clean = df.copy()

# Remove duplicate records
df_clean = df_clean.drop_duplicates()

# Standardize text columns
df_clean["Location"] = df_clean["Location"].str.strip()

df_clean["Location"] = df_clean["Location"].replace({
    "kochi": "Kochi",
    "BANGALORE": "Bengaluru",
    "Trivandrum": "Thiruvananthapuram"
})

df_clean["Property_Type"] = df_clean["Property_Type"].str.strip()

df_clean["Property_Type"] = df_clean["Property_Type"].replace({
    "apartment": "Apartment",
    "VILLA": "Villa"
})

df_clean["Furnishing"] = df_clean["Furnishing"].str.strip()

df_clean["Furnishing"] = df_clean["Furnishing"].replace({
    "Semi furnished": "Semi-Furnished",
    "un furnished": "Unfurnished"
})

# Treat zero property area as missing
df_clean.loc[df_clean["Area_sqft"] == 0, "Area_sqft"] = pd.NA

# Create data folder if it does not exist
Path("data").mkdir(exist_ok=True)

# Save cleaned dataset
df_clean.to_csv("data/cleaned_house_prices.csv", index=False)

print("\nData Cleaning Completed!")
print("Original Records:", len(df))
print("Cleaned Records:", len(df_clean))
print("Remaining Duplicates:", df_clean.duplicated().sum())

print("\nMissing Values After Cleaning:")
print(df_clean.isnull().sum())

print("\nCleaned Dataset Shape:", df_clean.shape)
print("\nCleaned dataset saved to data/cleaned_house_prices.csv")
