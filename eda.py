
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset

df = pd.read_csv("03_house_price_prediction.csv")

print("\nFirst 5 Rows")
print(df.head())

# 2. Dataset Information

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
df.info()

# 3. Missing Values

print("\nMissing Values")
print(df.isnull().sum())

# 4. Duplicate Records

print("\nDuplicate Records")
print(df.duplicated().sum())

# 5. Statistical Summary

print("\nStatistical Summary")
print(df.describe())

# 6. Zero Area Records

print("\nZero Area Records")
print((df["Area_sqft"] == 0).sum())

# 7. Categorical Value Counts

print("\nLocation Counts")
print(df["Location"].value_counts())

print("\nProperty Type Counts")
print(df["Property_Type"].value_counts())

print("\nFurnishing Counts")
print(df["Furnishing"].value_counts())

# 8. Correlation Analysis

print("\nCorrelation with House Price")

correlation = df.corr(numeric_only=True)["Price_INR"]

print(correlation.sort_values(ascending=False).round(3))

# 9. House Price Distribution

plt.figure(figsize=(10, 5))

sns.histplot(data=df, x="Price_INR", bins=30)

plt.title("Distribution of House Prices")
plt.xlabel("House Price (INR)")
plt.ylabel("Number of Properties")

plt.tight_layout()
plt.show()

# 10. Area vs House Price

plt.figure(figsize=(10, 5))

sns.scatterplot(
    data=df,
    x="Area_sqft",
    y="Price_INR",
    alpha=0.5
)

plt.title("Area vs House Price")
plt.xlabel("Area (Square Feet)")
plt.ylabel("House Price (INR)")

plt.tight_layout()
plt.show()

# 11. Correlation Heatmap

numeric_data = df.select_dtypes(include="number").drop(
    columns=["Property_ID"],
    errors="ignore"
)

plt.figure(figsize=(10, 7))

sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()

print("\nEDA Completed Successfully!")
