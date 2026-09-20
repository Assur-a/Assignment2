import pandas as pd
from sklearn.datasets import load_wine

# Load dataset
wine = load_wine()

# Create DataFrame
df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

# Display dataset
print("First 5 Rows:")
print(df.head())

# Shape
print("\nDataset Shape:")
print(df.shape)

# Information
print("\nDataset Information:")
df.info()

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate Values
print("\nDuplicate Values:")
print(df.duplicated().sum())

# Target Information
print("\nTarget Names:")
print(wine.target_names)

print("\nTarget Distribution:")
print(pd.Series(wine.target).value_counts())