import pandas as pd
from sklearn.datasets import load_wine


wine = load_wine()


df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)


print("First 5 Rows:")
print(df.head())


print("\nDataset Shape:")
print(df.shape)


print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


print("\nMissing Values:")
print(df.isnull().sum())


print("\nDuplicate Values:")
print(df.duplicated().sum())


print("\nTarget Names:")
print(wine.target_names)

print("\nTarget Distribution:")
print(pd.Series(wine.target).value_counts())
