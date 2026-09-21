import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine


wine = load_wine()

df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

# Calculate correlation matrix
correlation_matrix = df.corr()

print("--- Correlation Matrix ---")
print(correlation_matrix)


plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Wine Dataset Correlation Heatmap")
plt.tight_layout()
plt.show()


corr = correlation_matrix.copy()

for i in range(len(corr.columns)):
    corr.iloc[i, i] = 0

max_corr = corr.stack().idxmax()
max_value = corr.stack().max()

print("\n--- Strongest Positive Correlation ---")
print("Feature 1:", max_corr[0])
print("Feature 2:", max_corr[1])
print("Correlation:", max_value)
