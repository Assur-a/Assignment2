import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine


wine = load_wine()

df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)


plt.figure(figsize=(15, 8))

df.boxplot()

plt.title("Boxplots of Wine Dataset")
plt.xlabel("Numerical Features")
plt.ylabel("Values")

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()
