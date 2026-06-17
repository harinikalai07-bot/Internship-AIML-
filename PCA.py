import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = {
    "Attendance": [60, 70, 80, 90, 95, 85, 75, 65, 88, 92],
    "Study_Hours": [2, 3, 4, 6, 8, 7, 5, 3, 7, 9],
    "Assignments": [3, 4, 5, 6, 8, 7, 5, 4, 8, 9],
    "Marks": [50, 60, 70, 80, 90, 85, 75, 65, 88, 92]
}

df = pd.DataFrame(data)

print("Original Dataset")
print(df)

scaled_data = StandardScaler().fit_transform(df)

pca = PCA(n_components=2)

pca_data = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(
    pca_data,
    columns=["Principal_Component_1", "Principal_Component_2"]
)

print("\nPCA Reduced Dataset")
print(pca_df)

print("\nExplained Variance Ratio")
print(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.scatter(
    pca_df["Principal_Component_1"],
    pca_df["Principal_Component_2"]
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization")
plt.grid(True)

plt.show()