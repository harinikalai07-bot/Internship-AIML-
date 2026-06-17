import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

data = {
    "Attendance": [60, 70, 80, 90, 95, 85, 75, 65, 88, 92],
    "Study_Hours": [2, 3, 4, 6, 8, 7, 5, 3, 7, 9]
}

df = pd.DataFrame(data)

print("Dataset")
print(df)

X = df[["Attendance", "Study_Hours"]]

linked = linkage(X, method="ward")

plt.figure(figsize=(8, 5))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Students")
plt.ylabel("Distance")
plt.show()

model = AgglomerativeClustering(n_clusters=2)

df["Cluster"] = model.fit_predict(X)

print("\nClustered Data")
print(df)