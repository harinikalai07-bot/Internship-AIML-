import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    "Attendance": [60, 70, 80, 90, 95, 85, 75, 65, 88, 92],
    "Study_Hours": [2, 3, 4, 6, 8, 7, 5, 3, 7, 9]
}

df = pd.DataFrame(data)

print("Dataset")
print(df)

X = df[["Attendance", "Study_Hours"]]

model = KMeans(n_clusters=2, random_state=42)

df["Cluster"] = model.fit_predict(X)

print("\nClustered Data")
print(df)

print("\nCluster Centers")
print(model.cluster_centers_)

plt.scatter(df["Attendance"], df["Study_Hours"], c=df["Cluster"])

plt.xlabel("Attendance")
plt.ylabel("Study Hours")
plt.title("K-Means Clustering")

plt.show()