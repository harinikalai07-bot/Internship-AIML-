import pandas as pd

data = {
    "Name": ["Harini", "Priya", "Kavi", "Ram", "Abi"],
    "Age": [21, 22, 20, 23, 24],
    "Marks": [85, 92, 78, 95, 88]
}

df = pd.DataFrame(data)

print("Dataset")
print(df)

print("\nDataset Info")
print(df.info())

print("\nFirst Rows")
print(df.head())

print("\nStatistics")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

print("\nAverage Marks")
print(df["Marks"].mean())

print("\nHighest Marks")
print(df["Marks"].max())