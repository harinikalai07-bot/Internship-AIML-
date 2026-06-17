import pandas as pd

df = pd.read_csv("titanic.csv")

print("=" * 50)
print("ORIGINAL DATASET")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].mean())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

if "Cabin" in df.columns:
    df = df.drop("Cabin", axis=1)

df = df.drop_duplicates()

print("\n" + "=" * 50)
print("AFTER PREPROCESSING")
print("=" * 50)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

df.to_csv("cleaned_titanic.csv", index=False)

print("\nCleaned Dataset Saved Successfully!")
print("File Name: cleaned_titanic.csv")