import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("student.csv")

# ---------------- EDA ----------------

print("===== DATASET =====")
print(df)

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== INFO =====")
print(df.info())

print("\n===== STATISTICS =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\nAverage Marks:", df["Marks"].mean())
print("Highest Marks:", df["Marks"].max())
print("Lowest Marks:", df["Marks"].min())

# ---------------- VISUALIZATION ----------------

# Bar Chart
plt.figure(figsize=(6,4))
plt.bar(df["Name"], df["Marks"])
plt.title("Student Marks")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.show()

# Line Chart
plt.figure(figsize=(6,4))
plt.plot(df["Name"], df["Marks"], marker="o")
plt.title("Marks Trend")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.show()

# Pie Chart
plt.figure(figsize=(6,6))
plt.pie(df["Marks"], labels=df["Name"], autopct="%1.1f%%")
plt.title("Marks Distribution")
plt.show()

# Histogram
plt.figure(figsize=(6,4))
plt.hist(df["Marks"], bins=5)
plt.title("Marks Histogram")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()

print("\nEDA and Visualization Completed Successfully!")