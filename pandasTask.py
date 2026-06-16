import pandas as pd

students = {
    "Name": ["Harini", "Priya", "Kavi", "Ram"],
    "Age": [21, 22, 20, 23],
    "Marks": [85, 92, 78, 95]
}

df = pd.DataFrame(students)

print("Student Data")
print(df)

print("\nAverage Marks:")
print(df["Marks"].mean())

print("\nHighest Marks:")
print(df["Marks"].max())

print("\nLowest Marks:")
print(df["Marks"].min())