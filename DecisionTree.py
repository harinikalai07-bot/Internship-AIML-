import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = {
    "Attendance": [60, 70, 80, 90, 95, 85, 75, 65, 88, 92],
    "Study_Hours": [2, 3, 4, 6, 8, 7, 5, 3, 7, 9],
    "Pass": [0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

print(df)

X = df[["Attendance", "Study_Hours"]]
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

new_student = [[85, 6]]

prediction = model.predict(new_student)

print("\nPrediction:")

if prediction[0] == 1:
    print("Pass")
else:
    print("Fail")