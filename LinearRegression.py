import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "Salary": [25000, 30000, 35000, 40000, 45000,
               50000, 55000, 60000, 65000, 70000,
               75000, 80000, 85000, 90000, 95000]
}

df = pd.DataFrame(data)

print("Dataset")
print(df)

X = df[["Experience"]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nPredicted Values")
print(y_pred)

print("\nMean Squared Error")
print(mean_squared_error(y_test, y_pred))

print("\nR2 Score")
print(r2_score(y_test, y_pred))

new_experience = [[16]]

predicted_salary = model.predict(new_experience)

print("\nPredicted Salary for 16 Years Experience")
print(predicted_salary[0])